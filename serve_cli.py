import sys,os
import argparse
import torch

from main import load_pretrained_model, SeparatorStyle, conv_templates, process_anyres_image, tokenizer_image_token, \
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from common import WORKER_HEART_BEAT_INTERVAL, build_logger, server_error_msg, pretty_print_semaphore

import requests
from PIL import Image
import io 
from transformers import AutoTokenizer, AutoConfig
from transformers import TextStreamer, StoppingCriteriaList, MaxLengthCriteria


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def main(args):
    # Model
    disable_torch_init()

    if args.conv_version in conv_templates:
        conv = conv_templates[args.conv_version]
    else:
        raise ValueError("wrong conv_version")

    context_len = 2048
    tokenizer, model, image_processor = load_pretrained_model(args.model_path, args.cache_dir, args.conv_version, \
            args.load_8bit, args.load_4bit, device=args.device, attn_implementation=args.attn_implementation)
    print('generation_config', model.generation_config)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(conv.stop_str)
    ]

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    messages = [{'role': 'system', 'content': conv.system}]

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py 
    image_tensor = process_anyres_image(image, image_processor, model.config.image_grid_pinpoints)
    print('image_tensor', image_tensor.shape)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{conv.roles[1]}: ", end="")

        if image is not None:
            # first message
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        

        messages.append({'role': conv.roles[0], 'content': inp})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        print('input_ids', input_ids.shape)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators, no_repeat_ngram_size=2,
                streamer=streamer,
                use_cache=True) 

        outputs = tokenizer.decode(output_ids[0]).strip()
        messages.append({'role': conv.roles[1], 'content': outputs})

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs, "output_ids": output_ids[0]}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, default="./examples/waterview.jpg")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="/Users/zhongz2/down/finetune_llama_3_1_with_pretrain")
    parser.add_argument("--model_name", type=str, default="llava_llama_3_1")
    parser.add_argument("--cache_dir", type=str, default="/Users/zhongz2/down/cache_dir")
    parser.add_argument("--conv_version", type=str, default="llama_3_1", choices=['llama_3_1', 'gemma_2', 'qwen_2'])
    parser.add_argument("--device", type=str, default="mps", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    args = parser.parse_args()
    main(args)
