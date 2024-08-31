

from typing import List, Optional, Tuple, Union, Any, Dict, Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
import os
import copy
import json
import base64
import io
import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import pathlib
import ast
import math
import re
import random
import shortuuid
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import Gemma2Config, Gemma2ForCausalLM
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

USE_TRANSFORMERS_TRAINER = True
if USE_TRANSFORMERS_TRAINER:
    from transformers.trainer import has_length, is_sagemaker_mp_enabled, \
        PreTrainedModel, TRAINING_ARGS_NAME, \
        SAFE_WEIGHTS_NAME, WEIGHTS_NAME
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
else: # using deepspeed (out of memory)
    from torch.utils.data import DataLoader
    import deepspeed
    from transformers import AdamW, get_scheduler

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    PLAIN = auto()
    LLAMA_3 = auto()
    LLAMA_3_1 = auto()
    GEMMA_2 = auto()
    QWEN_2 = auto()
    CHATGLM_4 = auto()


@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_multimodal(
    sources: Sequence[str],
) -> Dict:

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

    return sources


data_path = ['/Users/zhongz2/down/llava_image_tune_cleaned_debug1.json']
image_folder = '/Users/zhongz2/down/'
root = '/lscratch/' + os.environ['SLURM_JOB_ID']
data_path = [os.path.join(root, 'train_json', 'llava_image_tune_cleaned_debug1.json')]
image_folder = root
list_data_dict = []
for data in data_path:
    data = json.load(open(data, "r"))
    for i in data:
        i['id'] = len(list_data_dict)
        list_data_dict.append(i)

i = 0
sources = list_data_dict[i]
if isinstance(i, int):
    sources = [sources]
sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))


cache_dir = '/Users/zhongz2/down/cache_dir'
cache_dir = '/data/zhongz2/data/cache_dir'
model_max_length = 8192

if False:
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.unk_token = "<|reserved_special_token_0|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    conv_llama_3 = Conversation(
        system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
        roles=("user", "assistant"),
        version="llama_3",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_3,
        stop_token_ids=[128009],
        sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
        sep2='<|start_header_id|>user<|end_header_id|>\n\n'
    )

    # For llama3
    has_image = True
    conversation = conv_llama_3
    if True:
        conv = conversation
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            messages = [{'role': 'system', 'content': conv.system}]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                messages.append({'role': role, 'content': sentence["value"]})
            conversations.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ))

        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets 
        # sep = conv.sep + conv.roles[1] + ": "
        for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cur_len = 0 
            target[:] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                
                parts = rou.split(conv.sep2)
                rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer))  # if add_generation_prompt=True
                # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
                if i!=0:
                    rou_len -= 1
                else:
                    cur_len += rou_len
                    continue

                ans_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
                target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

                cur_len += rou_len    

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                    
        if tokenizer.bos_token_id is not None and input_ids[0][0] != tokenizer.bos_token_id:
            input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
            targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]


# llama-3.1
if False:
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.unk_token = "<|reserved_special_token_0|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    conv_llama_3_1 = Conversation(
        system="You are a pirate chatbot who always responds in pirate speak!",
        roles=("user", "assistant"),
        version="llama_3_1",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_3_1,
        stop_token_ids=[128009],
        sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
        sep2='<|start_header_id|>user<|end_header_id|>\n\n'
    )

    # For llama3
    has_image = True
    conversation = conv_llama_3_1
    if True:
        conv = conversation
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            messages = [{'role': 'system', 'content': conv.system}]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                messages.append({'role': role, 'content': sentence["value"]})
            conversations.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ))

        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets 
        # sep = conv.sep + conv.roles[1] + ": "
        for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cur_len = 0 
            target[:] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                
                parts = rou.split(conv.sep2)
                rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer))  # if add_generation_prompt=True
                # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
                if i!=0:
                    rou_len -= 1
                    pass
                else:
                    cur_len += rou_len
                    continue

                ans_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
                target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

                cur_len += rou_len    

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                    
        if input_ids[0][0] != tokenizer.bos_token_id:
            input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
            targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]






if False:
    conv_gemma_2 = Conversation(
        system="",
        roles=("user", "model"),
        version="gemma_2",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.GEMMA_2,
        stop_token_ids=None,
        sep='\n<start_of_turn>model\n',
        sep2='\n<start_of_turn>user\n'
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", cache_dir=cache_dir)
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", cache_dir=cache_dir)
    prompt = "What is your favorite condiment?"
    tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )

    # For gemma2
    has_image = True
    conversation = conv_gemma_2
    if True:
        conv = conversation
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            # messages = [{'role': 'system', 'content': conv.system}]
            messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                messages.append({'role': role, 'content': sentence["value"]})
            conversations.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ))

        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets 
        for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cur_len = 0 
            target[:] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                
                parts = rou.split(conv.sep2)
                rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer))  # if add_generation_prompt=True
                # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
                if i!=0:
                    rou_len -= 1
                else:
                    cur_len += rou_len
                    continue

                ans_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
                target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

                cur_len += rou_len    

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                    
        if input_ids[0][0] != tokenizer.bos_token_id:
            input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
            targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]

# Qwen2
if False:
    model_name_or_path = "Qwen/Qwen2-7B-Instruct"
    model_name_or_path = "Qwen/Qwen2-7B"
    conv_qwen_2 = Conversation(
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        version="qwen_2",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.QWEN_2,
        stop_token_ids=None,
        sep='\n<|im_start|>assistant\n',
        sep2='\n<|im_start|>user\n'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    # tokenizer.bos_token = '<|im_start|>'
    # if tokenizer.unk_token is None:
    #     tokenizer.unk_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", cache_dir=cache_dir)
    prompt = "What is your favorite condiment?"
    tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': conv_qwen_2.system},
                    {'role': 'user', 'content': prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )

    # For gemma2
    has_image = True
    conv = conv_qwen_2
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        messages = [{'role': 'system', 'content': conv.system}]
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            messages.append({'role': role, 'content': sentence["value"]})
        conversations.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    # Mask targets
    for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        cur_len = 0 
        target[:] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            
            parts = rou.split(conv.sep2)
            rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer))
            # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
            if i!=0:
                # rou_len -= 1
                pass
            else:
                cur_len += rou_len
                continue

            ans_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

            cur_len += rou_len    

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    if tokenizer.bos_token_id is not None and input_ids[0][0] != tokenizer.bos_token_id:
        input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
        targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]



# ChatGLM-4
if True:
    model_name_or_path = "Qwen/Qwen2-7B-Instruct"
    model_name_or_path = "Qwen/Qwen2-7B"
    model_name_or_path = "THUDM/glm-4-9b-chat"
    conv_glm_4 = Conversation(
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        version="glm_4",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.CHATGLM_4,
        stop_token_ids=None,
        sep='<|assistant|>\n',
        sep2='<|user|>\n'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    # tokenizer.bos_token = '<|im_start|>'
    # if tokenizer.unk_token is None:
    #     tokenizer.unk_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", cache_dir=cache_dir)
    prompt = "What is your favorite condiment?"
    tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': conv_glm_4.system},
                    {'role': 'user', 'content': prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )

    inputs = tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': conv_glm_4.system},
                    {'role': 'user', 'content': prompt}],
                                        add_generation_prompt=True,
                                        tokenize=False
                                        )

    device = "cuda"
    query = "你好"
    inputs = tokenizer.apply_chat_template([
        {'role': 'system', 'content': conv_glm_4.system},
        {"role": "user", "content": query}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        )

    inputs = inputs.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to(device).eval()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=False))
   
    # For gemma2
    has_image = True
    conv = conv_glm_4
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        messages = [{'role': 'system', 'content': conv.system}]
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            messages.append({'role': role, 'content': sentence["value"]})
        conversations.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            ))
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    # Mask targets
    for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # rounds = conversation.split(conv.sep)
        # cur_len = 0 
        # target[:] = IGNORE_INDEX
        # for i, rou in enumerate(rounds):
        #     if rou == "":
        #         break
            
        #     parts = rou.split(conv.sep2)
        #     rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer)) - 2
        #     # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
        #     if i!=0:
        #         # rou_len -= 2
        #         pass
        #     else:
        #         cur_len += rou_len
        #         continue

        #     ans_len = len(tokenizer_image_token(parts[0], tokenizer))
        #     target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

        #     cur_len += rou_len    

        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         print(
        #             f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #             f" (ignored)"
        #         )
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        # target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            print(i, rou)
            if rou == "":
                break
            if i==0:
                round_len = len(tokenizer_image_token(rou+conv.sep2, tokenizer))
                cur_len += round_len
                target[:cur_len] = IGNORE_INDEX
                continue

            parts = rou.split(conv.sep)
            if len(parts) != 2:
                break
            parts[0] += conv.sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len - 2 : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        cur_len -= 2
        # target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    if tokenizer.bos_token_id is not None and input_ids[0][0] != tokenizer.bos_token_id:
        input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
        targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]




if True:
    import numpy as np
    from PIL import Image
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from timm.models.vision_transformer import VisionTransformer

    vision_tower_name_or_ckpt = "openai/clip-vit-large-patch14-336"
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_ckpt)
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name_or_ckpt)

    vision_tower_name_or_ckpt = '/data/zhongz2/temp29/debug/vinid_plip/'
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_ckpt)
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name_or_ckpt)

    images = [Image.fromarray(np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8))]
    inputs = image_processor(images, return_tensors="pt")['pixel_values']
    vision_tower_outputs = vision_tower(inputs, output_hidden_states=True) 
    image_features = vision_tower_outputs.hidden_states[-2][:, 1:]  # 1x576x1024

    trunk = VisionTransformer(embed_dim=768, 
                              depth=12, 
                              num_heads=12, 
                              mlp_ratio=4,
                              img_size=448, 
                              patch_size=16,
                              num_classes=0,
                              dynamic_img_size=True)
    CONCN_weight_file = '/data/zhongz2/temp29/debug/CONCH_weights_pytorch_model.bin'
    state_dict = torch.load(CONCN_weight_file, weights_only=True)
    vision_params = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if 'visual.trunk' in k}
    trunk.load_state_dict(vision_params)

    from conch.open_clip_custom import create_vision_model_from_pretrained
    model, preprocess = create_vision_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/data/zhongz2/temp29/debug/CONCH_weights_pytorch_model.bin")
    inputs = torch.stack([preprocess(image) for image in images])
    a,b = model(inputs)

    processor = CoCaImageProcessor()
    inputs = processor(images, return_tensors="pt")['pixel_values']


    from PIL import Image
    import requests
    from transformers import AutoProcessor, AutoModel
    import torch

    cache_dir = '/data/zhongz2/data/cache_dir'
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=cache_dir)
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    texts = ["a photo of 2 cats", "a photo of 2 dogs"]
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image) # these are the probabilities
    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")





