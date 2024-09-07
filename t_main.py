
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
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import Gemma2Config, Gemma2ForCausalLM
from transformers import Qwen2Config, Qwen2ForCausalLM

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers.cache_utils import HybridCache
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer

USE_TRANSFORMERS_TRAINER = True
if USE_TRANSFORMERS_TRAINER:
    from transformers.trainer import has_length, is_sagemaker_mp_enabled, \
        PreTrainedModel, TRAINING_ARGS_NAME, \
        SAFE_WEIGHTS_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME
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
    MPT = auto()
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

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.LLAMA_3_1:
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})
            ret = chat_template_messages

        elif self.sep_style == SeparatorStyle.QWEN_2:
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})
            ret = chat_template_messages

        elif self.sep_style == SeparatorStyle.CHATGLM_4:
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})
            ret = chat_template_messages

        elif self.sep_style == SeparatorStyle.GEMMA_2:
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})
            ret = chat_template_messages

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format="PNG"):
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = io.BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        images.append(img)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = "<image>\n" + msg.replace("<image>", "").strip()
                    else:
                        msg = re.sub(r"(<image>)\n(?=<image>)", r"\1 ", msg)
                    for img in image:
                        img_b64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}"/>'
                        msg = msg.replace("<image>", img_str, 1).strip()
                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2
        }

conv_templates = {
    'plain': Conversation(
        system="",
        roles=("", ""),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.PLAIN,
        sep="\n",
    ),
    'llama_3': Conversation(
        system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
        roles=("user", "assistant"),
        version="llama_3",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_3,
        stop_token_ids=[128009],
        sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
        sep2='<|start_header_id|>user<|end_header_id|>\n\n',
        stop_str='<|eot_id|>'
    ),
    'llama_3_1': Conversation(
        system="You are a pirate chatbot who always responds in pirate speak!",
        roles=("user", "assistant"),
        version="llama_3_1",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_3_1,
        stop_token_ids=[128009],
        sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
        sep2='<|start_header_id|>user<|end_header_id|>\n\n',
        stop_str='<|eot_id|>'
    ), 
    'gemma_2': Conversation(
        system="",
        roles=("user", "model"),
        version="gemma_2",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.GEMMA_2,
        stop_token_ids=None,
        sep='\n<start_of_turn>model\n',
        sep2='\n<start_of_turn>user\n',
        stop_str='<end_of_turn>'
    ), 
    'qwen_2': Conversation(
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        version="qwen_2",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.QWEN_2,
        stop_token_ids=None,
        sep='\n<|im_start|>assistant\n',
        sep2='\n<|im_start|>user\n',
        stop_str='<|im_end|>'
    ),
    'chatglm_4': Conversation(
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        version="chatglm_4",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.CHATGLM_4,
        stop_token_ids=None,
        sep='<|assistant|>\n',
        sep2='<|user|>\n'
    ),
}
default_conversation = conv_templates['llama_3_1']


@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_crop_resolution: int = 224
    image_grid_pinpoints: Optional[str] = field(default="[]")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    conv_version: Optional[str] = field(default="plain")
    pretrain_ckpt_path: Optional[str] = field(default=None)
    vision_tower_name_or_path: Optional[str] = field(default="openai/clip-vit-large-patch14-336")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    group_by_modality_length: bool = field(default=False)    
    remove_unused_columns: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


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



def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    # if isinstance(grid_pinpoints, str):
    #     assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
    #     grid_pinpoints = grid_pinpoints.replace(" ", "").replace("x", ",")[1:-1].split("),(")
    #     grid_pinpoints = [[int(x) * patch_size for x in item.split(",")] for item in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'] if "crop_size" in processor.__dict__ else processor.size["height"])

    shortest_edge = processor.size['shortest_edge'] if "shortest_edge" in processor.size else processor.size["height"]
    image_original_resize = image.resize((shortest_edge, shortest_edge))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)




def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class DebugLlavaConfig(LlamaConfig):
    model_type = "debug_llava"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    rope_scaling: Optional[dict] = {}


class DebugLlavaForCausalLM(LlamaForCausalLM):
    config_class = DebugLlavaConfig

    def __init__(self, config):
        super().__init__(config) 

    def initialize_vision_modules(self, device="auto", dtype=torch.bfloat16, mm_projector_type="mlp2x_gelu", pretrain_ckpt_path=None):
        vision_tower_name_or_path = 'openai/clip-vit-large-patch14-336'
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name_or_path).to(device=device, dtype=dtype)

        if mm_projector_type == 'linear':
            self.mm_projector = nn.Linear(self.vision_tower.config.hidden_size, self.config.hidden_size, device=device, dtype=dtype)
        elif mm_projector_type == 'mlp2x_gelu':
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.vision_tower.config.hidden_size, self.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.mm_projector = nn.Sequential(*modules)

        if pretrain_ckpt_path is not None:
            print('loading pretrain ckpt path for mm_projector')
            self.mm_projector.load_state_dict(torch.load(pretrain_ckpt_path), strict=False)
        self.mm_projector.to(device=device, dtype=dtype)

        self.num_patches_per_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size

        embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=dtype))
        self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=dtype, device=device) * embed_std)

    def encode_images(self, images):
        vision_tower_outputs = self.vision_tower(images, output_hidden_states=True) 
        image_features = vision_tower_outputs.hidden_states[-2][:, 1:]
        image_features = self.mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:

            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)

                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    
                    if self.config.image_aspect_ratio == "anyres":
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.vision_tower.config.image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    # spatial_unpad
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

                new_image_features.append(image_feature)

            image_features = new_image_features

        else:
            image_features = self.encode_images(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask  FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.model.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        print('FINAL0 position_ids', position_ids.shape if position_ids is not None else 'None')
        print('FINAL0 attention_mask', attention_mask.shape if attention_mask is not None else 'None')
        print('FINAL0 past_key_values', past_key_values.shape if past_key_values is not None else 'None')
        print('FINAL0 new_input_embeds', new_input_embeds.shape if new_input_embeds is not None else 'None')
        print('FINAL0 new_labels_padded', new_labels_padded.shape if new_labels_padded is not None else 'None')

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        print('FINAL position_ids', position_ids.shape if position_ids is not None else 'None')
        print('FINAL attention_mask', attention_mask.shape if attention_mask is not None else 'None')
        print('FINAL past_key_values', past_key_values.shape if past_key_values is not None else 'None')
        print('FINAL new_input_embeds', new_input_embeds.shape if new_input_embeds is not None else 'None')
        print('FINAL new_labels', new_labels.shape if new_labels is not None else 'None')
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.model.embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("debug_llava", DebugLlavaConfig)
AutoModelForCausalLM.register(DebugLlavaConfig, DebugLlavaForCausalLM)



def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    # Load the index
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    with open(safe_index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    for shard_file in shard_files:
        state_dict = safe_load_file(os.path.join(folder, shard_file))
        if 'model.embed_tokens.weight' in state_dict:
            state_dict['model.lm_head.weight'] = state_dict['model.embed_tokens.weight'].clone()
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

def load_pretrained_model(model_path, cache_dir, conv_version, load_8bit, load_4bit, device=None, attn_implementation='eager'):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    device = torch.device("cpu") if device is None else device
    kwargs = {
        "device_map": "cpu" if device is None else device,
        "attn_implementation": attn_implementation,
        "cache_dir": cache_dir
    }

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        # kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.float16

    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    model = DebugLlavaForCausalLM.from_pretrained(model_path, config=cfg_pretrained, **kwargs)
    
    model.initialize_vision_modules(device=device, dtype=torch.float16)
    model.to(device)
    model.eval()
    load_sharded_checkpoint(model, model_path)
    model.to(torch.float16)

    if True:
        import intel_extension_for_pytorch as ipex
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=torch.quint4x2, # or torch.qint8
            lowp_mode=ipex.quantization.WoqLowpMode.INT8, # or FP16, BF16, INT8
        )
        checkpoint = None
        model_ipex = ipex.llm.optimize(model, quantization_config=qconfig, low_precision_checkpoint=checkpoint)
        return tokenizer, model_ipex, model.image_processor 

    return tokenizer, model, model.image_processor 

