# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""
"""Fine-tuning protection script for Stable Diffusion for text2image with support for LoRA."""
from PIL import Image
import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import torch.nn as nn

# for protector
from clipstyler import fast_stylenet
from clipstyler.trans import *
import clip
from protection import SemanticProtection
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_name, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_name}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

access_token = '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'

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    ######################################################################################
    ######################################################################################
    ######################################################################################
    parser.add_argument('--style_vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--name', default='none',
                        help='name')

    parser.add_argument('--style_load', type=bool, default=False)
    parser.add_argument('--style_train', type=bool, default=False)
    parser.add_argument('--style_path', type=str, default='')
    parser.add_argument('--style_lr', type=float, default=1e-4)
    parser.add_argument('--style_lr_decay', type=float, default=5e-5)
    # parser.add_argument('--style_max_iter', type=int, default=1)
    parser.add_argument('--style_max_iter', type=int, default=5)
    parser.add_argument('--style_batch_size', type=int, default=4)
    parser.add_argument('--style_content_weight', type=float, default=1.0)
    parser.add_argument('--style_clip_weight', type=float, default=50.0)
    # parser.add_argument('--style_clip_weight', type=float, default=10.0)
    parser.add_argument('--style_tv_weight', type=float, default=1e-4)
    parser.add_argument('--style_glob_weight', type=float, default=50.0)
    # parser.add_argument('--style_glob_weight', type=float, default=1.0)
    parser.add_argument('--style_n_threads', type=int, default=16)
    parser.add_argument('--style_num_test', type=int, default=16)
    parser.add_argument('--style_save_model_interval', type=int, default=200)
    parser.add_argument('--style_save_img_interval', type=int, default=100)
    parser.add_argument('--style_crop_size', type=int, default=224)
    parser.add_argument('--style_thresh', type=float, default=0.7)
    parser.add_argument('--style_decoder', type=str, default='./models/decoder.pth')

    parser.add_argument('--style_recon_weight', type=float, default=0)
    parser.add_argument('--style_semantic_weight', type=float, default=0)
    # parser.add_argument('--style_recon_weight', type=float, default=10)
    # parser.add_argument('--style_semantic_weight', type=float, default=1)
    #
    # parser.add_argument('--style_mix_ratio', type=float, default=0.7)
    parser.add_argument('--style_mix_ratio', type=float, default=0.5)
    parser.add_argument('--style_mix_strength', type=float, default=0.8)
    parser.add_argument('--style_maintain_strength', type=float, default=0.3)

    parser.add_argument('--test_use_downsample', type=bool, default=False)

    ######################################################################################
    ######################################################################################
    ######################################################################################


    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def main(args):

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo_name = create_repo(repo_name, exist_ok=True)
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(f'{args.output_dir}/prot_images', exist_ok=True)
            os.makedirs(f'{args.output_dir}/prot_images/finals', exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-amp ere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    #######################################################################################
    # we looking forward a noise for raw images that lead to zero optimization for lora
    style_network, style_optimizer, semantic_network, semantic_optimizer, \
    clip_model, preprocess, content_tf, hr_tf, test_tf, augment_trans = prepare_protection_components(args, device=accelerator.device)
    crop_num = 16

    source = "a drawing by Van Gogh"
    target = 'Fire'

    # target = 'a bad watercolor drawing'
    # target = 'a drawing by Salvador Dali'
    # target = 'a oil painting by Van Gogh'
    print(target)
    with torch.no_grad():
        template_target = compose_text_with_templates(target, imagenet_templates)
        tokens_target = clip.tokenize(template_target).to(accelerator.device)
        text_target_ = clip_model.encode_text(tokens_target).detach()
        text_target_ = text_target_.mean(axis=0, keepdim=True)
        text_target_ /= text_target_.norm(dim=-1, keepdim=True)
        text_target_ = text_target_.requires_grad_()

        template_source = compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(accelerator.device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        text_source = text_source.requires_grad_()

        text_hidden = tokenizer(source, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        text_hidden = text_encoder(text_hidden.input_ids.cuda())[0].detach()

        text_hidden_target = tokenizer(target, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                                return_tensors="pt"
                                )
        text_hidden_target = text_encoder(text_hidden_target.input_ids.cuda())[0].detach()

    #######################################################################################

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        pass
        # data_files = {}
        # dataset_name = 'van_gogh'
        # # if args.train_data_dir is not None:
        # data_files["train"] = os.path.join(args.train_data_dir, "**")
        # print(data_files["train"])
        # dataset = load_dataset(
        #     'imagefolder',
        #     data_files=data_files,
        #     cache_dir=args.cache_dir,
        # )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names
    # print(column_names)
    #
    # # 6. Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    # if args.image_column is None:
    #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     image_column = args.image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    # if args.caption_column is None:
    #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     caption_column = args.caption_column
    #     if caption_column not in column_names:
    #         raise ValueError(
    #             f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
    #         )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    # def tokenize_captions(examples, is_train=True):
    #     captions = []
    #     for caption in examples[caption_column]:
    #         if isinstance(caption, str):
    #             captions.append(caption)
    #         elif isinstance(caption, (list, np.ndarray)):
    #             # take a random caption if there are multiple
    #             captions.append(random.choice(caption) if is_train else caption[0])
    #         else:
    #             raise ValueError(
    #                 f"Caption column `{caption_column}` should contain either strings or lists of strings."
    #             )
    #     inputs = tokenizer(
    #         captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     )
    #     return inputs.input_ids
    #
    # # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # def preprocess_train(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     examples["input_ids"] = tokenize_captions(examples)
    #     return examples
    #
    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms
    #     train_dataset = dataset["train"].with_transform(preprocess_train)
    #
    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     input_ids = torch.stack([example["input_ids"] for example in examples])
    #     return {"pixel_values": pixel_values, "input_ids": input_ids}
    #
    # # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )
    #
    # # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True
    #
    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )
    #
    # # Prepare everything with our `accelerator`.
    # lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     lora_layers, optimizer, train_dataloader, lr_scheduler
    # )
    #
    # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    #
    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training style protection*****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.style_max_iter}")
    # logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # global_step = 0
    # first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # TODO loadiing
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    print('saving prot_images')

    image_path1 = '/Data_PHD/phd22_zhaorui_tan/data/wikiart/images/vincent-van-gogh/selected/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
    image_path2 = '/Data_PHD/phd22_zhaorui_tan/data/wikiart/images/vincent-van-gogh/selected/207189.jpg'

    image_paths = [image_path1, image_path2]
    imgs = [Image.open(img_path) for img_path in image_paths]
    images = [image.convert("RGB") for image in imgs]
    pixel_values = [train_transforms(image) for image in images]
    pixel_values = torch.stack(pixel_values).cuda()
    print(pixel_values.shape)
    global_step = 0
    step = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # for step, batch in enumerate(train_dataloader):
            old = pixel_values.to(dtype=weight_dtype)
            style_network.eval()
            loss_c, out_img = style_network(old)
            get_in_process_image(old, weight_dtype, global_step, step, path=f'{args.output_dir}/prot_images/finals')
            get_in_process_image(out_img, weight_dtype, global_step, step, path=f'{args.output_dir}/prot_images/finals', key='new')
            # mix = old * ( 1-args.style_mix_ratio) + out_img * args.style_mix_ratio
            # get_in_process_image(mix, weight_dtype, global_step, step, path=f'{args.output_dir}/prot_images/finals',
            #                      key='mix')
            mix = []
            for i in range(len(old)):
                mix_ = frequency_mix(old.detach()[i], out_img.detach()[i], args.style_mix_ratio,
                                     args.style_maintain_strength, args.style_mix_strength)
                mix.append(mix_)
            mix = np.array(mix)
            mix = torch.tensor(mix).cuda()
            get_in_process_image(mix, weight_dtype, global_step, step, path=f'{args.output_dir}/prot_images/finals',
                                 key='mix')

    print('saved prot_images')

    # del style_network, style_optimizer, semantic_network, semantic_optimizer, \
    #     clip_model, preprocess, content_tf, hr_tf, test_tf, augment_trans, \
    #     noise_scheduler, tokenizer, unet, vae, text_encoder, train_transforms, \
    #     lora_layers, optimizer, train_dataloader, lr_scheduler,


def prepare_protection_components(args, device):
    # image decoder component
    decoder = fast_stylenet.decoder
    #  image encoder
    vgg = fast_stylenet.vgg
    vgg.load_state_dict(torch.load(args.style_vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(args.style_decoder))
    style_network = fast_stylenet.Net(vgg, decoder)

    style_optimizer = torch.optim.Adam(style_network.decoder.parameters(), lr=args.style_lr)

    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
    content_tf = train_transform(args.style_crop_size)
    hr_tf = hr_transform()
    test_tf = test_transform()
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])


    semantic_network = SemanticProtection(512)
    if args.style_load:
        print('+' * 100)
        print('loading style transfer')
        state_dict = torch.load(args.style_path)
        style_network = state_dict['style_network']
        semantic_network = state_dict['semantic_network']
        semantic_optimizer= state_dict['semantic_optimizer']

    style_network.to(device)
    semantic_network.to(device)
    semantic_optimizer = torch.optim.Adam(semantic_network.parameters(), lr=args.style_lr)

    return style_network, style_optimizer, semantic_network,semantic_optimizer,\
           clip_model, preprocess, content_tf, hr_tf, test_tf, augment_trans


def get_in_process_image(images, weight_dtype, global_step, step, path='prot',key='old'):
    # images = adjust_contrast(images, 1.5)
    images = torch.clamp(images, -1., 1.)
    # print('key',images)
    for j in range(len(images)):
        im = images[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        ######################################################
        # (3) Save fake images
        ######################################################
        fullpath = f"{path}/{global_step}_{step}_{key}_{j}.jpg"
        im.save(fullpath)
    return


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


from clipstyler.template import imagenet_templates
def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def frequency_mix(img, watermark, alpha, beta, style_mix_strength):
    img = img.detach().cpu().numpy()
    watermark = watermark.detach().cpu().numpy()
    img_f = np.fft.fft2(img)
    watermark_f = np.fft.fft2(watermark)

    img_f_shift = np.fft.fftshift(img_f)
    watermark_f_shift = np.fft.fftshift(watermark_f)

    c, w, h, = watermark.shape
    mask = np.ones((w//2, h))
    mask_ = np.random.choice([0, 1], size=(w//2, h//2), p=[1 - alpha, alpha])
    mask[:, :h // 2] = mask_
    # 镜像
    mask[:, h // 2: ] = np.fliplr(mask_)
    # 翻转
    mask__ = np.flipud(mask)
    mask = np.concatenate([mask, mask__], axis=0)

    margin = int((1-beta) * w)
    mask[:, margin:(w - margin)][margin:(w - margin), :] = 0
    mask = np.repeat(np.expand_dims(mask, axis=0), c, axis=0)
    res_f_shift = (1-mask) * img_f_shift + mask * img_f_shift * (1-style_mix_strength)
    res_f_shift += mask * watermark_f_shift * style_mix_strength
    res_f = np.fft.ifftshift(res_f_shift)
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    return res


if __name__ == "__main__":
    args = parse_args()
    search_style_mix_ratio = [0.75, 0.5, 0.25,]
    search_style_maintain_strength = [0.1, 0.2, 0.3, ]
    search_style_mix_strength = [0.75, 0.5, 0.25,]
    # search_style_mix_strength = [ 0.25, ]
    # search_style_mix_strength = [0.75, 0.5, 0.25,]
    # search_style_mix_ratio = [0.0,]
    # search_style_maintain_strength = [0.0,]
    # search_style_mix_strength = [0.0,]

    base_dir = args.output_dir
    for style_mix_ratio in search_style_mix_ratio:
        for style_maintain_strength in search_style_maintain_strength:
            for style_mix_strength in search_style_mix_strength:
                args.style_mix_ratio = style_mix_ratio
                args.style_maintain_strength = style_maintain_strength
                args.style_mix_strength = style_mix_strength
                args.output_dir = base_dir + f'/{int(style_mix_ratio * 100)}_{int(style_maintain_strength * 100)}_{int(style_mix_strength * 100)}'
                print(args)
                main(args)