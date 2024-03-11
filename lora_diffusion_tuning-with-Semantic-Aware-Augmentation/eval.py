# load prot model
# use mix
# cal mse
# cal logits cross entropy

from PIL import Image
import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

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
    parser.add_argument('--style_train', type=bool, default=True)
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

    parser.add_argument('--test_use_downsample', type=bool, default=True)

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

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    ######################################################################################
    ######################################################################################
    ######################################################################################
    args = parser.parse_args()
    print(args)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

DATASET_NAME_MAPPING = {
    "van_gogh": ("image"),
}

def main():
    args = parse_args()
    # logging_dir = os.path.join(args.output_dir, args.logging_dir)
    #
    # accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    #
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with=args.report_to,
    #     logging_dir=logging_dir,
    #     project_config=accelerator_project_config,
    # )
    # if args.report_to == "wandb":
    #     if not is_wandb_available():
    #         raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    #     import wandb

    # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_warning()
    #     diffusers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()
    #     diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.push_to_hub:
    #         if args.hub_model_id is None:
    #             repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
    #         else:
    #             repo_name = args.hub_model_id
    #         repo_name = create_repo(repo_name, exist_ok=True)
    #         repo = Repository(args.output_dir, clone_from=repo_name)
    #
    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    #         os.makedirs(f'{args.output_dir}/prot_images', exist_ok=True)
    #         os.makedirs(f'{args.output_dir}/prot_images/finals', exist_ok=True)

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )
    # # freeze parameters of models to save more memory
    # unet.requires_grad_(False)
    # vae.requires_grad_(False)
    #
    # text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #
    # # Move unet, vae and text_encoder to device and cast to weight_dtype
    # unet.to(accelerator.device, dtype=weight_dtype)
    # vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)

    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers
    #
    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

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
    # lora_attn_procs = {}
    # for name in unet.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
    #
    #     lora_attn_procs[name] = LoRACrossAttnProcessor(
    #         hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
    #     )
    #
    # unet.set_attn_processor(lora_attn_procs)
    # lora_layers = AttnProcsLayers(unet.attn_processors)
    #
    # # Enable TF32 for faster training on Ampere GPUs,
    # # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-amp ere-devices
    # if args.allow_tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #
    # if args.scale_lr:
    #     args.learning_rate = (
    #         args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    #     )
    #
    # # Initialize the optimizer
    # if args.use_8bit_adam:
    #     try:
    #         import bitsandbytes as bnb
    #     except ImportError:
    #         raise ImportError(
    #             "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
    #         )
    #
    #     optimizer_cls = bnb.optim.AdamW8bit
    # else:
    #     optimizer_cls = torch.optim.AdamW

    # optimizer = optimizer_cls(
    #     lora_layers.parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    #######################################################################################
    # we looking forward a noise for raw images that lead to zero optimization for lora
    style_network, style_optimizer, semantic_network, semantic_optimizer, \
    clip_model, preprocess, content_tf, hr_tf, test_tf, augment_trans = prepare_protection_components(args, 'cuda')


    #######################################################################################

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    data_files = {}
    dataset_name = 'van_gogh'
    # if args.train_data_dir is not None:
    data_files["train"] = os.path.join(args.train_data_dir, "**")
    print(data_files["train"])
    dataset = load_dataset(
        'imagefolder',
        data_files=data_files,
        cache_dir=args.cache_dir,
    )
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    print(column_names)

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        # examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values,}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    weight_dtype = torch.float16


    ####################################################################################################

    # style transfer stage

    ####################################################################################################
    print(args)
    enc_layers = list(*clip_model.visual.transformer.children())
    print(len(enc_layers))
    enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*enc_layers[4:8])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*enc_layers[8:])  # relu2_1 -> relu3_1
    encs = [enc_1.requires_grad_(False), enc_2.requires_grad_(False), enc_3.requires_grad_(False)]
    print('saving prot_images')
    all_mse = []
    all_div_mse = []
    all_res = []

    with torch.no_grad():
        style_network.eval()

        for step, batch in enumerate(train_dataloader):
            old = batch["pixel_values"].to(dtype=weight_dtype).cuda()
            with torch.cuda.amp.autocast():
                loss_c, out_img = style_network(old)
            mix = []
            for i in range(len(old)):
                mix_ = frequency_mix(old.detach()[i], out_img.detach()[i], args.style_mix_ratio,
                                     args.style_maintain_strength, args.style_mix_strength)
                mix.append(mix_)
            mix = np.array(mix)
            mix = torch.tensor(mix).cuda()
            mse =  F.mse_loss(old,mix)
            div_mse = 0
            cur_old,  cur_mix = old, mix
            old_, mix_ = get_hidden(clip_model, encs, clip_normalize(cur_old)), get_hidden(clip_model, encs, clip_normalize(cur_mix))
            for old_1, mix_1 in zip(old_, mix_):
                div_mse += F.mse_loss(old_1, mix_1)
            div_mse /= len(encs)
            res = mse / (div_mse + 1e-8)
            print('mse', mse.item(), 'div_mse', div_mse.item(), 'div', res.item() )
            all_mse.append(mse.item())
            all_div_mse.append(div_mse.item())
            all_res.append(res.item())

    print("*"*100)
    print('all_mse', np.mean(all_mse), 'all_div_mse',np.mean(all_div_mse), 'all_res', np.mean(all_res), )

def get_hidden(model, models_for_hidden, x):
    d_type = model.dtype
    model = model.visual
    x = model.conv1(x.type(d_type))  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    hiddens = []
    for i in range(len(models_for_hidden)):
        x = models_for_hidden[i](x)
        hiddens.append(x)

    x = x.permute(1, 0, 2)  # LND -> NLD

    x = model.ln_post(x[:, 0, :])

    if model.proj is not None:
        x = x @ model.proj
        hiddens.append(x)

    return hiddens



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
    main()