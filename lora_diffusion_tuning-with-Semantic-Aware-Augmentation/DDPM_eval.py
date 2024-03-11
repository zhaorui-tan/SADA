#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

# import accelerate
# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
# from accelerate import Accelerator
# from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from datasets_N import prepare_data
from datasets_N import TextDatasetBirds, TextDatasetCOCO
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0.dev0")



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        default="sd-model-finetuned",
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
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
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

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}
from miscc.config import cfg, cfg_from_file
from miscc.config import cfg

def main():

    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)



    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    bshuffle = False
    split_dir = 'test'
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Potentially load in the weights and states from a previous save
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar.set_description("Steps")


    # Create the pipeline using the trained modules and save it.
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = accelerator.unwrap_model(unet)
    #     if args.use_ema:
    #         ema_unet.copy_to(unet.parameters())
    batch_size = 2
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        revision=args.revision,
    )

    cfg_from_file('/Data_PHD/phd22_zhaorui_tan/SDE_test/cfg/eval_coco.yml')
    # cfg_from_file('/Data_PHD/phd22_zhaorui_tan/SDE_test/cfg/eval_bird.yml')
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.DATASET_NAME == 'birds':
        dataset = TextDatasetBirds(cfg.DATA_DIR, split_dir,
                                   base_size=cfg.TREE.BASE_SIZE,
                                   transform=image_transform)
    else:
        dataset = TextDatasetCOCO(cfg.DATA_DIR, split_dir,
                                  base_size=cfg.TREE.BASE_SIZE,
                                  transform=image_transform)
    print('got dataset')
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    # unet.eval()
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     text_encoder=text_encoder,
    #     vae=vae,
    #     unet=unet,
    #     revision=args.revision,
    # )
    # tmp_path = f"{args.output_dir}"
    # pipe = StableDiffusionPipeline.from_pretrained(tmp_path, torch_dtype=torch.float16)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.to("cuda")

    # image = pipe(prompt="yoda").images[0]
    # image.save("images/yoda-pokemon.png")
    sampling_ssd(pipeline,batch_size, dataloader, dataset.ixtoword)
    # if args.push_to_hub:
    #     repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    # accelerator.end_training()


def sampling_ssd(pipe, batch_size, data_loader, ixtoword):
    import tensorflow as tf
    from ssd_tf import ssd
    from miscc.utils import mkdir_p
    from FID.fid_score import calculate_fid_given_paths
    os.environ['MKL_NUM_THREADS'] = '1'
    gpu = tf.config.list_physical_devices('GPU')
    print(gpu,)

    # Build and load the generator
    device = 'cuda'
    clip_model,preprocess, clip_pool, clip_trans = load_clip(device)

    # the path to save generated images
    save_dir = 'fid_ssd_eval_coco'
    mkdir_p(save_dir)

    cnt = 0
    R_count = 0
    # R = np.zeros(30000)
    n = 30000
    comp_scores = np.zeros(n)
    cont = True


    all_real = np.zeros((n, 512))
    all_fake = np.zeros((n, 512))
    all_caps = np.zeros((n, 512))


    for i in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(data_loader, 0):
            torch.cuda.empty_cache()
            cnt += batch_size
            if R_count % 100 == 0:
                print('R_count: ', R_count)
            if (cont == False):
                break

            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            #######################################################
            # (2) Generate fake images
            ######################################################

            sent_txt = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                sent_txt.append(tmp_s)

            sent_clip, sent_emb_clip = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)

            imgs = imgs[-1].to(device)
            real_for_clip = clip_trans(clip_pool(imgs))
            real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)



            for j in range(batch_size):
                caps_txt = ' '.join(
                    [ixtoword[k] for k in captions[j].to('cpu').numpy().tolist() if k != 0])

                with torch.no_grad():
                    print(caps_txt)
                    image = pipe(prompt=caps_txt).images[0]
                    # image.save("images/yoda-pokemon.png")
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    fullpath = '%s_%3d_%s.png' % (s_tmp, i, caps_txt.replace(' ', '_'))
                    image.save(fullpath)
                fake_for_clip = clip_trans(clip_pool(preprocess(image).unsqueeze(0)))
                fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)
                fake_for_clip /= fake_for_clip.norm(dim=-1, keepdim=True)
                all_fake[R_count] = fake_for_clip.detach().cpu().numpy()

                # fake_for_clip[j] /= fake_for_clip[j].norm(dim=-1, keepdim=True)
                real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
                sent_emb_clip[j] /= sent_emb_clip[j].norm(dim=-1, keepdim=True)
                all_real[R_count] = real_for_clip[j].detach().cpu().numpy()
                # all_fake[R_count] = fake_for_clip[j].detach().cpu().numpy()
                all_caps[R_count] = sent_emb_clip[j].detach().cpu().numpy()
                R_count += 1
                # print('R_count: ', R_count)

                if R_count >= n:
                    print('data loaded')
                    all_real = tf.convert_to_tensor(all_real)
                    all_fake = tf.convert_to_tensor(all_fake)
                    all_caps = tf.convert_to_tensor(all_caps)
                    ssd_scores = ssd(all_real, all_fake, all_caps, )
                    result = f'SSD:{ssd_scores[0]}, SS:{ssd_scores[1]}, dSV:{ssd_scores[2]}, TrSV:{ssd_scores[3]}, '
                    print(result)

                    # if 'bird' in s_tmp:
                    #     paths = [f'{save_dir}',
                    #              '/data1/phd21_zhaorui_tan/data_raw/birds/CUB_200_2011/images']
                    # else:
                    paths = [f'{save_dir}',
                                 '/Data_PHD/phd22_zhaorui_tan/data_raw/coco/test/test2014']

                    fid_value = calculate_fid_given_paths(paths, batch_size, None, 2048)
                    result += f'FID: {fid_value}'
                    print(result)
                    break
            if R_count >= n:
                break
        if R_count >= n:
            break


import clip

def load_clip(device):
    clip_model, preprocess = clip.load('ViT-B/32', device)
    clip_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
    clip_trans = transforms.Compose(
        [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model, preprocess, clip_pool, clip_trans


def clip_text_embedding(text, clip_model, device):
    text = clip.tokenize(text).to(device)
    text_features = clip_model.encode_text(text)
    return text, text_features.float()


def clip_image_embedding(image, clip_model, device):
    image = image.to(device)
    image_features = clip_model.encode_image(image)
    return image_features.float()

if __name__ == "__main__":
    main()