import torch
import torch.nn as nn
import argparse

# import clip
import CLIP as clip
from perpare import prepare_dataloaders
from datasets import prepare_data, encode_tokens, clip_image_embedding
from utils import merge_args_yaml
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# clip_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
# clip_trans = transforms.Compose(
#     [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='cfg/coco.yml',
                        help='optional config file')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: 4)')
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--imsize', type=int, default=256,
                        help='input imsize')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--train', type=bool, default=True,
                        help='if train model')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='resume epoch')
    parser.add_argument('--resume_model_path', type=str, default='model',
                        help='the model for resume training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if multi-gpu training under ddp')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true', default=True,
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def convert_models_to_mix(model):
    clip.model.convert_weights(model)


def get_rss(h1, h2):
    h1, h2 = h1.float(), h2.float()
    h1t = h1.permute(1, 0)
    if h1.shape[0] >= h1.shape[1]:
        pinv = torch.matmul(torch.linalg.pinv(torch.matmul(h1t, h1)), h1t)
    else:
        # print(torch.matmul(h1,h1t).shape)
        pinv = torch.matmul(h1t,torch.linalg.pinv(torch.matmul(h1, h1t)))
    K = torch.matmul(pinv, h2)
    # print(K.shape,h1.shape, (K*h1).shape)
    rss = 0
    # for i in range(len(pinv)):
    h1_ = nn.functional.linear(h1,K)
    rss += nn.MSELoss()(h1_, h2)
    return rss


@torch.no_grad()
def itm_eval(text_embeddings, image_embeddings):
    # sim_matrix_i2t = image_embeddings @ text_embeddings.t()
    # sim_matrix_t2i = text_embeddings @ image_embeddings.t()

    ## Image -> Text
    # ranks = np.zeros(len(sim_matrix_i2t))
    ranks = np.zeros(len(image_embeddings))

    for index in range(0, len(image_embeddings), 5):
        scores = image_embeddings[index] @ text_embeddings.t()
        # scores = sim_matrix_i2t[index]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if index <= li[i] and li[i] <= index + 4:
                rank = i
                break
        ranks[index] = rank

        # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)

    ## Image -> Text
    ranks = np.zeros(len(text_embeddings))
    for index in range(len(text_embeddings)):
        scores = text_embeddings[index] @ image_embeddings.t()
        # for index, scores in tqdm(enumerate(sim_matrix_t2i)):
        scores = scores[::5]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if li[i] == index // 5:
                rank = i
                break
        ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    ir20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r20': tr20,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r20': ir20,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}

    return eval_result

if __name__ == '__main__':
    import os
    from torch.utils.tensorboard import SummaryWriter
    import copy
    args = merge_args_yaml(parse_args())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    save_dir = './outputs_clip_ori_1e-6'
    writer = SummaryWriter(f'{save_dir}/log')
    os.makedirs(save_dir, exist_ok=True)

    train_dl, valid_dl, train_ds, valid_ds, sampler = prepare_dataloaders(args, preprocess=preprocess)

    # if device == "cpu":
    #     model.float()
    # else:
    #     clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16
    model.cuda()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-8,
    #                        weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.99),
                           weight_decay=0.1)
    ixtoword = train_ds.ixtoword
    # add your own code to track the training progress.
    step = 0
    for epoch in range(args.epoch):
        model.train()
        pbar = tqdm(train_dl)
        for data in pbar:
            # prepare_data
            imgs, captions = prepare_data(data, ixtoword,)
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions = clip.tokenize(captions).to(device)

            image_features = model.encode_image(imgs)
            text_features = model.encode_text(captions)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            rss = get_rss(image_features, text_features)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # logits_per_image, logits_per_text = model(imgs, captions )
            ground_truth = torch.arange(len(imgs), dtype=torch.long, device=device)
            loss1 = loss_img(logits_per_image, ground_truth)
            loss2= loss_txt(logits_per_text, ground_truth)
            total_loss = ( loss1+loss2 ) / 2
            pbar.set_description(f'loss_img: {loss1.item():.4f}, '
                                 f'loss_txt: {loss2.item():.4f}, '
                                 f'rss: {rss.item():.4f}, '
                                 f'total_loss: {total_loss.item():.4f}, '
                                 )
            writer.add_scalar(f'train/loss_img', loss1.item(), step)
            writer.add_scalar(f'train/loss_txt', loss2.item(), step)
            writer.add_scalar(f'train/rss', rss.item(), step)
            writer.add_scalar(f'train/total_loss', total_loss.item(), step)

            total_loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            convert_models_to_mix(model)

            step += 1


        if epoch % 5 == 0:
            model.eval()
            all_images = []
            all_caps = []
            pbar = tqdm(valid_dl)

            for data in pbar:
                with torch.no_grad():
                # for step, data in enumerate(valid_dl):
                    imgs, captions = prepare_data(data, ixtoword, )
                    imgs = imgs.to(device)
                    captions = clip.tokenize(captions).to(device)
                    # logits_per_image, logits_per_text = model(imgs, captions)
                    image_features = model.encode_image(imgs)
                    text_features = model.encode_text(captions)
                    # normalized features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_images.append(image_features)
                all_caps.append(text_features)

            all_images = torch.cat(all_images, )
            all_caps = torch.cat(all_caps, )
            print(all_images.shape, all_caps.shape)

            res = itm_eval(all_caps, all_images)
            for key, value in res.items():
                print(f'{key}, {value:.4f}')
                if 'txt' in key:
                    writer.add_scalar(f'txt_eval/{key}', value, epoch)
                elif 'img' in key:
                    writer.add_scalar(f'img_eval/{key}', value, epoch)
                else:
                    writer.add_scalar(f'mean_eval/{key}', value, epoch)


            torch.save(model.state_dict(), save_dir + f"/model_{epoch}.pth")
            print('saved model at ' + save_dir + f"/model_{epoch}.pth")