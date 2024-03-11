import torch
import torch.nn as nn
import argparse

# import clip
import CLIP as clip
from perpare import prepare_dataloaders
from datasets import prepare_data, encode_tokens, clip_image_embedding
from utils import merge_args_yaml
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from copy import deepcopy

import torch.nn.functional as F
# clip_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
# clip_trans = transforms.Compose(
#     [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='cfg/coco.yml',
                        help='optional config file')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: 4)')
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--imsize', type=int, default=256,
                        help='input imsize')
    # parser.add_argument('--batch_size', type=int, default=128,
    #                     help='batch size')
    parser.add_argument('--batch_size', type=int, default=150,
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

    # parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam momentum factor (Beta 1)")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam rmsprop factor (Beta 2)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam eps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Adam weight decay")
    parser.add_argument("--num_warmup_steps", type=int, default=10000,
                        help="Number of steps to warmup the learning rate")
    parser.add_argument("--cylambda1", type=float, default=0.25, help="Cyclic regularization lambda 1")
    parser.add_argument("--cylambda2", type=float, default=0.25, help="Cyclic regularization lambda 2")
    args = parser.parse_args()
    return args

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        try:
            p.grad.data = p.grad.data.float()
        except Exception:
            pass # pass data with no grad

def convert_models_to_mix(model):
    clip.model.convert_weights(model)


def get_rss(h1, h2):
    h1, h2 = h1.float(), h2.float()
    h1t = h1.permute(1, 0)
    if h1.shape[0] >= h1.shape[1]:
        # print(torch.linalg.pinv(torch.matmul(h1t, h1)))
        pinv = torch.matmul(torch.linalg.pinv(torch.matmul(h1t, h1)), h1t)
    else:
        # print(torch.matmul(h1,h1t).shape)
        # print(torch.linalg.pinv(torch.matmul(h1, h1t)))
        pinv = torch.matmul(h1t,torch.linalg.pinv(torch.matmul(h1, h1t)))
    # print(pinv)
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

    for index in range(len(image_embeddings)):
        # the index image compare to all texts
        # image paired to  [index*5 * 5 , (index+1)*5]
        scores = image_embeddings[index] @ text_embeddings.t()
        # scores = sim_matrix_i2t[index]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if index*5 <= li[i] and li[i]<= index*5 + 4:
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
        # scores = scores[::5]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            # if li[i] == index // 5:
            # if li[i] == index // 5:
            if li[i] == index//5:
            # if index <= li[i] and li[i] <= index + 4:
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



def collate_func(data):
    data = list(zip(*data))
    imgs = data[0]
    imgs = torch.stack(imgs, 0)
    txts = data[1]
    return imgs, txts


def cosine_scheduler(optimizer, base_lr, num_warmup_steps, total_steps):
    def _scheduler(current_step):
        if (current_step < num_warmup_steps):
            lr = base_lr * (current_step + 1) / num_warmup_steps
        else:
            n = current_step - num_warmup_steps
            d = total_steps - num_warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _scheduler



def sample_covariance(a, b):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    N = a.shape[0]
    C = torch.matmul(a.T, b) / N
    return C


def copy_params(model):
    print('copying_params')
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    print('loading_params')
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

if __name__ == '__main__':
    import os
    from torch.utils.tensorboard import SummaryWriter
    import copy
    args = merge_args_yaml(parse_args())

    eigen_w = 0.0
    rss_w = 0.1
    cross_w = 0.0
    cov_w = 0
    t = 0.0
    T = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.


    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
    # T_model, T_preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    save_dir = f'./outputs_clip_COCO5k_ce_ub_cro_cov_bs_nz_ema_KD2_rss_c{T}_{str(args.batch_size)}_{str(args.lr)}_eigen_w{str(eigen_w)}_cov_w{str(cov_w)}_cross_w{str(cross_w)}_rss_w{str(rss_w)}'
    writer = SummaryWriter(f'{save_dir}/log')
    os.makedirs(save_dir, exist_ok=True)

    # train_dl, valid_dl, train_ds, valid_ds, sampler = prepare_dataloaders(args, preprocess=preprocess)

    # if device == "cpu":
    #     model.float()
    # else:
    #     clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

    train_dataset = dataset.CocoCaptions(root = '/data1/phd21_zhaorui_tan/data_raw/coco2017/train2017',
                        annFile = '/data1/phd21_zhaorui_tan/data_raw/coco2017/annotations/captions_train2017.json',
                        # transform=transforms.PILToTensor()
                        # transform=None
                        transform = preprocess
                        )

    valid_dataset = dataset.CocoCaptions(root = '/data1/phd21_zhaorui_tan/data_raw/coco2017/val2017',
                        annFile = '/data1/phd21_zhaorui_tan/data_raw/coco2017/annotations/captions_val2017.json',
                        # transform=transforms.PILToTensor()
                         # transform=None
                        transform=preprocess
                        )

    # test_ds = dataset.CocoCaptions(root='/data1/phd21_zhaorui_tan/data_raw/coco2017/test2017',
    #                                 annFile='/data1/phd21_zhaorui_tan/data_raw/coco2017/annotations/captions_test2017.json',
    #                                 transform=preprocess)


    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, shuffle='True', collate_fn = collate_func)

    valid_dl = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers, shuffle=False, collate_fn=collate_func)


    model.cuda()

    for param in T_model.parameters():
        param.requires_grad = False
        # model = nn.DataParallel(model)
    T_model.cuda()
    # try:
    #     model.module = model
    # except Exception:
    #     pass
    # loss_img = nn.CrossEntropyLoss()
    # loss_txt = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean", log_target=True)


    weight_decay_parameters = []
    no_weight_decay_parameters = []

    for name, parameter in model.named_parameters():
        if (all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
            weight_decay_parameters.append(parameter)

        if (any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
            no_weight_decay_parameters.append(parameter)



    optimizer = torch.optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0},
                             {"params": weight_decay_parameters, "weight_decay": args.weight_decay}], lr=args.lr,
                            betas=(args.beta1, args.beta2), eps=args.eps)
    scheduler = cosine_scheduler(optimizer, args.lr, args.num_warmup_steps,
                                 len(train_dl) * args.epoch)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.99),
    #                        weight_decay=0.1)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # sup_in_l =
    # print(model.visual, model.transformer)

    # ixtoword = train_ds.ixtoword
    # add your own code to track the training progress.
    step = 0
    avg_param = copy_params(model)
    ilc = 0.002

    # fc1 = nn.Linear(512, 512 - 64)
    # fc2 = nn.Linear(512, 512 - 64)
    # fc3 = nn.Linear(512, 512 - 64)
    # fc4 = nn.Linear(512, 512 - 64)
    # optim1 = torch.optim.Adam(fc1.parameters(), lr=args.lr)
    # optim2 = torch.optim.Adam(fc2.parameters(), lr=args.lr)
    # optim3 = torch.optim.Adam(fc3.parameters(), lr=args.lr)
    # optim4 = torch.optim.Adam(fc4.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        pbar = tqdm(train_dl)

        for data in pbar:
            # prepare_data
            # imgs, captions = prepare_data(data, ixtoword,)
            imgs, caps = data
            # imgs = preprocess(torch.tensor(imgs))
            # imgs = imgs.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            imgs = imgs.to(device)
            # print(imgs.shape)
            # imgs = imgs

            captions = []
            for cap in caps:
                cap = clip.tokenize(cap).to(device)
                id = np.random.choice(list(range(5)))
                captions.append(cap[id])
            captions = torch.stack(captions)
            # imgs = imgs.view((-1, 3, 224, 224))
            captions = captions.view((-1, 77))

            # imgs = preprocess(imgs)

            image_features = model.encode_image(imgs)
            text_features = model.encode_text(captions)

            T_image_features = T_model.encode_image(imgs)
            T_text_features = T_model.encode_text(captions)
            # noise = torch.randn_like(text_features).cuda() * 0.0001
            # text_features += noise.requires_grad_()

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            T_image_features = T_image_features / T_image_features.norm(dim=-1, keepdim=True)
            T_text_features = T_text_features / T_text_features.norm(dim=-1, keepdim=True)

            # rss_i2t = get_rss(image_features, T_image_features)
            # rss_t2i = get_rss(text_features, T_text_features)

            # image_cov = torch.corrcoef(image_features).float().cuda()
            # mask = torch.eye(len(image_cov)).float().cuda()
            # image_cov_loss = (nn.MSELoss()(image_cov * (1-mask), torch.zeros_like(image_cov)) +
            #                   nn.MSELoss()(image_cov*mask, mask)) * cov_w
            #
            # text_cov = torch.corrcoef(text_features).float().cuda()
            # text_cov_loss = (nn.MSELoss()(text_cov * (1-mask), torch.zeros_like(text_cov)) +
            #                  nn.MSELoss()(text_cov*mask, mask)) * cov_w
            # cov_loss = nn.MSELoss()(image_cov, text_cov)
            # cov_loss = criterion(image_cov, text_cov)
            # cov_loss = image_cov_loss  + text_cov_loss * 0
            # cov = sample_covariance(image_features - torch.mean(image_features, dim=0).detach(),
            #                         text_features - torch.mean(text_features, dim=0).detach())
            # cov = cov.float() / cov.float().norm(dim=-1, keepdim=True).detach()
            # mask = torch.eye(len(cov)).float().cuda()
            # cov_loss = nn.MSELoss()(cov * (1-mask), torch.zeros_like(cov)) + nn.MSELoss()(cov*mask, mask)


            # loss, contrastive_loss, cyclic_loss = get_loss(model, image_features, text_features, criterion, args)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # T_logit_scale = T_model.logit_scale.exp()
            # T_logits_per_image = T_logit_scale * T_image_features @ T_text_features.t()
            # T_logits_per_text = T_logits_per_image.t()

            # logits_per_image, logits_per_text = model(imgs, captions )
            ground_truth = torch.arange(len(imgs), dtype=torch.long, device=device)
            loss1= criterion(logits_per_image, ground_truth)
            loss2= criterion(logits_per_text, ground_truth)


            # rss_i_s2t = torch.relu(get_rss(fc1(image_features), fc2(T_image_features)) - ilc)
            # rss_i_t2s = torch.relu(get_rss(fc2(T_image_features), fc1(image_features)) - ilc)
            #
            # rss_t_s2t = torch.relu(get_rss(fc3(text_features), fc4(T_text_features)) - ilc)
            # rss_t_t2s = torch.relu(get_rss(fc4(T_text_features), fc3(text_features)) - ilc)
            #
            # rss_t = torch.relu(rss_t_t2s + rss_t_s2t)
            # rss_i = torch.relu(rss_i_t2s + rss_i_s2t)

            # rss_t2i = get_rss(text_features, image_features)


            # image_features = F.log_softmax(image_features, dim=1)
            # T_image_features = F.log_softmax(T_image_features, dim=1)
            # text_features = F.log_softmax(text_features, dim=1)
            # T_text_features = F.log_softmax(T_text_features, dim=1)

            # T_loss1 = kl(image_features, T_image_features)
            # T_loss2 = kl(text_features, T_text_features)

            loss = (loss1 + loss2) / 2
            # T_loss = (rss_t + rss_i) / 2 * rss_w

            # total_loss = loss   + (rss_i2t + rss_t2i)/2 + (eigen_image_loss + eigen_text_loss)/2 * 0
            # total_loss = loss + (image_cov_loss + text_cov_loss) / 2 + cross_loss + (eigen_image_loss + eigen_text_loss) / 2 * 0
            total_loss = loss + T_loss

            pbar.set_description(f'loss_img_txt: {loss.item():.4f}, '
                                 # f'T_loss: {T_loss.item() :.4f}, '
                                 # f'rss_t: {rss_t.item():.4f}, '
                                 # f'rss_i: {rss_i.item():.4f}, '
                                 f'total_loss: {total_loss.item():.4f}, '
                                 )
            writer.add_scalar(f'train/loss_img_txt', loss.item(), step)
            # writer.add_scalar(f'train/T_loss', T_loss.item(), step)
            # writer.add_scalar(f'train/rss_t', rss_t.item(), step)
            # writer.add_scalar(f'train/rss_i', rss_i.item(), step)
            writer.add_scalar(f'train/total_loss', total_loss.item(), step)

            total_loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            convert_models_to_mix(model)
            step +=1

            for p, avg_p in zip(model.parameters(), avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

        # if epoch % 5 == 0:
        model.eval()
        model.cpu()
        backup_param = copy_params(model)
        load_params(model, avg_param)
        model.cuda()
        all_images = []
        all_caps = []
        pbar = tqdm(valid_dl)
        for data in pbar:
            with torch.no_grad():
            # for step, data in enumerate(valid_dl):
            #     imgs, captions = prepare_data(data, ixtoword, )
                imgs, caps = data
                # imgs = preprocess(torch.tensor(imgs))
                # imgs = imgs.unsqueeze(1).repeat(1, 5, 1, 1, 1)
                imgs = imgs.to(device)
                # print(imgs.shape)
                # imgs = imgs

                captions = []
                for cap in caps:
                    cap = clip.tokenize(cap).to(device)
                    captions.append(cap[:5])
                captions = torch.stack(captions)
                # imgs = imgs.view((-1, 3, 224, 224))
                captions = captions.view((-1, 77))

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
        model.cpu()
        load_params(model, backup_param)
        model.cuda()
