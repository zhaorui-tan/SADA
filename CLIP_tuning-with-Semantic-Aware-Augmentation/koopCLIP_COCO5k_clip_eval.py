import torch
import torch.nn as nn
import argparse

# import clip
import CLIP as clip
# from perpare import prepare_dataloaders
# from datasets import prepare_data, encode_tokens, clip_image_embedding
from utils import merge_args_yaml
import torchvision.datasets as dataset
# import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from copy import deepcopy


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

    # parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    # parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
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


class SemanticAugCal(nn.Module):
    def __init__(self, c_dir, cond_dim, p=0.95):
        super(SemanticAugCal, self).__init__()
        c = np.load(c_dir + 'c_text_true_given_image_true.npy')
        self.C1 = torch.diagonal(torch.from_numpy(c.reshape(cond_dim, cond_dim)) * p).cuda()
        print(self.C1, p)

    def forward(self, x, y):
        with torch.no_grad():
            aug = torch.mul(x, self.C1) + y
        return aug.float()



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
    cov_dir = '/Data_PHD/phd22_zhaorui_tan/data_raw/cov/clip/coco/'
    aug = SemanticAugCal(cov_dir, 512, 0.001).cuda()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training


    save_dir = f'./outputs_clip_COCO5k_{str(args.lr)}'
    writer = SummaryWriter(f'{save_dir}/log')
    os.makedirs(save_dir, exist_ok=True)

    # train_dl, valid_dl, train_ds, valid_ds, sampler = prepare_dataloaders(args, preprocess=preprocess)

    # if device == "cpu":
    #     model.float()
    # else:
    #     clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

    train_dataset = dataset.CocoCaptions(root = '/Data_PHD/phd22_zhaorui_tan/data_raw/coco2017/train2017',
                        annFile = '/Data_PHD/phd22_zhaorui_tan/data_raw/coco2017/annotations/captions_train2017.json',
                        # transform=transforms.PILToTensor()
                        # transform=None
                        transform = preprocess
                        )

    valid_dataset = dataset.CocoCaptions(root = '/Data_PHD/phd22_zhaorui_tan/data_raw/coco2017/val2017',
                        annFile = '/Data_PHD/phd22_zhaorui_tan/data_raw/coco2017/annotations/captions_val2017.json',
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
    # loss_img = nn.CrossEntropyLoss()
    # loss_txt = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()


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


    # ixtoword = train_ds.ixtoword
    # add your own code to track the training progress.
    step = 0
    avg_param = copy_params(model)


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
    # print(all_images.shape, all_caps.shape)

    res = itm_eval(all_caps, all_images)
    for key, value in res.items():
        print(f'{key}, {value:.4f}')
        if 'txt' in key:
            writer.add_scalar(f'txt_eval/{key}', value, 0)
        elif 'img' in key:
            writer.add_scalar(f'img_eval/{key}', value, 0)
        else:
            writer.add_scalar(f'mean_eval/{key}', value, 0)
