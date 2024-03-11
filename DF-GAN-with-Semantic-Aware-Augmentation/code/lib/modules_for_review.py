import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from lib.utils import truncated_noise
from lib.utils import mkdir_p, get_rank

from lib.datasets import TextImgDataset as Dataset
from lib.datasets import prepare_data, encode_tokens, clip_image_embedding
from models.inception import InceptionV3

from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist


def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp, truncation,
           trunc_rate,
           ixtoword, args):
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder, ixtoword, args)
        sent_emb = sent_emb.to(device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            if truncation == True:
                noise = truncated_noise(batch_size, z_dim, trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise, sent_emb)
        for j in range(batch_size):
            s_tmp = '%s/single/%s' % (save_dir, keys[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            ######################################################
            # (3) Save fake images
            ######################################################            
            if multi_gpus == True:
                filename = 'd%d_s%s.png' % (get_rank(), stamp)
            else:
                filename = 's%s.png' % (stamp)
            fullpath = '%s_%s.png' % (s_tmp, filename)
            im.save(fullpath)


def test(dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
         times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8, ixtoword=None, args=None):
    fid = calculate_fid(dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
                        times, z_dim, batch_size, truncation, trunc_rate, ixtoword, args)
    return fid


def calculate_fid(dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
                  times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8, ixtoword=None, args=None):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
    ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu != 1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length * times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder, ixtoword, args)
            sent_emb = sent_emb.to(device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation == True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise, sent_emb)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu != 1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluate Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu != 1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def eval(dataloader, text_encoder, netG, device, m1, s1, save_imgs, save_dir,
         times, z_dim, batch_size, truncation=True, trunc_rate=0.86, ixtoword=None, args=None):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
    ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu != 1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length * times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder, ixtoword, args)
            sent_emb = sent_emb.to(device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation == True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise, sent_emb)
                if save_imgs == True:
                    save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu != 1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()
    if (n_gpu != 1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def eval_cs(dataloader, text_encoder, netG, device, m1, s1, save_imgs, save_dir,
             times, z_dim, batch_size, truncation=True, trunc_rate=0.86, ixtoword=None, args=None):
    """ Calculates the CS """
    # only use one gpu
    # prepare ssd and clip
    import tensorflow as tf
    from lib.cs import cs
    from lib.perpare import load_clip
    from lib.datasets import clip_text_embedding, clip_image_embedding, clip_pool, clip_trans

    gpu_ori_setting = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ori_setting.split(',')[-1]
    os.environ["MKL_NUM_THREADS"] = "1"
    gpu = tf.config.list_physical_devices('GPU')
    split_dir = 'valid'
    target_device = 'cpu'
    dl_length = dataloader.__len__()

    netG.eval()
    clip_model = load_clip()
    cnt = 0
    R_count = 0
    n = 30000

    cont = True
    all_real = np.zeros((n, 512))
    all_fake = np.zeros((n, 512))
    all_caps = np.zeros((n, 512))

    for time in range(times):
        if not cont:
            break
        else:
            for i, data in enumerate(dataloader):
                ######################################################
                # (1) Prepare_data
                ######################################################
                imgs, sent_emb, words_embs, keys, captions = prepare_data(data, text_encoder, ixtoword, args, return_captions= True)
                sent_emb = sent_emb.to(device)

                ######################################################
                # (2) Generate fake images
                ######################################################
                batch_size = sent_emb.size(0)
                netG.eval()
                with torch.no_grad():
                    if truncation == True:
                        noise = truncated_noise(batch_size, z_dim, trunc_rate)
                        noise = torch.tensor(noise, dtype=torch.float).to(device)
                    else:
                        noise = torch.randn(batch_size, z_dim).to(device)
                    fake_imgs = netG(noise, sent_emb)

                real_for_clip = clip_trans(clip_pool(imgs))
                real_for_clip = clip_image_embedding(real_for_clip, clip_model, )
                fake_for_clip = clip_trans(clip_pool(fake_imgs))
                fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, )

                sent = []
                is_tensor = False
                if type(captions[0]) == torch.Tensor:
                    is_tensor = True
                for b in range(len(captions)):
                    if type(captions[b].to('cpu').numpy().tolist()[0]) == list:
                        if is_tensor:
                            tmp_w = [ixtoword[k[0]] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                        else:
                            tmp_w = [ixtoword[k[0]]for k in captions[b] if k != 0]
                        tmp_s = ' '.join(tmp_w)
                        sent.append(tmp_s)
                    else:
                        if is_tensor:
                            tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                        else:
                            tmp_w = [ixtoword[k] for k in captions[b] if k != 0]
                        tmp_s = ' '.join(tmp_w)
                        sent.append(tmp_s)

                _, sent_emb_clip = clip_text_embedding(text=sent, clip_model=clip_model)


                for j in range(batch_size):
                    if save_imgs:
                        save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size)
                    fake_for_clip[j] /= fake_for_clip[j].norm(dim=-1, keepdim=True)
                    real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
                    sent_emb_clip[j] /= sent_emb_clip[j].norm(dim=-1, keepdim=True)

                    all_real[R_count] = real_for_clip[j].to(target_device).numpy()
                    all_fake[R_count] = fake_for_clip[j].to(target_device).numpy()
                    all_caps[R_count] = sent_emb_clip[j].to(target_device).numpy()
                    R_count += 1
                    print('R_count: ', R_count)
                    if R_count >= n:
                        cont = False
                        break
                if R_count >= n:
                    cont = False
                    break

    print('data loaded')
    all_real = tf.convert_to_tensor(all_real)
    all_fake = tf.convert_to_tensor(all_fake)
    all_caps = tf.convert_to_tensor(all_caps)
    cs_score = cs(all_real, all_fake, all_caps,)
    result = f'CS:{cs_score} '
    return result


def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            # print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png' % (time * dl_len + batch_size * batch_n + j, get_rank())
        fullpath = osp.join(folder, filename)
        im.save(fullpath)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def sample_one_batch(noise, sent, netG, aug, multi_gpus, epoch, img_save_dir, writer, args):
    if (multi_gpus == True) and (get_rank() != 0):
        None
    else:
        fixed_results, fixed_results2 = generate_samples(noise, sent, netG, aug, args)
        if writer != None:
            fixed_grid = make_grid(fixed_results.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results', fixed_grid, epoch)
        img_name = 'samples_epoch_%03d.png' % (epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)

        if writer != None:
            fixed_grid = make_grid(fixed_results2.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results aug', fixed_grid, epoch)
        img_name = 'aug_samples_epoch_%03d.png' % (epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results2.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)


def generate_samples(noise, caption, model, aug, args):
    with torch.no_grad():
        fake = model(noise, caption)
        if args.AUG.TRAIN:
            sent_emb_aug = aug(caption, caption)
            fake_aug = model(noise, sent_emb_aug)
        else:
            c_noise = ((torch.rand(len(caption), args.AUG.EMBEDDING_DIM) - 0.5) * 2).to('cuda')
            # sent_emb_aug = sent_emb + aug(c_noise, sent_emb)
            sent_emb_aug = aug(c_noise, caption)
            fake_aug = model(noise, sent_emb_aug)
    return fake, fake_aug


#########   MAGP   ########
def MA_GP(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                                inputs=(img, sent),
                                grad_outputs=torch.ones(out.size()).cuda(),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0, grad1), dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = 2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output, err


def hinge_loss(output, negtive):
    if negtive == False:
        err = torch.nn.ReLU()(1.0 - output).mean()
    else:
        err = torch.nn.ReLU()(1.0 + output).mean()
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size, 1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size, 1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive == False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def aug_loss(image_encoder, image_features, image_features_aug, sent_code, sent_code_aug, args):
    """
    L_db
    """
    if args.TEXT.USE_CLIP:
        cnn_code1_t = clip_image_embedding(image_features, image_encoder)
        cnn_code2_t = clip_image_embedding(image_features_aug, image_encoder)
    else:
        _, cnn_code1_t = image_encoder(image_features)
        _, cnn_code2_t = image_encoder(image_features_aug)

    image_code_t = cnn_code2_t - cnn_code1_t
    text_code_t = sent_code_aug - sent_code
    sim = torch.mean(torch.cosine_similarity(image_code_t, text_code_t, ))
    return 1 - sim
