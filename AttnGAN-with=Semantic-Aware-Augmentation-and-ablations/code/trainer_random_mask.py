from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data_random_mask, prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss_random_mask, KL_loss, aug_loss
import os
import time
import numpy as np
import sys

import clip
import torchvision.transforms as transforms
from ssd_tf import ssd
from eval.FID.fid_score import calculate_fid_given_paths


def load_clip(device):
    clip_model, preprocess = clip.load('ViT-B/32', device)
    clip_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
    clip_trans = transforms.Compose(
        [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model, clip_pool, clip_trans


def clip_text_embedding(text, clip_model, device):
    text = clip.tokenize(text).to(device)
    text_features = clip_model.encode_text(text)
    return text, text_features.float()


def clip_image_embedding(image, clip_model, device):
    image = image.to(device)
    image_features = clip_model.encode_image(image)
    return image_features.float()


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM == 1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            # netG.load_state_dict(state_dict)
            current_dict = netG.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in current_dict}
            # 2. overwrite entries in the existing state dict
            # current_dict.update(pretrained_dict)
            # 3. load the new state dict
            netG.load_state_dict(pretrained_dict)

            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')

            try:
                epoch = cfg.TRAIN.NET_G[istart:iend]
                epoch = int(epoch)
            except Exception:
                epoch = 0
            print(f'epoch {epoch}')

            if cfg.TRAIN.B_NET_D:
                try:
                    Gname = cfg.TRAIN.NET_G
                    for i in range(len(netsD)):
                        s_tmp = Gname[:Gname.rfind('/')]
                        Dname = '%s/netD_%d_%d.pth' % (s_tmp, epoch, i)
                        # Dname = '%s/netD%d.pth' % (s_tmp, i)
                        print('Load D from: ', Dname)
                        state_dict = \
                            torch.load(Dname, map_location=lambda storage, loc: storage)
                        netsD[i].load_state_dict(state_dict)
                except Exception:
                    pass
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch + 1]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)

        # for i in range(len(netsD)):
        #     netD = netsD[i]
        #     torch.save(netD.state_dict(),
        #                '%s/netD%d.pth' % (self.model_dir, i))
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                       '%s/netD_%d_%d.pth' % (self.model_dir, epoch, i))

        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png' \
                           % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png' \
                       % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        from model import SemanticAug, SemanticDiagR
        import torchvision.utils as vutils

        # p1, p2 = 0.3, 0.3
        # p1, p2 = 0.1, 0.1
        # p1, p2 = 0.05, 0.05

        # p1, p2 = 0.001, 0.001
        # p = 0.95
        #
        # if cfg.DATASET_NAME == 'birds':
        #     print('using birds')
        #     p = 0.05
        #     r_p = 0.1
        #     # r_p = 0
        #     aug = SemanticAug('/data1/phd21_zhaorui_tan/data_raw/cov/DAMSM/birds/', 256, p)
        #     R = SemanticDiagR('/data1/phd21_zhaorui_tan/data_raw/cov/DAMSM/birds/', 256, p)
        # else:
        #     p = 0.01
        #     r_p = 0.001
        #     # r_p = 0
        #     print('using coco')
        #     aug = SemanticAug('/data1/phd21_zhaorui_tan/data_raw/cov/DAMSM/coco/', 256, p)
        #     R = SemanticDiagR('/data1/phd21_zhaorui_tan/data_raw/cov/DAMSM/coco/', 256, p)
        # print(f'using aug {p, r_p}')

        warm_up = 0
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        mse_loss = nn.MSELoss()
        # gen_iterations = start_epoch * self.num_batches
        epoch = start_epoch -1
        while epoch < self.max_epoch:
        # for epoch in range(start_epoch, self.max_epoch):
            torch.cuda.empty_cache()
            start_t = time.time()
            backup_para = copy_G_params(netG)

            # data_iter = iter(self.data_loader)
            # step = 0
            # while step < self.num_batches:
            for step, data in enumerate(self.data_loader, 0):
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                # data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data_random_mask(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar, = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################

                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD_all = errD
                    nn.utils.clip_grad_norm(netsD[i].parameters(), max_norm=5, norm_type=2)

                    errD_all.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                # step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)

                errG, G_logs = \
                    generator_loss_random_mask(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)


                errG_total = errG + kl_loss
                netG.zero_grad()
                errG_total.backward()
                nn.utils.clip_grad_norm(netG.parameters(), max_norm=5, norm_type=2)
                optimizerG.step()



                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 50 == 0:
                    print('mask' + D_logs + G_logs)
                    # backup_para = copy_G_params(netG)
                    # load_params(netG, avg_param_G)
                    # fake_imgs_for_save = fake_imgs[-1].detach().cpu()
                    # fullpath = '%s/G_%d_%d.png' \
                    #            % (self.image_dir, gen_iterations, epoch)
                    # vutils.save_image(fake_imgs_for_save.data,
                    #                   fullpath,
                    #                   normalize=True)
                    # load_params(netG, backup_para)

                if torch.isnan(errG_total).any():
                    print('nan, no bp, backup')
                    load_params(netG, backup_para)
                    epoch -=1
                    break

            epoch += 1
            end_t = time.time()

            print('''mixup [%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))


            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)
            if epoch % 1 == 0:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)
                fake_imgs_for_save = fake_imgs[-1].detach().cpu()
                fullpath = '%s/G_%d_%d.png' \
                           % (self.image_dir, gen_iterations, epoch)
                vutils.save_image(fake_imgs_for_save.data,
                                  fullpath,
                                  normalize=True)
                load_params(netG, backup_para)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    # def sampling(self, split_dir):
    #     if cfg.TRAIN.NET_G == '':
    #         print('Error: the path for morels is not found!')
    #     else:
    #         if split_dir == 'test':
    #             split_dir = 'valid'
    #         # Build and load the generator
    #         if cfg.GAN.B_DCGAN:
    #             netG = G_DCGAN()
    #         else:
    #             netG = G_NET()
    #         netG.apply(weights_init)
    #         netG.cuda()
    #         netG.eval()
    #
    #         #
    #         n = 30000
    #         device = 'cuda'
    #         clip_model, clip_pool, clip_trans = load_clip(device)
    #         ixtoword = self.ixtoword
    #         fake_scores = np.zeros(n)
    #
    #         text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    #         state_dict = \
    #             torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    #         text_encoder.load_state_dict(state_dict)
    #         print('Load text encoder from:', cfg.TRAIN.NET_E)
    #         text_encoder = text_encoder.cuda()
    #         text_encoder.eval()
    #
    #         batch_size = self.batch_size
    #         nz = cfg.GAN.Z_DIM
    #         noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
    #         noise = noise.cuda()
    #
    #         model_dir = cfg.TRAIN.NET_G
    #         state_dict = \
    #             torch.load(model_dir, map_location=lambda storage, loc: storage)
    #         # state_dict = torch.load(cfg.TRAIN.NET_G)
    #         netG.load_state_dict(state_dict)
    #         print('Load G from: ', model_dir)
    #
    #         # the path to save generated images
    #         s_tmp = model_dir[:model_dir.rfind('.pth')]
    #         save_dir = '%s/%s' % (s_tmp, split_dir)
    #         mkdir_p(save_dir)
    #
    #         cnt = 0
    #         R_count = 0
    #         cont = True
    #
    #         for _ in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
    #             for step, data in enumerate(self.data_loader, 0):
    #                 torch.cuda.empty_cache()
    #                 cnt += batch_size
    #                 if R_count % 100 == 0:
    #                     print('R_count: ', R_count)
    #                 # if step > 50:
    #                 #     break
    #                 if (cont == False):
    #                     break
    #
    #                 imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
    #
    #                 hidden = text_encoder.init_hidden(batch_size)
    #                 # words_embs: batch_size x nef x seq_len
    #                 # sent_emb: batch_size x nef
    #                 words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    #                 words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    #                 mask = (captions == 0)
    #                 num_words = words_embs.size(2)
    #                 if mask.size(1) > num_words:
    #                     mask = mask[:, :num_words]
    #
    #                 #######################################################
    #                 # (2) Generate fake images
    #                 ######################################################
    #                 with torch.no_grad():
    #                     noise.data.normal_(0, 1)
    #                     # noise = torch.randn(batch_size, 100)
    #                     noise = noise.to(device)
    #                     fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
    #
    #                 sent_txt = []
    #                 for b in range(batch_size):
    #                     tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
    #                     tmp_s = ' '.join(tmp_w)
    #                     sent_txt.append(tmp_s)
    #
    #                 sent, sent_emb = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)
    #
    #                 imgs = imgs[-1].to(device)
    #                 real_for_clip = clip_trans(clip_pool(imgs))
    #                 real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)
    #
    #                 fake_for_clip = clip_trans(clip_pool(fake_imgs[-1]))
    #                 fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)
    #
    #                 for j in range(batch_size):
    #                     fake_for_clip[j] /= fake_for_clip[j].norm(dim=-1, keepdim=True)
    #                     real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
    #                     sent_emb /= sent_emb.norm(dim=-1, keepdim=True)
    #
    #                     clip_fake_score = fake_for_clip[j] @ sent_emb[1, :].T
    #                     fake_scores[R_count] = clip_fake_score
    #                     R_count += 1
    #
    #                     s_tmp = '%s/single/%s' % (save_dir, keys[j])
    #                     folder = s_tmp[:s_tmp.rfind('/')]
    #                     if not os.path.isdir(folder):
    #                         print('Make a new folder: ', folder)
    #                         mkdir_p(folder)
    #
    #                     caps_txt = sent_txt[j]
    #                     k = -1
    #                     # for k in range(len(fake_imgs)):
    #                     im = fake_imgs[k][j].data.cpu().numpy()
    #                     # [-1, 1] --> [0, 255]
    #                     im = (im + 1.0) * 127.5
    #                     im = im.astype(np.uint8)
    #                     im = np.transpose(im, (1, 2, 0))
    #                     im = Image.fromarray(im)
    #                     # fullpath = '%s_s%d.png' % (s_tmp, k)
    #                     fullpath = '%s_%3d_%.4f_%s.png' % (s_tmp, k, clip_fake_score, caps_txt)
    #                     im.save(fullpath)
    #
    #                     if R_count >= n:
    #                         print(f'R_count >= {R_count} ')
    #                         fake_final = np.zeros(10)
    #                         np.random.shuffle(fake_scores)
    #                         for i in range(10):
    #                             fake_final[i] = np.average(fake_scores[i * (n // 10):(i + 1) * (n // 10) - 1])
    #
    #                         fake_mean = np.average(fake_final)
    #                         fake_std = np.std(fake_final)
    #
    #                         print("fake mean:{:.4f} std:{:.4f}".format(fake_mean, fake_std))
    #                         cont = False

    def sampling(self, split_dir, use_aug=True):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            import tensorflow as tf
            from ssd_tf import ssd
            from eval.FID.fid_score import calculate_fid_given_paths

            gpu_ori_setting = os.environ['CUDA_VISIBLE_DEVICES']
            print(gpu_ori_setting)
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ori_setting.split(',')[-1]
            print(os.environ['CUDA_VISIBLE_DEVICES'])

            os.environ["MKL_NUM_THREADS"] = "1"
            gpu = tf.config.list_physical_devices('GPU')
            if split_dir == 'test':
                split_dir = 'valid'
            target_device = 'cpu'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()

            #
            n = 30000
            device = 'cuda'
            clip_model, clip_pool, clip_trans = load_clip(device)
            ixtoword = self.ixtoword
            fake_scores = np.zeros(n)

            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            for p in image_encoder.parameters():
                p.requires_grad = False
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            cont = True

            all_real = np.zeros((n, 256))
            all_fake = np.zeros((n, 256))
            all_caps = np.zeros((n, 256))

            for i in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    torch.cuda.empty_cache()
                    cnt += batch_size
                    if R_count % 100 == 0:
                        print('R_count: ', R_count)
                    # if step > 50:
                    #     break
                    if (cont == False):
                        break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    with torch.no_grad():
                        noise.data.normal_(0, 1)
                        # noise = torch.randn(batch_size, 100)
                        noise = noise.to(device)
                        fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

                    fake = fake_imgs[-1]
                    region_features, fake_cnn_code = image_encoder(fake)
                    # sent_txt = []
                    # for b in range(batch_size):
                    #     tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                    #     tmp_s = ' '.join(tmp_w)
                    #     sent_txt.append(tmp_s)
                    #
                    # sent, sent_emb = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)

                    real = imgs[-1].to(device)
                    region_features, real_cnn_code = image_encoder(real)
                    # real_for_clip = clip_trans(clip_pool(imgs))
                    # real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)
                    #
                    # fake_for_clip = clip_trans(clip_pool(fake_imgs[-1]))
                    # fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)

                    for j in range(batch_size):
                        # caps_txt = '_'.join([ixtoword[k] for k in captions[j].to('cpu').numpy().tolist() if k != 0])
                        # s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        # folder = s_tmp[:s_tmp.rfind('/')]
                        # if not os.path.isdir(folder):
                        #     print('Make a new folder: ', folder)
                        #     mkdir_p(folder)
                        # k = -1
                        # im = fake_imgs[k][j].data.cpu().numpy()
                        # # [-1, 1] --> [0, 255]
                        # im = (im + 1.0) * 127.5
                        # im = im.astype(np.uint8)
                        # im = np.transpose(im, (1, 2, 0))
                        # im = Image.fromarray(im)
                        # fullpath = '%s_%3d_%s.png' % (s_tmp, i, caps_txt)
                        # im.save(fullpath)

                        fake_cnn_code[j] /= fake_cnn_code[j].norm(dim=-1, keepdim=True)
                        real_cnn_code[j] /= real_cnn_code[j].norm(dim=-1, keepdim=True)
                        sent_emb[j] /= sent_emb[j].norm(dim=-1, keepdim=True)
                        all_real[R_count] = real_cnn_code[j].to('cpu').numpy()
                        all_fake[R_count] = fake_cnn_code[j].to('cpu').numpy()
                        all_caps[R_count] = sent_emb[j].to('cpu').numpy()
                        R_count += 1
                        if R_count % 1000 == 0:
                            print('R_count: ', R_count)

                        if R_count >= n:
                            cont = False
                            break

            print('data loaded')
            all_real = tf.convert_to_tensor(all_real)
            all_fake = tf.convert_to_tensor(all_fake)
            all_caps = tf.convert_to_tensor(all_caps)
            ssd_scores = ssd(all_real, all_fake, all_caps, )
            result = f'SSD:{ssd_scores[0]}, SS:{ssd_scores[1]}, dSV:{ssd_scores[2]}, TrSV:{ssd_scores[3]}, '
            print(result)

            if 'bird' in s_tmp:
                paths = [f'{save_dir}/single', '/data1/phd21_zhaorui_tan/data_raw/birds/CUB_200_2011/images']
            else:
                paths = [f'{save_dir}/single', '/data1/phd21_zhaorui_tan/data_raw/coco/train/val/val2014']

            fid_value = calculate_fid_given_paths(paths, batch_size, gpu[0], 2048)
            result += f'FID: {fid_value}'

            print(result)

    def sampling_cfid(self, split_dir):
        import tensorflow as tf
        from cfid import cfid
        os.environ['MKL_NUM_THREADS'] = '1'
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()

            device = 'cuda'
            clip_model, clip_pool, clip_trans = load_clip(device)

            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            # R = np.zeros(30000)
            n = 30000
            comp_scores = np.zeros(n)
            cont = True
            ixtoword = self.ixtoword

            all_real = np.zeros((n, 512))
            all_fake = np.zeros((n, 512))
            all_caps = np.zeros((n, 512))

            for _ in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    torch.cuda.empty_cache()
                    cnt += batch_size
                    if R_count % 100 == 0:
                        print('R_count: ', R_count)
                    # if step > 50:
                    #     break
                    if (cont == False):
                        break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    with torch.no_grad():
                        noise.data.normal_(0, 1)
                        # noise = torch.randn(batch_size, 100)
                        noise = noise.to(device)
                        fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

                    sent_txt = []
                    for b in range(batch_size):
                        tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                        tmp_s = ' '.join(tmp_w)
                        sent_txt.append(tmp_s)

                    sent, sent_emb = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)

                    imgs = imgs[-1].to(device)
                    real_for_clip = clip_trans(clip_pool(imgs))
                    real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)

                    fake_for_clip = clip_trans(clip_pool(fake_imgs[-1]))
                    fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)

                    for j in range(batch_size):
                        fake_for_clip[j] /= fake_for_clip[j].norm(dim=-1, keepdim=True)
                        real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
                        sent_emb[j] /= sent_emb[j].norm(dim=-1, keepdim=True)
                        all_real[R_count] = real_for_clip[j].detach().cpu().numpy()
                        all_fake[R_count] = fake_for_clip[j].detach().cpu().numpy()
                        all_caps[R_count] = sent_emb[j].detach().cpu().numpy()
                        R_count += 1
                        # print('R_count: ', R_count)

                        if R_count >= n:
                            print('data loaded')
                            all_real = tf.convert_to_tensor(all_real)
                            all_fake = tf.convert_to_tensor(all_fake)
                            all_caps = tf.convert_to_tensor(all_caps)
                            # all_real = tf.keras.utils.normalize(all_real, axis=-1, order=2)
                            # all_fake = tf.keras.utils.normalize(all_fake, axis=-1, order=2)
                            # all_caps = tf.keras.utils.normalize(all_caps, axis=-1, order=2)
                            fid = cfid(all_real, all_fake, all_caps, )
                            print(fid)
                            print(f'term1, {fid[0]}, term2,  {fid[1]}')
                            cont = False

            if R_count >= n:
                print('data loaded')
                all_real = tf.convert_to_tensor(all_real)
                all_fake = tf.convert_to_tensor(all_fake)
                all_caps = tf.convert_to_tensor(all_caps)
                # all_real = tf.keras.utils.normalize(all_real, axis=-1, order=2)
                # all_fake = tf.keras.utils.normalize(all_fake, axis=-1, order=2)
                # all_caps = tf.keras.utils.normalize(all_caps, axis=-1, order=2)
                fid = cfid(all_real, all_fake, all_caps, )
                print(f'term1, {fid[0]}, term2,  {fid[1]}')

    def sampling_ssd(self, split_dir):
        import tensorflow as tf
        from cfid import cfid
        os.environ['MKL_NUM_THREADS'] = '1'
        gpu = tf.config.list_physical_devices('GPU')
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()

            text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()

            from model_train_aug import SemanticAug
            aug = SemanticAug(cfg.TEXT.EMBEDDING_DIM).cuda()
            optimizerS = torch.optim.Adam(aug.parameters(),
                                          lr=0.0001,
                                          betas=(0.0, 0.9))
            aug_pth = cfg.TRAIN.NET_G.replace('netG', 'aug')

            state_dict = \
                torch.load(aug_pth, map_location=lambda storage, loc: storage)
            model_dict = aug.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(filtered_dict)
            aug.load_state_dict(model_dict)
            # netG.load_state_dict(state_dict)
            print('Load aug from: ', aug_pth)

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz))
            fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
            if cfg.CUDA:
                noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

            device = 'cuda'
            clip_model, clip_pool, clip_trans = load_clip(device)

            # text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            # state_dict = \
            #     torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            # text_encoder.load_state_dict(state_dict)
            # print('Load text encoder from:', cfg.TRAIN.NET_E)
            # text_encoder = text_encoder.cuda()
            # text_encoder.eval()
            #
            # batch_size = self.batch_size
            # nz = cfg.GAN.Z_DIM
            # noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            # noise = noise.cuda()
            #
            # model_dir = cfg.TRAIN.NET_G
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            # state_dict = \
            #     torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            # model_dict = netG.state_dict()
            # filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # model_dict.update(filtered_dict)
            # netG.load_state_dict(model_dict)
            # print('Load G from: ', model_dir)

            # model_dir = cfg.TRAIN.NET_G
            # state_dict = \
            #     torch.load(model_dir, map_location=lambda storage, loc: storage)
            # # state_dict = torch.load(cfg.TRAIN.NET_G)
            # netG.load_state_dict(state_dict)
            # print('Load G from: ', model_dir)

            # the path to save generated images
            model_dir = cfg.TRAIN.NET_G
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            # R = np.zeros(30000)
            n = 30000
            comp_scores = np.zeros(n)
            cont = True
            ixtoword = self.ixtoword

            all_real = np.zeros((n, 512))
            all_fake = np.zeros((n, 512))
            all_caps = np.zeros((n, 512))

            for i in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    torch.cuda.empty_cache()
                    cnt += batch_size
                    if R_count % 100 == 0:
                        print('R_count: ', R_count)
                    # if step > 50:
                    #     break
                    if (cont == False):
                        break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    with torch.no_grad():
                        noise.data.normal_(0, 1)
                        # noise = torch.randn(batch_size, 100)
                        noise = noise.to(device)
                        fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

                    sent_txt = []
                    for b in range(batch_size):
                        tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                        tmp_s = ' '.join(tmp_w)
                        sent_txt.append(tmp_s)

                    sent_clip, sent_emb_clip = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)

                    imgs = imgs[-1].to(device)
                    real_for_clip = clip_trans(clip_pool(imgs))
                    real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)

                    fake_for_clip = clip_trans(clip_pool(fake_imgs[-1]))
                    fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)

                    for j in range(batch_size):
                        caps_txt = '_'.join(
                            [ixtoword[k] for k in captions[j].to('cpu').numpy().tolist() if k != 0])
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_%3d_%s.png' % (s_tmp, i, caps_txt)
                        im.save(fullpath)

                        fake_for_clip[j] /= fake_for_clip[j].norm(dim=-1, keepdim=True)
                        real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
                        sent_emb_clip[j] /= sent_emb_clip[j].norm(dim=-1, keepdim=True)
                        all_real[R_count] = real_for_clip[j].detach().cpu().numpy()
                        all_fake[R_count] = fake_for_clip[j].detach().cpu().numpy()
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

                            if 'bird' in s_tmp:
                                paths = [f'{save_dir}/single',
                                         '/data1/phd21_zhaorui_tan/data_raw/birds/CUB_200_2011/images']
                            else:
                                paths = [f'{save_dir}/single',
                                         '/data1/phd21_zhaorui_tan/data_raw/coco/train/val/val2014']

                            fid_value = calculate_fid_given_paths(paths, batch_size, gpu[0], 2048)
                            result += f'FID: {fid_value}'
                            print(result)

            # if R_count >= n:
            #     print('data loaded')
            #     all_real = tf.convert_to_tensor(all_real)
            #     all_fake = tf.convert_to_tensor(all_fake)
            #     all_caps = tf.convert_to_tensor(all_caps)
            #     # all_real = tf.keras.utils.normalize(all_real, axis=-1, order=2)
            #     # all_fake = tf.keras.utils.normalize(all_fake, axis=-1, order=2)
            #     # all_caps = tf.keras.utils.normalize(all_caps, axis=-1, order=2)
            #     fid = cfid(all_real, all_fake, all_caps, )
            #     print(f'term1, {fid[0]}, term2,  {fid[1]}')

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
