from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset, prepare_data
from trainer import condGANTrainer as trainer
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from model import RNN_ENCODER, CNN_ENCODER
from miscc.utils import mkdir_p
from PIL import Image
from torch.autograd import Variable
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = '/data1/phd21_zhaorui_tan/DAE-GAN-ori/output/base/%s_%s_%s' % \
             (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAE-GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_DAEGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=3)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--NET_G', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def get_generated_image(fake_img):
    img = fake_img.data.cpu().numpy()
    img = (img + 1.0) * 127.5
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img_ = np.array(Image.fromarray(img).resize((256, 256)))
    return img_

def visual(gpu, args):
    start_t = time.time()
    #   rank = args.nr * args.gpu_id + gpu
    #   dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    torch.cuda.set_device(gpu)
    if cfg.TRAIN.NET_G == '':
        print('Error: the path for morels is not found!')
    else:
        # Build and load the generator
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        else:
            netG = G_NET()
        netG.apply(weights_init)
        netG.cuda(gpu)
        netG.eval()

        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        dataset = TextDataset(cfg.DATA_DIR, "test",
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        print('dataset', dataset.n_words)
        assert dataset

        #      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
        #                                                                      num_replicas=args.world_size,
        #                                                                      rank=rank)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=int(cfg.WORKERS))

        algo = trainer(output_dir, train_loader, dataset.n_words, dataset.ixtoword, dataset)
        ixtoword = dataset.ixtoword

        # load text encoder
        # print(algo.n_words)
        text_encoder = RNN_ENCODER(algo.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        # print('state_dict:', state_dict.keys(), state_dict['encoder.weight'].shape)
        # print(text_encoder.encoder.weight.shape)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda(gpu)
        text_encoder.eval()

        # load image encoder
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        print('Load image encoder from:', img_encoder_path)
        image_encoder = image_encoder.cuda(gpu)
        image_encoder.eval()

        batch_size = algo.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
        noise = noise.cuda(non_blocking=True)

        model_dir = cfg.TRAIN.NET_G
        model_dir = '/'.join(model_dir.split('/')[:-1])
        print('model_dir', model_dir)
        model_list = [
            'netG_epoch_50.pth',
            'netG_epoch_100.pth',
            'netG_epoch_150.pth',
            'netG_epoch_200.pth',
            'netG_epoch_250.pth',
            'netG_epoch_300.pth',
            'netG_epoch_350.pth',
            'netG_epoch_400.pth',
            'netG_epoch_450.pth',
            'netG_epoch_500.pth',
            'netG_epoch_550.pth',
            'netG_epoch_600.pth',
        ]

        for model_name in model_list:
            tmp_model_dir = f'{model_dir}/{model_name}'

            state_dict = torch.load(tmp_model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_name[:model_name.rfind('.pth')]
            save_dir = '%s/visual/%s' % (model_dir,s_tmp,)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            max_count = 50
            R = np.zeros(max_count)
            cont = True
            for ii in range(2):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                if (cont == False):
                    break
                for step, data in enumerate(algo.data_loader, 0):
                    cnt += batch_size
                    if (cont == False):
                        break
                    if step % 10 == 0:
                        print('cnt: ', cnt)
                    if cnt > max_count:
                        break

                    imgs, captions, cap_lens, class_ids, keys, attrs = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                    # attrs processing
                    attr_len = torch.Tensor([cfg.MAX_ATTR_LEN] * cap_lens.size(0))
                    _, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attr_len, hidden)
                    _, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attr_len, hidden)
                    _, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attr_len, hidden)
                    attr_embs = torch.stack((attr_emb0, attr_emb1, attr_emb2), dim=2)  # [batch_size, nef, attr_num]

                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, attr_embs, mask, cap_lens)
                    for j in range(batch_size):
                        s_tmp = '%s' % (save_dir, )
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            # print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        # k = -1
                        # # for k in range(len(fake_imgs)):
                        # im = fake_imgs[k][j].data.cpu().numpy()
                        # # [-1, 1] --> [0, 255]
                        # im = (im + 1.0) * 127.5
                        # im = im.astype(np.uint8)
                        # im = np.transpose(im, (1, 2, 0))
                        # im = Image.fromarray(im)
                        # fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                        # im.save(fullpath)
                        for j in range(batch_size):
                            imgs_for_visual = []
                            for i in range(len(fake_imgs)):
                                fake_img = get_generated_image(fake_imgs[i][j])
                                imgs_for_visual.append(fake_img)

                            im = np.hstack(imgs_for_visual)
                            im = Image.fromarray(im)

                            filename = keys[j].replace("/", "_")
                            cap_txt = ' '.join(
                                [ixtoword[k] for k in captions[j].to('cpu').numpy().tolist() if k != 0])
                            fullpath = '%s/%d_%s_%s.png' % (save_dir, step, filename, cap_txt)
                            im.save(fullpath)
                            cont+=1


        end_t = time.time()
        print('Total time for training:', end_t - start_t)



def main():
    args = parse_args()
    args.world_size = args.gpu_id * args.nodes
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print(args.nodes)
    print(args.gpu_id)
    print(IPAddr)
    print(args.world_size)

    os.environ['MASTER_ADDR'] = IPAddr
    os.environ['MASTER_PORT'] = '5555'
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.NET_G != '':
        cfg.TRAIN.NET_G = args.NET_G

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    pprint.pprint(args)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    print("Seed: %d" % (args.manualSeed))

    # if cfg.TRAIN.FLAG:
    #     print(args.gpu_id)
    #     mp.spawn(train, nprocs=args.gpu_id, args=(args,))
    #     # train(0, args)
    #     # train(1, args)
    # else:
    visual(0, cfg)


if __name__ == "__main__":
    main()
