import sys
sys.path.append('../../')
from data.clip_dataloader.flickr_token import FlickrDataModule
import pytorch_lightning as pl
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import math
import copy
import argparse
from transformers import CLIPModel, BertForSequenceClassification
#from pl_bolts.optimizers import linear_warmup_decay

class CLIPLightning(pl.LightningModule):
    def __init__(self, model_name='ViT-B/32', minibatch_size=2):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.prepare_data_per_node = True
        self.model_name = 'ViT-B/32'
        #self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch32")
        #self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # NOTE load from openAI
        self.text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese")
        self.minibatch_size = minibatch_size
        self.isViT = 'ViT' in self.model_name
        self.automatic_optimization = False

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp

    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text, labels = train_batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)
        #ims = [F.normalize(self.clip_model.get_image_features(im), dim=1) for im in image_mbs]

        with torch.cuda.amp.autocast():
        # detach
            with torch.no_grad():
                ims = [F.normalize(self.clip_model.get_image_features(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.text_encoder(t).logits, dim=1) for t in text_mbs]
            # gather from all GPUs 这里的LOSS要把所有GPU的汇集起来一起算才对，torch.cat是 把一个batch 分的所有块的feature 放在一起，all_gather 是把所有卡的 参数收集到一起
            ims = self.all_gather(torch.cat(ims)).detach()
            txt = self.all_gather(torch.cat(txt))
            #print(ims.shape)

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.clip_model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) +
                    F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)},prog_bar=True)
            #self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            optimizer.zero_grad()

        # image loss
        # for j, mb in enumerate(image_mbs[:-1]):
        #     # 最后一部分样本舍弃。（对齐的bug）
        #     images_tmp = copy.deepcopy(ims)
        #     images_tmp[self.global_rank][j * self.minibatch_size:(j+1)*self.minibatch_size] = \
        #         F.normalize(self.clip_model.get_image_features(mb), dim=1)
        #     image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.clip_model.logit_scale.exp()
        #     ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
        #     loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
        #     self.manual_backward(loss)

            # text loss
            for j, mb in enumerate(text_mbs[:-1]):
                text_tmp = copy.deepcopy(txt)
                text_tmp[self.global_rank][j * self.minibatch_size:(j+1)*self.minibatch_size] = \
                    F.normalize(self.text_encoder(mb).logits, dim=1)
                image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.clip_model.logit_scale.exp()
                ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
                loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
                self.manual_backward(loss)

            optimizer.step()
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
            self.clip_model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    def validation_step(self, val_batch, idx):
        image, text, labels = val_batch
        img_embed = self.clip_model.get_image_features(image)
        txt_embed = self.text_encoder(text).logits
        # print(img_embed.shape)
        image_norm = F.normalize(img_embed, dim=1)
        text_norm = F.normalize(txt_embed, dim=1)
        image_logits = image_norm @ text_norm.t() * self.clip_model.logit_scale.exp()
        text_logits = text_norm @ image_norm.t() * self.clip_model.logit_scale.exp()
        # print(image_logits.shape)
        # image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss, prog_bar=True)
        return [image_norm, text_norm, labels]

    def validation_epoch_end(self, outputs):
        image_features = torch.cat([x[0] for x in outputs])
        text_features = torch.cat([x[1] for x in outputs])
        labels = [label for x in outputs for label in x[2]]
        #print(image_features.shape, text_features.shape, len(labels))
        self.get_metrics(image_features, text_features, labels, 100, 100)

    def test_step(self, test_batch, idx):
        image, text, labels = test_batch
        image_features = self.clip_model.get_image_features(image)
        text_features = self.text_encoder(text).logits
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return [image_features, text_features, labels]

    def test_epoch_end(self, outputs):
        image_features = torch.cat([x[0] for x in outputs])
        text_features = torch.cat([x[1] for x in outputs])
        labels = [label for x in outputs for label in x[2]]
        #print(image_features.shape, text_features.shape, len(labels))
        self.get_metrics(image_features, text_features, labels, 100, 100)

    def get_metrics(self, image_features, text_features, labels, logit_scale, n_batch):
        # 计算相似度，支持多个样本的情况（比如一个图片有多个caption）
        # img2txt计算的时候要用到，因为一张图片可能对应多个文本。
        # txt2img计算的时候不需要（一般一个text只有一个对应图片）
        # metrics = {}
        #   返回的是元祖类型
        text_feature_batchs = torch.chunk(text_features, n_batch, 0)
        logits_per_images = []
        logits_per_texts = []
        for text_feature_batch in text_feature_batchs:
            #image feature 1490*512 512*15(16*16*3)
            logits_per_image = (logit_scale * image_features @ text_feature_batch.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            logits_per_images.append(logits_per_image)
            logits_per_texts.append(logits_per_text)
        all_logits_per_images = torch.cat(logits_per_images, dim=1)
        all_logits_per_texts = torch.cat(logits_per_texts, dim=0)
        logits = {"image_to_text": all_logits_per_images, "text_to_image": all_logits_per_texts}

        label2idx = {}  # 计算label到idx的映射。
        repeat_id = []
        for i, label in enumerate(labels):
            if label not in label2idx:
                label2idx[label] = [i]
            else:
                # 表示该index的标签出现过，记录这个index，后续算txt2img分数的时候，这些index的权值要降低。
                label2idx[label].append(i)
                repeat_id.append(i)
        # print(label2idx)    # 标注了每个label的idx

        # print('repeat_id:', repeat_id)
        ground_truth = [label2idx[label] for label in labels]
        # print(ground_truth)

        for name, logit in logits.items():
            # print(name, logit.shape)
            if name == 'text_to_image':
                logit[:, repeat_id] -= 1e8  # 这部分的分数要降低。（重复出现的图片，直接忽略）
            r1_stat, r5_stat, r10_stat = [], [], []
            ranking = torch.argsort(logit, descending=True)  # index of the largest element to the smallest
            # print(name, ranking[:, :10])
            for i, each_query in enumerate(ranking[:, :10]):
                for j, q in enumerate(each_query):
                    if q in ground_truth[i]:
                        if j == 0:
                            r1_stat.append(1)
                            r5_stat.append(1)
                            r10_stat.append(1)
                            break
                        if j < 5:
                            r5_stat.append(1)
                            r10_stat.append(1)
                            break
                        if j < 10:
                            r10_stat.append(1)
                            break
            self.log_dict(
                {'r1': sum(r1_stat) / len(logit), 'r5': sum(r5_stat) / len(logit), 'r10': sum(r10_stat) / len(logit)},prog_bar=True)
            print(
                f'{name} r1:{sum(r1_stat) / len(logit)}, r5:{sum(r5_stat) / len(logit)}, r10:{sum(r10_stat) / len(logit)}')

    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            # "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/32": 4e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5
        }[self.model_name]

        optimizer = torch.optim.AdamW(
            [{'params': self.clip_model.parameters()}, {'params': self.text_encoder.parameters()}],
            lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )
        # lr_scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         linear_warmup_decay(warmup_step=1000, total_steps=self.num_epoches, cosine=True),
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20000000
        )
        # # CosineAnnealingWarmupRestarts
        # lr_scheduler = torch.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        #return {'optimizer': optimizer}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model_name
    parser.add_argument('--model', type=str,
                        default="ViT-B/32",
                        help='model definition')

    # experiment setting
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoches', type=int, default=200)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--auto_select_gpus', type=int, default=False)

    # dataset
    parser.add_argument('--train_filename', type=str,
                        help='dir or csv file')
    parser.add_argument('--train_root', type=str,
                        help='image root path')
    parser.add_argument('--val_filename', type=str,
                        help='dir or csv file')
    parser.add_argument('--val_root', type=str,
                        help='image root path')
    parser.add_argument('--test_filename', type=str,
                        help='dir or csv file')
    parser.add_argument('--test_root', type=str,
                        help='image root path')
    parser.add_argument('--num_workers', type=int, default=0)

    # huggingface pretrain model 定义
    parser.add_argument('--pretrain_model', type=str,
                        default="openai/clip-vit-base-patch32",
                        help='defalut load from openai')    # "wf-genius/TaiYi-CLIP-ViT-B-32" 是我训好的 NOTE

    args = parser.parse_args()
    dm = FlickrDataModule(args)

    model = CLIPLightning(model_name=args.model, minibatch_size=args.batch_size//2)
    trainer = pl.Trainer( accelerator="gpu", devices="2",auto_select_gpus = args.auto_select_gpus, precision=16, max_epochs=args.num_epoches)
    trainer.test(model, dm)  # zero-shot test
    #trainer.test(model,dm,ckpt_path="./fengshen/examples/clip_finetune/lightning_logs/version_25/checkpoints/epoch=99-step=1200.ckpt")
    #trainer.fit(model, dm)  # finetune on train set
    # trainer.test(model, dm,ckpt_path="./lightning_logs/version_25/checkpoints/epoch=99-step=1200.ckpt")  # zero-shot test
    trainer.test(model, dm,ckpt_path="./lightning_logs/version_28/checkpoints/epoch=149-step=1800.ckpt")  # finetune on train set
    #trainer.test(model, dm, ckpt_path="./lightning_logs/version_45/checkpoints/epoch=149-step=38400.ckpt")
    #trainer.test(model, dm)  # test again

