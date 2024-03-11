import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from lib.datasets import prepare_data, encode_tokens, clip_image_embedding

class SemanticAugCal(nn.Module):
    """
    ITA_C
    """
    def __init__(self, c_dir, cond_dim, p=0.95):
        super(SemanticAugCal, self).__init__()
        c = np.load(c_dir + 'c_text_true_given_image_true.npy')
        self.C1 = torch.diagonal(torch.from_numpy(c.reshape(cond_dim, cond_dim)) * p).cuda()
        print(self.C1, p)

    def forward(self, x, y):
        with torch.no_grad():
            aug = torch.mul(x, self.C1) + y
        return aug.float()


class SemanticDiagR(nn.Module):
    """
    L_r
    """
    def __init__(self, c_dir, cond_dim, p=0.95):
        super(SemanticDiagR, self).__init__()
        c = np.load(c_dir + 'c_image_true_given_text_true.npy')
        self.C1 = torch.diagonal(torch.from_numpy(c.reshape(cond_dim, cond_dim)) * p).cuda().to(torch.float32)
        print(self.C1, p)
        self.loss = nn.MSELoss()

    def forward(self, eff, ef, n):
        r = self.loss((eff - ef), n * self.C1)
        return r


class SemanticAug(nn.Module):
    """
    ITA_T
    """
    def __init__(self, cond_dim, ):
        super(SemanticAug, self).__init__()
        self.affine0 = Affine(cond_dim, )
        self.affine1 = Affine(cond_dim, )
        self.affine2 = Affine(cond_dim, )
        self.affine3 = Affine(cond_dim, )

    def forward(self, z, y=None):
        aug = self.affine0(z, y)
        aug = nn.LeakyReLU(0.2, inplace=True)(aug)
        aug = self.affine1(aug, y)
        aug = nn.LeakyReLU(0.2, inplace=True)(aug)
        return aug


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


class Affine(nn.Module):
    def __init__(self, cond_dim):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, cond_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cond_dim, cond_dim)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, cond_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cond_dim, cond_dim)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)
        return weight * x + bias
