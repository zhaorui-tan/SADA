from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
import nltk
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import copy

random.seed(42)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# nltk.data.path.append("Your nltk package path")
nltk.download('averaged_perceptron_tagger')


def prepare_data(data):
    # imgs, captions, captions_lens, class_ids, keys, attrs = data
    # imgs, captions, captions_lens, class_ids, keys, attrs, vs, mis_cap, mis_attrs, mis_vs = data

    imgs, class_ids, keys, \
    caps, cap_lens, aspects, ns, attrs, vs, \
    mis_caps, cap_lens, mis_ns, mis_attrs, mis_vs, = data
    # caps_2, cap_lens_2, ns_2, attrs_2, vs_2

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(cap_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    # sorted_cap_lens_2, sorted_cap_indices_2 = \
    #     torch.sort(cap_lens_2, 0, True)
    # imgs_2 = imgs.copy()
    # real_imgs_2 = []
    # for i in range(len(imgs_2)):
    #     imgs_2[i] = imgs_2[i][sorted_cap_indices_2]
    #     if cfg.CUDA:
    #         real_imgs_2.append(Variable(imgs_2[i]).cuda())
    #     else:
    #         real_imgs_2.append(Variable(imgs_2[i]))

    caps = caps[sorted_cap_indices].squeeze()
    aspects = aspects[sorted_cap_indices].squeeze()
    ns = ns[sorted_cap_indices].squeeze()
    attrs = attrs[sorted_cap_indices].squeeze()
    vs = vs[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices]

    # caps_2 = caps[sorted_cap_indices_2].squeeze()
    # ns_2 = ns_2[sorted_cap_indices].squeeze()
    # attrs_2 = attrs[sorted_cap_indices_2].squeeze()
    # vs_2 = vs[sorted_cap_indices_2].squeeze()
    # class_ids_2 = class_ids[sorted_cap_indices_2]

    mis_caps = mis_caps[sorted_cap_indices].squeeze()
    mis_ns = mis_ns[sorted_cap_indices].squeeze()
    mis_attrs = mis_attrs[sorted_cap_indices].squeeze()
    mis_vs = mis_vs[sorted_cap_indices].squeeze()

    keys = [keys[i] for i in sorted_cap_indices]
    # sent_indices = sent_indices[sorted_cap_indices]

    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        caps = Variable(caps).cuda()
        aspects = Variable(aspects).cuda()
        ns = Variable(ns).cuda()
        attrs = Variable(attrs).cuda()
        vs = Variable(vs).cuda()

        # caps_2 = Variable(caps_2).cuda()
        # ns_2 = Variable(ns_2).cuda()
        # attrs_2 = Variable(attrs_2).cuda()
        # vs_2 = Variable(vs_2).cuda()

        mis_caps = Variable(mis_caps).cuda()
        mis_ns = Variable(mis_ns).cuda()
        mis_attrs = Variable(mis_attrs).cuda()
        mis_vs = Variable(mis_vs).cuda()

        # sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        # sorted_cap_lens_2 = Variable(sorted_cap_lens_2).cuda()
    else:
        caps = Variable(caps)
        aspects = Variable(aspects)
        ns = Variable(ns)
        attrs = Variable(attrs)
        vs = Variable(vs)

        # caps_2 = Variable(caps_2)
        # ns_2 = Variable(ns_2)
        # attrs_2 = Variable(attrs_2)
        # vs_2 = Variable(vs_2)

        mis_caps = Variable(mis_caps)
        mis_ns = Variable(mis_ns)
        mis_attrs = Variable(mis_attrs)
        mis_vs = Variable(mis_vs)

        sorted_cap_lens = Variable(sorted_cap_lens)

        # sorted_cap_lens_2 = Variable(sorted_cap_lens_2)
    # print(captions.shape, attrs.shape)
    # return [real_imgs, real_imgs_2,
    #         class_ids, class_ids_2,
    #         sorted_cap_indices, sorted_cap_indices_2, keys,
    #         caps, sorted_cap_lens, ns, attrs, vs,
    #         caps_2, sorted_cap_lens_2, ns_2, attrs_2, vs_2,
    #         mis_caps, sorted_cap_lens, mis_ns, mis_attrs, mis_vs, ]

    return [real_imgs,
            class_ids,
            sorted_cap_indices, keys,
            caps, sorted_cap_lens, aspects, ns, attrs, vs,
            mis_caps, sorted_cap_lens, mis_ns, mis_attrs, mis_vs, ]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    ret.append(normalize(img))
    # if cfg.GAN.B_DCGAN:
    '''
    for i in range(cfg.TREE.BRANCH_NUM):
        # print(imsize[i])
        re_img = transforms.Resize(imsize[i])(img)
        ret.append(normalize(re_img))
    '''

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 # base_size=64,
                 base_size=16,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM + 2):
            # print("base_size, i:", base_size, i)
            self.imsize.append(base_size)
            if i < 3 or i == (cfg.TREE.BRANCH_NUM + 2 - 2):
                base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        # self.filenames, self.captions, self.attrs, self.ixtoword, \
        # self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.filenames, self.captions, self.aspects, \
        self.n, self.attrs, self.v, \
        self.n_list, self.a_list, self.v_list, \
        self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        # self.filenames, self.captions, self.attrs, self.v, \
        # self.a_list, self.v_list, self.n_list, \
        # self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.cand_list = self.n_list + self.a_list + self.v_list

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        print(f'loading bbox data from {bbox_path}')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        print('loaded')
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        print(f'loading captions data')
        all_captions = []
        all_aspects = []
        all_n = []
        all_a = []
        all_v = []

        global_n_list = []
        global_a_list = []
        global_v_list = []

        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r", encoding='utf-8') as f:
                captions = f.read().split('\n')
                # captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    # filter too short caps
                    if len(cap) < 2:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # sentence = tokenizer.tokenize(text.lower())
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue
                    # Attribute extraction
                    sentence_tag = nltk.pos_tag(tokens)

                    # CUB
                    grammar = """
                                NP: {<DT>*<JJ>*<CC|IN>*<JJ>+<NN|NNS>+|<DT>*<NN|NNS>+<VBZ>+<JJ>+<IN|CC>*<JJ>*}
                                A: {<JJ|JJR|JJS>} 
                                V: {<V.*>}
                                N: {<N.*>}
                            """

                    # grammar = """
                    #             A: {<JJ|JJR|JJS>}
                    #             V: {<V.*>}
                    #             N: {<N.*>}
                    #             NP: {<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+|<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+<IN>+<NN|NNS>+|<VB|VBD|VBG|VBN|VBP|VBZ>+<CD|DT>*<JJ|PRP$>*<NN|NNS>+|<IN>+<DT|CD|JJ|PRP$>*<NN|NNS>+<IN>*<CD|DT>*<JJ|PRP$>*<NN|NNS>*}
                    #             LNP: {<NP> <p>? <NP>+}
                    #             VP: { <DT|WDT>? <NP|N|LNP>* <V>+ <DT|WDT>? <NP|N|LNP>*}
                    #         """

                    cp = nltk.RegexpParser(grammar)
                    tree = cp.parse(sentence_tag)
                    aspects_list = []
                    all_n_list = []
                    all_a_list = []
                    all_v_list = []
                    for ctree in tree.subtrees():
                        # if ctree.label() == 'VP':
                        #     local_v_list = [_[0] for _ in ctree.leaves()]
                        #     all_v_list.append(local_v_list)
                        # elif ctree.label() == 'NP' or ctree.label() == 'LNP':
                        if ctree.label() == 'NP':
                            local_n_list = [_[0] for _ in ctree.leaves()]
                            aspects_list.append(local_n_list)

                        elif ctree.label() == 'A':
                            global_a_list += [_[0] for _ in ctree.leaves()]
                        elif ctree.label() == 'N':
                            global_n_list += [_[0] for _ in ctree.leaves()]
                        elif ctree.label() == 'V':
                            global_v_list += [_[0] for _ in ctree.leaves()]

                    all_aspects.append(aspects_list if len(aspects_list) > 0 else [[]])
                    all_n.append(all_n_list if len(all_n_list) > 0 else [[]])
                    all_a.append(all_a_list if len(all_n_list) > 0 else [[]])
                    all_v.append(all_v_list if len(all_v_list) > 0 else [[]])

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)  # [cnt, num_words]
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                #  print("len(captions):",len(captions))
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d, len(captions)=%d, embedding_num=%d'
                          % (filenames[i], cnt, len(captions), self.embeddings_num))

        print(f'len cap, atter, v: {len(all_captions)}, {len(all_n)}, {len(all_v)}')
        print('loaded')
        return all_captions, all_aspects, all_n, all_a, all_v, list(set(global_a_list)), list(set(global_v_list)), list(
            set(global_n_list))

    def build_dictionary(self,
                         train_captions, test_captions,
                         train_aspects, test_aspects,
                         train_n, test_n,
                         train_attrs, test_attrs,
                         train_v, test_v,
                         train_a_list, test_a_list,
                         train_v_list, test_v_list,
                         train_n_list, test_n_list):
        print('building dictionary')
        word_counts = defaultdict(float)
        captions = train_captions + test_captions

        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        # process captions
        def rev_caption(ori_token_list):
            new_token_list = []
            for t in ori_token_list:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                new_token_list.append(rev)
            return new_token_list

        train_captions_new = rev_caption(train_captions)
        test_captions_new = rev_caption(test_captions)

        # process noun, attrs and verbs
        def rev_n_a_v(ori_token_list):
            new_token_list = []
            for t in ori_token_list:
                rev = []
                for tt in t:
                    _new = []
                    for w in tt:
                        if w in wordtoix:
                            _new.append(wordtoix[w])
                    rev.append(_new)
                new_token_list.append(rev)
            return new_token_list

        train_aspects_new = rev_n_a_v(train_aspects)
        test_aspects_new = rev_n_a_v(test_aspects)
        train_attrs_new = rev_n_a_v(train_attrs)
        test_attrs_new = rev_n_a_v(test_attrs)
        train_n_new = rev_n_a_v(train_n)
        test_n_new = rev_n_a_v(test_n)
        train_v_new = rev_n_a_v(train_v)
        test_v_new = rev_n_a_v(test_v)

        # process a v n

        a_list_new = list(set(train_a_list + test_a_list))
        v_list_new = list(set(train_v_list + test_v_list))
        n_list_new = list(set(train_n_list + test_n_list))

        def rev_a_v_n(ori_token_list):
            new_token_list = []
            for w in ori_token_list:
                if w in wordtoix:
                    new_token_list.append(wordtoix[w])
            return new_token_list

        a_list_new = rev_a_v_n(a_list_new)
        v_list_new = rev_a_v_n(v_list_new)
        n_list_new = rev_a_v_n(n_list_new)

        print('built')
        return [train_captions_new, test_captions_new,
                train_aspects_new, test_aspects_new,
                train_n_new, test_n_new,
                train_attrs_new, test_attrs_new,
                train_v_new, test_v_new,
                a_list_new, v_list_new, n_list_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions_dae.pickle')
        print("filepath:", filepath)
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            # if os.path.isfile(filepath):
            print(f'saving text data to {filepath}')
            train_captions, train_aspects, train_n, train_attrs, train_v, \
            train_a_list, train_v_list, train_n_list = self.load_captions(data_dir, train_names)
            test_captions, test_aspects, test_n, test_attrs, test_v, \
            test_a_list, test_v_list, test_n_list = self.load_captions(data_dir, test_names)
            # print(train_a_list, train_v_list, train_n_list)

            train_captions, test_captions, \
            train_aspects, test_aspects, \
            train_n, test_n, \
            train_attrs, test_attrs, \
            train_v, test_v, \
            a_list, v_list, n_list, \
            ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions,
                                      train_aspects, test_aspects,
                                      train_n, test_n,
                                      train_attrs, test_attrs,
                                      train_v, test_v,
                                      train_a_list, test_a_list,
                                      train_v_list, test_v_list,
                                      train_n_list, test_n_list
                                      )

            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             train_aspects, test_aspects,
                             train_n, test_n,
                             train_attrs, test_attrs,
                             train_v, test_v,
                             a_list, v_list, n_list,
                             ixtoword, wordtoix, n_words], f, protocol=2)
                print('Save to: ', filepath)

        else:
            print(f'loading text data from {filepath}')
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions, \
                train_aspects, test_aspects, \
                train_n, test_n, \
                train_attrs, test_attrs, \
                train_v, test_v, \
                a_list, v_list, n_list, \
                ixtoword, wordtoix, n_words = \
                    x[0], x[1], \
                    x[2], x[3], \
                    x[4], x[5], \
                    x[6], x[7], \
                    x[8], x[9], \
                    x[10], x[11], x[12], \
                    x[13], x[14], x[15]

                # train_captions, test_captions, train_attrs, test_attrs = x[0], x[1], x[4], x[5]
                # ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            aspects = train_aspects
            n = train_n
            attrs = train_attrs
            v = train_v
            filenames = train_names

        else:  # split=='test'
            captions = test_captions
            aspects = test_aspects
            n = test_n
            attrs = test_attrs
            v = test_v
            filenames = test_names
        # print(a_list, v_list, n_list,)
        return filenames, captions, aspects, n, attrs, v, n_list, a_list, v_list, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        print('loading class id')
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        print('loaded')
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_key(self, sent_ix, key_list):
        # sen_k = np.asarray(self.ks[sent_ix]).astype('int64')
        sen_k = key_list[sent_ix]
        # num_ks = len(sen_k)  # num of k per sentence
        # sen_k_new = []
        k_cnt = 0
        # new
        sen_k_new = np.zeros((cfg.MAX_ATTR_NUM, cfg.MAX_ATTR_LEN, 1), dtype='int64')

        for k in sen_k:
            k = np.asarray(k).astype('int64')
            # print(k.shape, "====", k)
            k_cnt = k_cnt + 1
            if k_cnt > cfg.MAX_ATTR_NUM:
                break

            k_len = len(k)
            if k_len <= cfg.MAX_ATTR_LEN:
                sen_k_new[k_cnt - 1][:k_len, 0] = k
            else:
                ix = list(np.arange(k_len))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:cfg.MAX_ATTR_LEN]
                ix = np.sort(ix)
                sen_k_new[k_cnt - 1][:, 0] = k[ix]
        return sen_k_new

    def sample_tokens(self, token, cand_list):
        mis_token = None
        cur_token_id = cand_list.index(token)
        while not mis_token:
            new_token_id = random.randint(0, len(cand_list))
            if cur_token_id != new_token_id:
                mis_token = cand_list[new_token_id]
            else:
                continue
        return mis_token

    def replace_token(self, token, mis_token, mis_cap, mis_n, mis_attrs, mis_v):
        mis_cap[np.where(mis_cap == token)] = mis_token
        mis_n[np.where(mis_n == token)] = mis_token
        mis_attrs[np.where(mis_attrs == token)] = mis_token
        mis_v[np.where(mis_v == token)] = mis_token
        return mis_cap, mis_n, mis_attrs, mis_v

    def get_mis_a_v_n(self, cap, n, attrs, v, cap_len, token_replace_ratio=0.3):
        """
        Args:
            cap: whole caps
            n: attr + n
            attrs: attr
            v: n + attr + v
            cap_len:

        Returns:

        """
        # TODO check
        mis_cap = copy.deepcopy(cap)
        mis_n = copy.deepcopy(n)
        mis_attrs = copy.deepcopy(attrs)
        mis_v = copy.deepcopy(v)

        # do 20 times sampling at most
        # print('cap', cap,)
        # print('n', n,)
        # print('attrs',attrs)
        # print('len', cap_len)
        i = 20
        token_replace = int(cap_len * token_replace_ratio)
        replaced_token = []
        while i > 0 and token_replace > 0:
            token_id = random.randint(0, cap_len)
            token = cap[token_id][0]

            if token in self.a_list and token not in replaced_token:
                mis_token = self.sample_tokens(token, self.a_list)
                mis_cap, mis_n, mis_attrs, mis_v = self.replace_token(token, mis_token, mis_cap, mis_n, mis_attrs,
                                                                      mis_v)
                replaced_token.append(token)
                replaced_token.append(mis_token)
                # return mis_cap, mis_n, mis_attrs, mis_v

            elif token in self.n_list and token not in replaced_token:
                mis_token = self.sample_tokens(token, self.n_list)
                mis_cap, mis_n, mis_attrs, mis_v = self.replace_token(token, mis_token, mis_cap, mis_n, mis_attrs,
                                                                      mis_v)
                replaced_token.append(token)
                replaced_token.append(mis_token)
                # return mis_cap, mis_n, mis_attrs, mis_v

            elif token in self.v_list and token not in replaced_token:
                mis_token = self.sample_tokens(token, self.v_list)
                mis_cap, mis_n, mis_attrs, mis_v = self.replace_token(token, mis_token, mis_cap, mis_n, mis_attrs,
                                                                      mis_v)
                replaced_token.append(token)
                replaced_token.append(mis_token)
                # return mis_cap, mis_n, mis_attrs, mis_v

            else:
                i -= 1
                token_replace -= 1

        if token_replace == 0:
            return mis_cap, mis_n, mis_attrs, mis_v

        else:
            print('mis a_n_v sample is not available ')
            print(mis_cap, mis_n, mis_attrs, mis_v)
            if cap_len > 1 and i == 0:
                print('mask half')
                mask_tokens = mis_cap[(cap_len // 2):]
                for t in mask_tokens:
                    token = t[0]
                    mis_cap, mis_n, mis_attrs, mis_v = self.replace_token(token, 0, mis_cap, mis_n, mis_attrs,
                                                                          mis_v)
                print(mis_cap, mis_n, mis_attrs, mis_v)
            if cap_len == 1 and i == 0:
                print('too short, replace')
                token = mis_cap[0][0]
                mis_token_id = random.randint(0, len(self.n_list))
                mis_token = self.n_list[mis_token_id]
                mis_cap, mis_n, mis_attrs, mis_v = self.replace_token(token, mis_token, mis_cap, mis_n, mis_attrs,
                                                                      mis_v)
                print(mis_cap, mis_n, mis_attrs, mis_v)
            return mis_cap, mis_n, mis_attrs, mis_v

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        cap, cap_len = self.get_caption(new_sent_ix)

        aspects = self.get_key(new_sent_ix, self.aspects)
        ns = self.get_key(new_sent_ix, self.n)
        attrs = self.get_key(new_sent_ix, self.attrs)
        vs = self.get_key(new_sent_ix, self.v)

        # # random a pos sample
        # sent_ix = random.randint(0, self.embeddings_num)
        # new_sent_ix = index * self.embeddings_num + sent_ix
        # cap_2, cap_len_2 = self.get_caption(new_sent_ix)
        # ns_2 = self.get_key(new_sent_ix, self.n)
        # attrs_2 = self.get_key(new_sent_ix, self.attrs)
        # vs_2 = self.get_key(new_sent_ix, self.v)
        # attrs_2 = self.get_attr(new_sent_ix)
        # vs_2 = self.get_v(new_sent_ix)
        # create neg sample

        # create a hard negative sample
        # print('cap', cap.shape,)
        # print('n', ns.shape,)
        # print('attrs',attrs.shape)
        # print('len', cap_len)
        mis_cap, mis_ns, mis_attrs, mis_vs = self.get_mis_a_v_n(cap, ns, attrs, vs, cap_len)

        return imgs, cls_id, key, \
               cap, cap_len, aspects, ns, attrs, vs, \
               mis_cap, cap_len, mis_ns, mis_attrs, mis_vs,
        # cap_2, cap_len_2, ns_2, attrs_2, vs_2

    # def get_mis_caption(self, cls_id):
    #     mis_match_captions_t = []
    #     mis_match_captions = torch.zeros(99, cfg.TEXT.WORDS_NUM)
    #     mis_match_captions_len = torch.zeros(99)
    #     i = 0
    #     while len(mis_match_captions_t) < 99:
    #         idx = random.randint(0, self.number_example)
    #         if cls_id == self.class_id[idx]:
    #             continue
    #         sent_ix = random.randint(0, self.embeddings_num)
    #         new_sent_ix = idx * self.embeddings_num + sent_ix
    #         caps_t, cap_len_t = self.get_caption(new_sent_ix)
    #         mis_match_captions_t.append(torch.from_numpy(caps_t).squeeze())
    #         mis_match_captions_len[i] = cap_len_t
    #         i = i + 1
    #     sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)
    #     # import ipdb
    #     # ipdb.set_trace()
    #     for i in range(99):
    #         mis_match_captions[i, :] = mis_match_captions_t[sorted_cap_indices[i]]
    #     return mis_match_captions.type(torch.LongTensor).cuda(), sorted_cap_lens.type(torch.LongTensor).cuda()

    def get_mis_caption(self, cap, cap_len, ns, attrs, vs, n=5):
        mis_match_captions_t = []
        mis_match_captions = torch.zeros(n, cfg.TEXT.WORDS_NUM)
        mis_match_captions_len = torch.zeros(n)
        i = 0

        cap, cap_len, ns, attrs, vs = cap.unsqueeze(-1).to('cpu').numpy(), cap_len, \
                                      ns.unsqueeze(-1).to('cpu').numpy(), attrs.unsqueeze(-1).to('cpu').numpy(), \
                                      vs.unsqueeze(-1).to('cpu').numpy()

        while len(mis_match_captions_t) < n:
            # idx = random.randint(0, self.number_example)
            # if cls_id == self.class_id[idx]:
            #     continue
            # sent_ix = random.randint(0, self.embeddings_num)
            # new_sent_ix = idx * self.embeddings_num + sent_ix
            # caps_t, cap_len_t = self.get_caption(new_sent_ix)
            mis_cap, mis_ns, mis_attrs, mis_vs = self.get_mis_a_v_n(cap, ns, attrs, vs, cap_len)
            mis_match_captions_t.append(torch.from_numpy(mis_cap).squeeze())
            mis_match_captions_len[i] = cap_len
            i = i + 1

        sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)
        for i in range(n):
            mis_match_captions[i, :] = mis_match_captions_t[sorted_cap_indices[i]]
        return mis_match_captions.type(torch.LongTensor).cuda(), sorted_cap_lens.type(torch.LongTensor).cuda()

    def __len__(self):
        return len(self.filenames)
