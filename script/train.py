import sys
sys.path.append('./')
import os
import json
import numpy as np
import pickle
from datetime import datetime as dt
import argparse

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import reporter
from chainer import initializers
from chainer.dataset.convert import to_device
from chainer import training
from chainer.training import extensions

from functools import reduce
import tables

chainer.config.tiny_test = False
chainer.config.test = False
chainer.config.clean_description = True
chainer.config.remove_stopwords = False
chainer.config.raw_description = False

from parse import compile
import string
from keras.preprocessing.text import Tokenizer

nltk.download('stopwords')

e_stopwords = stopwords.words('english')
DESC_TEMPLATE_0 = compile('i should {} because {}')
DESC_TEMPLATE_1 = compile('{} because {}')
TABLE = str.maketrans('', '', string.punctuation)

def prepare_tokenizer(stopwords_removal_on=False):
    dataset = Dataset('train')

    action_train = []
    reason_train = []
    for i in range(len(dataset)):
        with chainer.using_config('remove_stopwords', stopwords_removal_on):
            item = dataset[i]
            action, reason, lable = target_transformer_action_reason_splitter(item)
            
        action_train += action
        reason_train += reason

    # prepare tokenizer for action
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(action_train + reason_train)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    with open('data/tokenizer%s.pickle' % ('_wo_stopwords' * stopwords_removal_on), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_wordemb(stopwords_removal_on=False):
    WORD2VEC_PATH = os.environ['WORD2VEC_PATH']
    
    if stopwords_removal_on:
        tokenizer_f = 'data/tokenizer_wo_stopwords.pickle'
    else:
        tokenizer_f = 'data/tokenizer.pickle'
        
    tokenizer = pickle.load(open(tokenizer_f, 'rb'))

    word_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

    M = np.zeros((len(tokenizer.word_index) + 1, 300)).astype('f')
    for w, i in tokenizer.word_index.items():
        try:
            wvec = word_vectors.get_vector(w)
        except:
            wvec = np.zeros((300,)).astype('f')
        M[i] = wvec
    
    if stopwords_removal_on:
        out_name = 'data/wordvec_wo_stopwords'
    else:
        out_name = 'data/wordvec'
    
    np.save(out_name, M)

def splitter(desc):
    res = DESC_TEMPLATE_0.parse(desc)

    if res is None:
        res = DESC_TEMPLATE_1.parse(desc)

    if res is None:
        res = desc[:len(desc)//2], desc[len(desc)//2:]
        
    return res

def description_transformer(item):
    if chainer.config.test:
        descriptions = item
    elif chainer.config.train:
        descriptions, _ = item
    elif not chainer.config.train: # validation
        _, descriptions = item
    
    actions = []
    reasons = []
    
    for desc in descriptions:
        act, rsn = splitter(desc)
        act = cleaning_description(act)
        rsn = cleaning_description(rsn)
        
        actions.append(act)
        reasons.append(rsn)
    
    return actions, reasons

def label_transformer(item):
    if chainer.config.test:
        return [0] * 15
    
    answer, candidates = item
    label = [1 if x in answer else 0 for x in candidates]
    return label

def target_transformer(target):
    answer, candidates = target
    label = [1 if x in answer else 0 for x in candidates]
    return candidates, label
    
def cleaning_description(desc):
    if chainer.config.clean_description:
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(TABLE) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        if chainer.config.remove_stopwords:
            desc = [word for word in desc if word not in e_stopwords]
        # concat
        desc = ' '.join(desc)
    return desc

def target_transformer_action_reason_splitter(target):
    
    if chainer.config.tiny_test or chainer.config.test:
        candidates = target
        label = [0] * len(candidates)
        if chainer.config.raw_description:
            return candidates, label
    else:
        candidates, label = target_transformer(target)
    
    actions = []
    reasons = []
    for c in candidates:
        res = splitter(c)
        
        action = cleaning_description(res[0])
        reason = cleaning_description(res[-1])
        
        actions.append(action)
        reasons.append(reason)
        
    return actions, reasons, label


def l2_distance(x0, x1):
    diff = x0 - x1
    dist_sq = F.sum(diff ** 2, axis=1)
    dist = F.sqrt(dist_sq)
    return dist

def eval_rank(dist, label):
    dist = F.reshape(dist, (-1, 15))
    label = F.reshape(label, (-1, 15))
    
    dist = dist.data
    label = label.data
    xp = chainer.cuda.get_array_module(dist)
    
    sort_i = xp.argsort(dist, axis=-1)
    ranks = xp.zeros(len(dist))
    
    for s_i, l, i in zip(sort_i, label, range(len(sort_i))):
        l = l[s_i]
        ranks[i] = xp.nonzero(l == 1)[0].min()
    
    r_1 = (ranks < 1).sum() / ranks.size
    r_5 = (ranks < 5).sum() / ranks.size
    r_10 = (ranks < 10).sum() / ranks.size
    
    return r_1, r_5, r_10

class TextCNN(chainer.Chain):
    def __init__(self, n_vocab, wvec):
        super(TextCNN, self).__init__()
        initialW = initializers.HeUniform()
        
        with self.init_scope():
            self.emb = L.EmbedID(n_vocab, 300, initialW=wvec, ignore_label=-1)
            self.conv_2 = L.Convolution2D(1, 100, ksize=(2, 300), stride=1, pad=(1,0), initialW=initialW)
            self.l = L.Linear(300, initialW=initializers.GlorotUniform())
    
    def word_embedding(self, x):
        x = F.pad_sequence(x, padding=-1)
        x = self.emb(x)
        return x
    
    def __call__(self, x):
        x = F.pad_sequence(x, padding=-1)
        x = self.emb(x)
        x = F.expand_dims(x, axis=1)
        h = F.relu(self.conv_2(x))
        h = F.max(h, axis=2)
        h = F.tanh(self.l(h))
        return h
    
class TextLSTM(chainer.Chain):
    def __init__(self, n_vocab, wvec):
        super(TextLSTM, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(n_vocab, 300, initialW=wvec, ignore_label=-1)
            self.lstm = L.NStepLSTM(1, 300, 1000, dropout=0.0)
    
    def word_embedding(self, x):
        x = F.pad_sequence(x, padding=-1)
        x = self.emb(x)
        return x
    
    def __call__(self, X):
        h = [self.emb(x) for x in X]
        h, _, _ = self.lstm(None, None, h)    
        h = h[0] # hidden stae of the last layer
        return h

class AttentionNetWTL(chainer.Chain):
    def __init__(self, h_size=500):
        super(AttentionNetWTL, self).__init__()
        with self.init_scope():            
            # attention net
            self.fc_ar = L.Linear(None, h_size, initialW=initializers.LeCunNormal())
            self.fc_ap = L.Linear(None, h_size, initialW=initializers.LeCunNormal())        

    def __call__(self, phr_feats, region_feats):
        hr = self.fc_ar(F.tanh(region_feats))
        N_v, C_v = hr.shape
        hr = F.reshape(hr, (-1, 10, C_v))

        hp = self.fc_ap(phr_feats)
        N_l, C_l = hp.shape
        hp = F.reshape(hp, (-1, 15, C_l))

        a = F.matmul(hr, hp, transb=True) # (B, N_v, N_l)
        a = F.softmax(a, axis=0)
        
        # compute weighted features
        _, C_v = region_feats.shape
        region_feats = F.reshape(region_feats, (-1, 10, C_v))
        wf = F.matmul(region_feats, a, transa=True) # (B, C_v, N_l)
        wf = F.transpose(wf, axes=(0, 2, 1))
        return wf, a

def compute_weigted_feat(region_feats, att):
    att = F.expand_dims(att, axis=-1)
    region_feats = region_feats * F.broadcast_to(att, region_feats.shape)
    return F.sum(region_feats, axis=1)


def word_attention(emb_a, emb_b):
    A = F.matmul(emb_a, emb_b, transb=True)
    A = F.sum(A, axis=-1, keepdims=True)
    A = F.softmax(A, axis=-2)
    
    B, N, C = emb_a.shape
    wf = F.matmul(emb_a, A, transa=True)
    wf = F.transpose(wf, axes=(0, 2, 1))
    return wf, A

class NonVisualNet(chainer.Chain):
    def __init__(self, lng_net, h_size=200, margin=.4):
        super(NonVisualNet, self).__init__()
        
        initialW = initializers.HeUniform()
        
        with self.init_scope():
            self.lng_net = lng_net
            self.ocr_l = L.Linear(None, h_size, initialW=initializers.HeNormal())
            self.act_l = L.Linear(None, h_size, initialW=initializers.HeNormal())
            self.rsn_l = L.Linear(None, h_size, initialW=initializers.HeNormal())
            
        
        self.margin = margin
    
    def encode_ocr(self, yocr_emb, y_emb):
        h_ocr, att = word_attention(yocr_emb, y_emb)
        h_ocr = F.squeeze(h_ocr)
        h_ocr = F.relu(self.ocr_l(h_ocr))
        return h_ocr
    
    def predict(self, v_feat, actions, reasons, label, ocr, layers=['dist']):
        outputs = {}
        
        # word embedding
        ya_emb = self.lng_net.word_embedding(actions)
        yr_emb = self.lng_net.word_embedding(reasons)        
        yocr_emb = self.lng_net.word_embedding(ocr)
        yocr_emb = F.repeat(yocr_emb, 15, axis=0)

        # action feat
        h_act = self.lng_net(actions)
        h_act = F.relu(self.act_l(h_act))
        h_act = F.normalize(h_act)
        
        # reason feat
        h_rsn = self.lng_net(reasons)
        h_rsn = F.relu(self.rsn_l(h_rsn))
        h_rsn = F.normalize(h_rsn)
        
        # attention over ocr words (action)
        h_ocr_act = self.encode_ocr(yocr_emb, ya_emb)
        h_ocr_act = F.normalize(h_ocr_act)
        
        loss_act = F.contrastive(h_ocr_act, h_act, label, margin=self.margin)
        
        # attention over ocr words (reason)
        h_ocr_rsn = self.encode_ocr(yocr_emb, yr_emb)
        h_ocr_rsn = F.normalize(h_ocr_rsn)
        
        loss_rsn = F.contrastive(h_ocr_rsn, h_rsn, label, margin=self.margin)
        
        outputs['loss'] = (loss_act + loss_rsn) * .5
        
        # distance between ocr feature and target text
        if 'dist' in layers:
            dist_ocr = l2_distance(h_ocr_act, h_act)
            dist_rsn = l2_distance(h_ocr_rsn, h_rsn)
            dist = (dist_ocr + dist_rsn) * .5
            outputs['dist'] = dist
        
        return outputs
    
    def __call__(self, v_feat, actions, reasons, label, ocr):
        layers = ['loss'] if chainer.config.train else ['loss', 'dist']
        outputs = self.predict(v_feat, actions, reasons, label, ocr, layers=layers)
    
        loss = outputs['loss']
        reporter.report({'loss': loss}, self)
        
        if not chainer.config.train:
            r_1, r_5, r_10 = eval_rank(outputs['dist'], label)
            reporter.report({'r@1': r_1, 'r@5': r_5, 'r@10': r_10, 'ranking_score': r_1 + r_5 + r_10}, self)
        
        if not self.xp.isfinite(loss.array).all():
            print('label N: %i, min: %i, max: %i' % label.size, label.min(), label.max())
            
        return loss
        

class Net(NonVisualNet):
    def __init__(self, 
                 lng_net,
                 att_net,
                 h_size=200, 
                 margin=1.):
        
        super(Net, self).__init__(lng_net, h_size, margin)
        
        
        initialW = initializers.HeUniform()
        
        with self.init_scope():
            self.att_net = att_net
            self.l_vis_act = L.Linear(None, h_size, initialW=initialW)
            self.l_vis_rsn = L.Linear(None, h_size, initialW=initialW)
            self.l_lng_act = L.Linear(None, h_size, initialW=initialW)
            self.l_lng_rsn = L.Linear(None, h_size, initialW=initialW)
    
    def ocr_mapping_net(self, h_ocr_act, h_act, h_ocr_rsn, h_rsn):
        h_act = F.relu(self.act_l(h_act))
        h_act = F.normalize(h_act)
        
        h_rsn = F.relu(self.rsn_l(h_rsn))
        h_rsn = F.normalize(h_rsn)
        
        h_ocr_act = F.relu(h_ocr_act)
        h_ocr_act = F.normalize(h_ocr_act)

        h_ocr_rsn = F.relu(h_ocr_rsn)
        h_ocr_rsn = F.normalize(h_ocr_rsn)
        
        return h_ocr_act, h_act, h_ocr_rsn, h_rsn
    
    def vis_mapping_net(self, v_feat, h_act, h_rsn):
        # embed to action manifold
        h_vis_act = self.l_vis_act(v_feat)
        h_vis_act, _ = self.att_net(h_act, h_vis_act) # (B, N_l, C)
        _, _, C = h_vis_act.shape
        h_vis_act = F.reshape(h_vis_act, (-1, C))
        h_vis_act = F.relu(h_vis_act)
        h_vis_act = F.normalize(h_vis_act)

        # embed to reason manifold
        h_vis_rsn = self.l_vis_rsn(v_feat)
        h_vis_rsn, _ = self.att_net(h_rsn, h_vis_rsn) # (B x N_l, C)
        h_vis_rsn = F.reshape(h_vis_rsn, (-1, C))
        h_vis_rsn = F.relu(h_vis_rsn)
        h_vis_rsn = F.normalize(h_vis_rsn)
        
        h_act = F.relu(self.l_lng_act(h_act))
        h_act = F.normalize(h_act)
        
        h_rsn = F.relu(self.l_lng_rsn(h_rsn))
        h_rsn = F.normalize(h_rsn)
        
        return h_vis_act, h_act, h_vis_rsn, h_rsn
        

        
    def predict(self, v_feat, actions, reasons, label, ocr, layers=['dist'], alpha=.9):
        outputs = {}
        ocr_on = self.xp.asarray([x.size > 0 for x in ocr], dtype=np.float32).repeat(15)
        
        # word embedding
        ya_emb = self.lng_net.word_embedding(actions)
        yr_emb = self.lng_net.word_embedding(reasons)        
        yocr_emb = self.lng_net.word_embedding(ocr)
        yocr_emb = F.repeat(yocr_emb, 15, axis=0)

        # action feat
        h_act = self.lng_net(actions)
        
        # reason feat
        h_rsn = self.lng_net(reasons)
        
        # attention over ocr words (action)
        h_ocr_act = self.encode_ocr(yocr_emb, ya_emb)

        # attention over ocr words (reason)
        h_ocr_rsn = self.encode_ocr(yocr_emb, yr_emb)
        
        y_ocr_act, y_act, y_ocr_rsn, y_rsn = self.ocr_mapping_net(h_ocr_act, h_act, h_ocr_rsn, h_rsn)
        
        # loss ocr
        loss_act = F.contrastive(y_ocr_act, y_act, label, margin=self.margin, reduce='no')
        loss_rsn = F.contrastive(y_ocr_rsn, y_rsn, label, margin=self.margin, reduce='no')
        loss_ocr = ocr_on * (loss_act + loss_rsn) * .5
        
        z_vis_act, z_act, z_vis_rsn, z_rsn = self.vis_mapping_net(v_feat, h_act, h_rsn)
        
        # loss visual
        loss_act_vis = F.contrastive(z_vis_act, z_act, label, margin=self.margin, reduce='no')
        loss_rsn_vis = F.contrastive(z_vis_rsn, z_rsn, label, margin=self.margin, reduce='no')
        loss_vis = (loss_act_vis + loss_rsn_vis) * .5
        
        loss = F.mean(alpha * loss_ocr + (1-alpha) * loss_vis)
        # loss = F.mean(loss_ocr)
        
        if 'loss_ocr' in layers:
            outputs['loss_ocr'] = loss_ocr
        
        if 'loss_vis' in layers:
            outputs['loss_vis'] = loss_vis
        
        if 'loss' in layers:
            outputs['loss'] = loss
        
        # distance between ocr feature and target text
        if 'dist' in layers:
            dist_act = l2_distance(y_ocr_act, y_act)
            dist_rsn = l2_distance(y_ocr_rsn, y_rsn)
            dist_ocr = ocr_on * (dist_act + dist_rsn) * .5
        
            dist_act = l2_distance(z_vis_act, z_act)
            dist_rsn = l2_distance(z_vis_rsn, z_rsn)
            dist_vis = (dist_act + dist_rsn) * .5
            outputs['dist'] = alpha * dist_ocr + (1-alpha) * dist_vis
            
        if 'dist_ocr' in layers:
            outputs['dist_ocr'] = dist_ocr
        
        if 'dist_vis' in layers:
            outputs['dist_vis'] = dist_vis
            
        return outputs
    
    def __call__(self, img, actions, reasons, label, t_att):
        layers = ['loss'] if chainer.config.train else ['loss', 'dist']
        outputs = self.predict(img, actions, reasons, label, t_att, layers=layers)
    
        loss = outputs['loss']
        reporter.report({'loss': loss}, self)
        
        if not chainer.config.train:
            r_1, r_5, r_10 = eval_rank(outputs['dist'], label)
            reporter.report({'r@1': r_1, 'r@5': r_5, 'r@10': r_10, 'ranking_score': r_1 + r_5 + r_10}, self)
        
        if not self.xp.isfinite(loss.array).all():
            print('label N: %i, min: %i, max: %i' % label.size, label.min(), label.max())
            
        return loss
    
def my_converter(batch, device=None):
    img = np.vstack([b[0] for b in batch])
    img = to_device(device, img)
    
    actions = [b[1] for b in batch]
    actions = [item for sublist in actions for item in sublist]
    actions = [to_device(device, np.asarray(x, dtype=np.int32)) for x in actions]
    
    reasons = [b[2] for b in batch]
    reasons = [item for sublist in reasons for item in sublist]
    reasons = [to_device(device, np.asarray(x, dtype=np.int32)) for x in reasons]
    
    label = [b[3] for b in batch]
    label = to_device(device, np.asarray(label, dtype=np.int32).ravel())
    
    ocr = [to_device(device, np.asarray(b[4]).astype('i')) for b in batch]
    
    return img, actions, reasons, label, ocr

class Dataset(chainer.dataset.DatasetMixin):
    
    def __init__(self, split, san_check=False):
        self.h5file = tables.open_file('data/frcnn_feat/%s.h5' % split)
        self.qa_data = json.load(open('data/%s/QA_Combined_Action_Reason_%s.json' % (split, split)))
            
        self.images = list(self.qa_data.keys())
        
        if san_check:
            self.images = self.images[-500:]

    def __len__(self):
        return len(self.images)
    
    def get_row_answer(self, i):
        im_name = self.images[i]
        item = self.qa_data[im_name]
        return item
    
    def get_example(self, i):
        return self.get_row_answer(i)
    
class DatasetOCR(Dataset):
    def __init__(self, split, ocr_type='cloudvision', san_check=False):
        super(DatasetOCR, self).__init__(split, san_check)
        
        if chainer.config.remove_stopwords:
            tokenizer_f = 'data/tokenizer_wo_stopwords.pickle'
        else:
            tokenizer_f = 'data/tokenizer.pickle'
        
        print('load %s' % tokenizer_f)
            
        self.tokenizer = pickle.load(open(tokenizer_f, 'rb'))
        
        self.ocr_df = pd.read_csv('data/ocr.csv')
        self.pref = '%s_images/' % split
        
        self.ocr_type = ocr_type
        
    def get_ocr(self, i):
        im_name = self.images[i]
        key = self.pref + im_name
        
        if self.ocr_type == 'cloudvision':
            text = self.ocr_df.text_by_cloudvision[self.ocr_df.image_path == key]
        elif self.ocr_type == 'tesseract':
            text = self.ocr_df.text_by_tesseract[self.ocr_df.image_path == key]
        else:
            raise RuntimeError
            
        text = text.values[0]
        
        if isinstance(text, float):
            return ''
        
        text = ' '.join(text.splitlines())
        return text

    def get_example(self, i):
        im_name = self.images[i]
        item = self.qa_data[im_name]
        node = self.h5file.get_node('/'+im_name.split('.')[0])
        
        x = np.zeros((10, 4096)).astype('f')
        feat = node.feat.read()
        x[:len(feat)] = feat
        targ = target_transformer_action_reason_splitter(item)
        actions, reasons, l = targ
        
        actions = self.tokenizer.texts_to_sequences(actions)
        y_a = [x if len(x) else [-1] for x in actions] # [-1] for empty description
        
        reasons = self.tokenizer.texts_to_sequences(reasons)
        y_r = [x if len(x) else [-1] for x in reasons] # [-1] for empty description
        
        ocr = self.get_ocr(i)
        ocr = self.tokenizer.texts_to_sequences([ocr])
        ocr = ocr[0][:400] # truncate detected words
        
        return x, y_a, y_r, l, ocr

    
class myTokenizer(Tokenizer):
    def fit_on_vocab(self, vocab):

        for w in vocab:
            if w in self.word_counts:
                self.word_counts[w] += 1
            else:
                self.word_counts[w] = 1
        for w in set(vocab):
            if w in self.word_docs:
                self.word_docs[w] += 1
            else:
                self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

def evaluate(model_dir, device=None):
    args = json.load(open(os.path.join(model_dir, 'args')))
    
    chainer.config.remove_stopwords = args['remove_stopwords']
    ocr_type = args['ocr_type']
    
    test = DatasetOCR('test', ocr_type)
    
    text_net = args['text_net']
    if text_net == 'cnn':
        lng_net = TextCNN(len(test.tokenizer.word_index) + 1, None)
    elif text_net == 'lstm':
        lng_net = TextLSTM(len(test.tokenizer.word_index) + 1, None)
    else:
        raise RuntimeError('invalid text_net')
    
    h_size=args['h_size']
    margin = args['margin']
    model_name = args['model_name']

    if model_name == 'ocr':
        model = NonVisualNet(lng_net, h_size=h_size, margin=margin)
    elif model_name == 'ocr+vis':
        att_net = AttentionNetWTL(h_size=100)
        model = Net(lng_net, att_net)
    else:
        raise RuntimeError
    
    chainer.serializers.load_npz(os.path.join(model_dir, 'model'), model)
    
    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()
    
    results = {}
    b_size = 100
    
    with chainer.using_config('train', False), chainer.using_config('test', True), chainer.using_config('enable_backprop', False):
        for i in range(0, len(test), b_size):
            item = test[i: i + b_size]
            img_id = test.images[i:i+b_size]

            with chainer.using_config('clean_description', False):
                desc = [test.get_row_answer(j) for j in range(i, min(len(test)-1, i+b_size))]

            batch = my_converter(item, device=device)
            outputs = model.predict(*batch, layers=['dist'])
            dist = outputs['dist']

            dist.to_cpu()
            dist = dist.data.ravel()
            dist = np.reshape(dist, (-1, 15))


            sort_i = dist.argsort(axis=1)

            for im_i, des, s_i in zip(img_id, desc, sort_i):
                results[im_i] = des[s_i[0]]
    
    json.dump(results, open(os.path.join(model_dir, 'results.txt'), 'w'))
            
        
    
def train(
    lr = 0.001,
    device = 0,
    epoch = 10,
    h_size= 1000,
    b_size = 100,
    weight_decay = 0.0005,
    margin=1.,
    saveto = 'output/checkpoint/',
    text_net = 'cnn',
    ocr_type = 'cloudvision',
    model_name = 'ocr',
    san_check = False,
    early_stopping=False,
    remove_stopwords=False,
    ):
    
    chainer.config.remove_stopwords = remove_stopwords
    
    args = locals()
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    json.dump(args, open(os.path.join(saveto,'args'), 'w'))

    log_interval = (10, 'iteration')
    val_interval = (1, 'epoch')
    
    dataset = DatasetOCR('train', ocr_type=ocr_type, san_check=san_check)
    train, val = chainer.datasets.split_dataset_random(dataset, first_size=int(len(dataset) * .9), seed=1234)
        
    print('train: %i, val: %i' % (len(train), len(val)))
    train_itr = chainer.iterators.SerialIterator(train, batch_size=b_size)
    val_itr = chainer.iterators.SerialIterator(val, batch_size=b_size, repeat=False, shuffle=False)
    
    if remove_stopwords:
        wvec_f = 'data/wordvec_wo_stopwords.npy'
    else:
        wvec_f = 'data/wordvec.npy'
    word_vec = np.load(wvec_f)
    
    if text_net == 'cnn':
        lng_net = TextCNN(len(dataset.tokenizer.word_index) + 1, word_vec)
    elif text_net == 'lstm':
        lng_net = TextLSTM(len(dataset.tokenizer.word_index) + 1, word_vec)
    else:
        raise RuntimeError('invalid text_net')
    
        
    if model_name == 'ocr':
        model = NonVisualNet(lng_net, h_size=h_size, margin=margin)
    elif model_name == 'ocr+vis':
        att_net = AttentionNetWTL(h_size=100)
        model = Net(lng_net, att_net)
    else:
        raise RuntimeError
    
    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.use_cleargrads()
    optimizer.setup(model)
    
    if text_net == 'lstm':
        optimizer.add_hook(chainer.optimizer.GradientClipping(5), name='grad_clip')
    
    if weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay), name='weight_decay')
    
    updater = training.StandardUpdater(train_itr, optimizer, converter=my_converter, device=device)
    
    stop_trigger = (epoch, 'epoch')
    
    if early_stopping:
        stop_trigger = training.triggers.EarlyStoppingTrigger(
            monitor='validation/main/r@1', patients=2, mode='max', verbose=True, max_trigger=(epoch, 'epoch'))
        
    trainer = training.Trainer(updater, stop_trigger, saveto)
    trainer.extend(extensions.FailOnNonNumber())
    trainer.extend(extensions.Evaluator(val_itr, model, converter=my_converter, device=device),
    trigger=val_interval)
    
    if not san_check:
        trainer.extend(extensions.ExponentialShift('alpha', 0.5), trigger=(5, 'epoch'))

    trainer.extend(extensions.LogReport(trigger=log_interval))
    
    trainer.extend(extensions.ProgressBar(update_interval=10))

    best_val_trigger = training.triggers.MaxValueTrigger('validation/main/ranking_score', trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, 'model'), trigger=best_val_trigger)

    trainer.run()

    return best_val_trigger._best_value
    
if not os.path.exists('data/tokenizer.pickle'):
    print('Preparing word tokenizer')
    prepare_tokenizer()
    
if not os.path.exists('data/tokenizer_wo_stopwords.pickle'):
    print('Preparing word tokenizer with stop words removal')
    prepare_tokenizer(stopwords_removal_on=True)

if not os.path.exists('data/wordvec.npy'):
    print('Pre-computing word embeddings')
    save_wordemb()

if not os.path.exists('data/wordvec_wo_stopwords.npy'):
    print('Pre-computing word embeddings with stop words removal')
    save_wordemb(stopwords_removal_on=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '-lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', '-wd', type=float, default=None)
    parser.add_argument('--margin', '-m', type=float, default=1.)
    parser.add_argument('--text_net', '-tn', type=str, default='cnn')
    parser.add_argument('--ocr_type', '-ot', type=str, default='cloudvision')
    parser.add_argument('--model_name', type=str, default='ocr')
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--b_size', '-b', type=int, default=100,
                        help='minibatch size <int> (default 100)')
    parser.add_argument('--h_size', type=int, default=1000)
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='max epoch')
    parser.add_argument('--san_check', '-sc', action='store_true', 
    help='sanity check mode')
    parser.add_argument('--early_stopping', '-e_stop', action='store_true', 
    help='turn on early stop trigger')
    parser.add_argument('--remove_stopwords', '-stopwords', action='store_true', 
    help='turn on stopwords removal')
    parser.add_argument('--saveto', type=str, default='output/checkpoint/')
    parser.add_argument('--eval', type=str, default=None,
                       help='output directory')

    args = parser.parse_args()
    
    if args.eval is not None:
        evaluate(args.eval, args.device)
    else:
        time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
        saveto = args.saveto + 'sc_' * args.san_check + args.model_name + time_stamp

        train(
        lr = args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        device = args.device,
        epoch = args.epoch,
        b_size = args.b_size,
        h_size = args.h_size,
        saveto = saveto,
        text_net = args.text_net,
        ocr_type = args.ocr_type,
        model_name = args.model_name,
        early_stopping=args.early_stopping,
        remove_stopwords=args.remove_stopwords,
        san_check = args.san_check
        )
