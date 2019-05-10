# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

import argparse
import os
import mxnet as mx
import sklearn
import numpy as np
from mxnet import gluon, nd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from gluonfr.model_zoo import get_model
from gluonfr.data import get_recognition_dataset
from gluonfr.metrics.verification import FaceVerification

parser = argparse.ArgumentParser(description='Train a margin based model for face recognition.')

parser.add_argument('--batch-size', type=int, default=512,
                    help='Training batch size.')
parser.add_argument('-n', '--model', type=str, default='l_se_resnet50v2',
                    help='Model to test.')
parser.add_argument('--model-params', type=str, required=True,
                    help='Model params to load.')
parser.add_argument('-t', '--val-dateset', dest='target', type=str, default='lfw',
                    help='Val datasets, default is lfw.'
                         'Options are lfw, calfw, cplfw, agedb_30, cfp_ff, vgg2_fp.')
parser.add_argument('--export', action='store_true',
                    help='Whether to export model.')
parser.add_argument('--export-path', type=str, default='',
                    help='Path to save export files.')
parser.add_argument('--ctx', type=str, default="0",
                    help='Use GPUs to train.')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--hybrid', action='store_true',
                    help='Whether to use hybrid.')
opt = parser.parse_args()

assert opt.batch_size % len(opt.ctx.split(",")) == 0, "Per batch on each GPU must be same."
assert opt.dtype in ('float32', 'float16'), "Data type only support FP16/FP32."
# targets = ['lfw', "calfw", "cplfw", "cfp_fp", "agedb_30", "cfp_ff", "vgg2_fp"]

transform_test = transforms.Compose([
    transforms.ToTensor()
])


def transform_test_flip(data, isf=False):
    flip_data = nd.flip(data, axis=1)
    if isf:
        data = nd.transpose(data, (2, 0, 1)).astype('float32')
        flip_data = nd.transpose(flip_data, (2, 0, 1)).astype('float32')
        return data, flip_data
    return transform_test(data), transform_test(flip_data)


export_path = os.path.dirname(opt.model_params) if opt.export_path == '' else opt.export_path
ctx = [mx.gpu(int(i)) for i in opt.ctx.split(",")]

batch_size = opt.batch_size

targets = opt.target
val_sets = [get_recognition_dataset(name, transform=transform_test_flip) for name in targets.split(",")]
val_datas = [DataLoader(dataset, batch_size, last_batch='keep') for dataset in val_sets]

dtype = opt.dtype
train_net = get_model(opt.model, need_cls_layer=False)
train_net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)


def validate(nfolds=10, ):
    metric = FaceVerification(nfolds)
    metric_flip = FaceVerification(nfolds)
    results = []
    for loader, name in zip(val_datas, targets.split(",")):
        metric.reset()
        for i, batch in enumerate(loader):
            data0s = gluon.utils.split_and_load(batch[0][0][0], ctx, even_split=False)
            data1s = gluon.utils.split_and_load(batch[0][1][0], ctx, even_split=False)
            data0s_flip = gluon.utils.split_and_load(batch[0][0][1], ctx, even_split=False)
            data1s_flip = gluon.utils.split_and_load(batch[0][1][1], ctx, even_split=False)
            issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

            embedding0s = [train_net(X) for X in data0s]
            embedding1s = [train_net(X) for X in data1s]
            embedding0s_flip = [train_net(X) for X in data0s_flip]
            embedding1s_flip = [train_net(X) for X in data1s_flip]

            emb0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
            emb1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]
            for embedding0, embedding1, issame in zip(emb0s, emb1s, issame_list):
                metric.update(issame, embedding0, embedding1)

            emb0s_flip = [sklearn.preprocessing.normalize(np.concatenate([e.asnumpy(), ef.asnumpy()], 1))
                          for e, ef in zip(embedding0s, embedding0s_flip)]
            emb1s_flip = [sklearn.preprocessing.normalize(np.concatenate([e.asnumpy(), ef.asnumpy()], 1))
                          for e, ef in zip(embedding1s, embedding1s_flip)]
            for embedding0, embedding1, issame in zip(emb0s_flip, emb1s_flip, issame_list):
                metric_flip.update(issame, embedding0, embedding1)

        tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
        print("{}: \t{:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))
        _, _, accuracy, _, _, _, accuracy_std = metric_flip.get()
        print("{}-flip: {:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))


if __name__ == '__main__':
    if opt.hybrid:
        train_net.hybridize()
    validate()
    if opt.export:
        assert opt.hybrid is True, 'Export need --hybrid to be True.'
        expot_name = os.path.join(export_path, opt.model)
        train_net.export(expot_name)
        print('export model is saved at {}'.format(expot_name))
