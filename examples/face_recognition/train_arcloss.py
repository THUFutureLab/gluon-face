# MIT License
#
# Copyright (c) 2018 Haoxintong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""""""

import os
import time
import logging
import sklearn
import mxnet as mx
import numpy as np
from tqdm import tqdm
from mxnet import gluon, autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from gluonfr.loss import ArcLoss, SoftmaxCrossEntropyLoss
from gluonfr.nn.basic_blocks import NormDense
from gluonfr.data import get_recognition_dataset
from gluonfr.metrics.verification import FaceVerification
from gluonfr.model_zoo.mobile_facenet import get_mobile_facenet
from gluonfr.model_zoo import get_attention_face


transform_test = transforms.Compose([
    transforms.ToTensor()
])

_transform_train = transforms.Compose([
    transforms.RandomBrightness(0.3),
    transforms.RandomContrast(0.3),
    transforms.RandomSaturation(0.3),
    # transforms.RandomFlipLeftRight(),
    transforms.ToTensor()
])


def transform_train(data, label):
    im = _transform_train(data)
    # if use_mix_up:
    #     label = nd.one_hot(nd.array([label]), 10)[0]
    return im, label


class FaceNet(nn.HybridBlock):
    def __init__(self, classes, embedding_size=512, **kwargs):
        super().__init__(**kwargs)
        self.feature = get_mobile_facenet(embedding_size)
        # self.feature = get_attention_face(embedding_size, 56)
        self.output = NormDense(classes=classes, in_units=embedding_size, weight_norm=True, feature_norm=True)

    def hybrid_forward(self, F, x, *args, **kwargs):
        embedding = self.feature(x)
        if ag.is_training():
            return self.output(embedding)
        else:
            return embedding


context = [mx.gpu(i) for i in range(4)]
batch_size = 160*4
margin_s = 60
margin_m = 0.5
num_worker = 24

train_set = get_recognition_dataset("vgg", transform=transform_train)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_worker)

targets = ['lfw']
val_sets = [get_recognition_dataset(name, transform=transform_test) for name in targets]
val_datas = [DataLoader(dataset, batch_size, num_workers=num_worker) for dataset in val_sets]

train_net = FaceNet(train_set.num_classes)
train_net.initialize(init=mx.init.MSRAPrelu(), ctx=context)
train_net.hybridize(static_alloc=True, static_shape=True)

logger = logging.getLogger('TRAIN')
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("./attention-face.log"))


def train(net, ctx, s, m, iters=200e3, lr_steps=None, lr=0.1, momentum=0.9, wd=5e-4):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    # loss = ArcLoss(train_set.num_classes, s=s, m=m, easy_margin=False)
    loss = SoftmaxCrossEntropyLoss()
    lr_counter = 0
    if lr_steps is None:
        lr_steps = [100e3, 140e3, 160e3]
    lr_steps.append(np.inf)
    logger.info([s, m, lr_steps, lr, momentum, wd, batch_size])
    # num_batch = len(train_data)
    it = 0
    loss_mtc = mx.metric.Loss()
    metric = mx.metric.Accuracy()
    tic = time.time()
    while it < iters:
        if it == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            lr_counter += 1

        for batch in tqdm(train_data):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            with ag.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]
            for l in losses:
                ag.backward(l)
            trainer.step(batch_size)
            metric.update(labels, outputs)
            loss_mtc.update(0, losses)
            it += 1

            if (it % 500) == 0 and it != 0:
                _, train_loss = loss_mtc.get()
                _, train_acc = metric.get()
                toc = time.time()
                logger.info('[it % 3d] train loss: %.6f, train_acc: %.6f | time: %.6f' % (it, train_loss, train_acc,
                                                                                          toc - tic))

                results = validate(net, ctx)
                for result in results:
                    logger.info('{}'.format(result))
                logger.info("\n")
                loss_mtc.reset()
                metric.reset()
                tic = time.time()
                net.save_parameters("../../models/attention-face-it-%d.params" % it)


def validate(net, ctx, nfolds=10, norm=True):
    metric = FaceVerification(nfolds)
    results = []
    for loader, name in zip(val_datas, targets):
        metric.reset()
        for i, batch in enumerate(loader):
            data0s = gluon.utils.split_and_load(batch[0][0], ctx, even_split=False)
            data1s = gluon.utils.split_and_load(batch[0][1], ctx, even_split=False)
            issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

            embedding0s = [net(X) for X in data0s]
            embedding1s = [net(X) for X in data1s]
            if norm:
                embedding0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
                embedding1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]

            for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
                metric.update(issame, embedding0, embedding1)

        tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
        results.append("{}: {:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))
    return results


if __name__ == '__main__':
    train(train_net, context, margin_s, margin_m, lr=0.1)
