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
import mxnet as mx
import numpy as np
import sklearn
from tqdm import tqdm
from mxnet import gluon, autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from gluonfr.loss import ArcLoss
from gluonfr.nn.basic_blocks import NormDense
from gluonfr.data import get_recognition_dataset
from gluonfr.metrics.verification import FaceVerification
# from gluonfr.model_zoo.mobile_facenet import get_mobile_facenet
from gluonfr.model_zoo import get_attention_face

os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

transform_test = transforms.Compose([
    transforms.ToTensor()
])

_transform_train = transforms.Compose([
    transforms.RandomBrightness(0.3),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor()
])


def transform_train(data, label):
    im = _transform_train(data)
    # if use_mix_up:
    #     label = nd.one_hot(nd.array([label]), 10)[0]
    return im, label


class FaceNet(nn.HybridBlock):
    def __init__(self, classes, embedding_size=512, s=30, **kwargs):
        super().__init__(**kwargs)
        self.feature = get_attention_face(embedding_size, 56)
        self.output = NormDense(classes=classes, s=s, in_units=embedding_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        embedding = self.feature(x)
        output = self.output(embedding)
        return embedding, output


context = [mx.gpu(i) for i in range(2)]
batch_size = 64
margin_s = 64
margin_m = 0.5

train_set = get_recognition_dataset("vgg", transform=transform_train)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)

targets = 'lfw'
val_sets = [get_recognition_dataset(name, transform=transform_test) for name in targets.split(",")]
val_datas = [DataLoader(dataset, batch_size) for dataset in val_sets]

train_net = FaceNet(train_set.num_classes, s=margin_s)
train_net.initialize(init=mx.init.MSRAPrelu(), ctx=context)
train_net.hybridize()


def train(net, use_arcloss, ctx, s=64, m=0.5, iters=200e3, lr_steps=None, lr=0.1, momentum=0.9, wd=5e-4):
    trainer = gluon.Trainer(net.collect_params(), 'nag', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    if use_arcloss:
        loss = ArcLoss(s, m, classes=train_set.num_classes, easy_margin=False)
    else:
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    if lr_steps is None:
        lr_steps = [100e3, 140e3, 160e3]
    lr_steps.append(np.inf)

    # num_batch = len(train_data)
    it = 0
    train_loss = 0

    tic = time.time()
    while it < iters:
        if it == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            if lr_counter + 1 < len(lr_steps):
                lr_counter += 1

        for batch in tqdm(train_data):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            with ag.record():
                outputs = [net(X)[1] for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]
            for l in losses:
                ag.backward(l)

            trainer.step(batch_size)

            train_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)
            it += 1

            if (it % 1e2) == 0 and it != 0:
                train_loss /= 1e2

                toc = time.time()
                print('[it % 3d] train loss: %.6f | time: %.6f' % (it, train_loss, toc - tic))

                results = validate(net, ctx)
                for result in results:
                    print('{}'.format(result), end=" ")
                print("\n")
                train_loss = 0
                tic = time.time()
                net.save_parameters("../../models/mobile-facenet-it-%d.params" % it)


def validate(net, ctx, nfolds=10, norm=True):
    metric = FaceVerification(nfolds)
    results = []
    for loader, name in zip(val_datas, targets.split(",")):
        metric.reset()
        for i, batch in enumerate(loader):
            data0s = gluon.utils.split_and_load(batch[0][0], ctx, even_split=False)
            data1s = gluon.utils.split_and_load(batch[0][1], ctx, even_split=False)
            issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

            embedding0s = [net(X)[0] for X in data0s]
            embedding1s = [net(X)[0] for X in data1s]
            if norm:
                embedding0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
                embedding1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]

            for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
                metric.update(issame, embedding0, embedding1)

        tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
        results.append("{}: {:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))
    return results


if __name__ == '__main__':
    train(train_net, True, context, margin_s, margin_m, lr=0.1)
