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

from mxnet import nd, gluon, metric as mtc, autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import MNIST
from gluonfr.nn.basic_blocks import NormDense
from gluonfr.loss import ArcLoss

import numpy as np
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


def validate(net, val_data, ctx):
    metric = mtc.Accuracy()
    cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    val_loss = 0

    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        loss = [cross_entropy(yhat, y) for yhat, y in zip(outputs, labels)]
        metric.update(labels, outputs)
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

    _, val_acc = metric.get()
    return val_acc, val_loss / len(val_data)


def transform_train(data, label):
    im = data.astype('float32') / 255

    im = nd.transpose(im, (2, 0, 1))
    return im, label


def transform_val(data, label):
    im = data.astype('float32') / 255

    im = nd.transpose(im, (2, 0, 1))
    return im, label


class MnistNet(nn.HybridBlock):
    def __init__(self, use_arcloss, is_test=False, s=30, **kwargs):
        super().__init__(**kwargs)
        self.feature = nn.HybridSequential()
        self.feature.add(
            # nn.Conv2D(32, 5, 2, activation="relu"),
            #              nn.Conv2D(64, 5, 2, activation="relu"),
            #              nn.Flatten(),
                         nn.Dense(128, activation="sigmoid")
                         )
        self.embedding = nn.Dense(2)
        if is_test:
            self.output = None
        else:

            if use_arcloss:
                self.output = NormDense(classes=10, s=s)
            else:
                self.output = nn.Dense(10, use_bias=False)
        self._test = is_test

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.feature(x)
        embedding = self.embedding(x)
        if self.output is None:
            return embedding
        else:
            return self.output(embedding)


def train(net, use_arcloss, ctx, s=64, m=0.5, epochs=100, lr_steps=None, lr=0.01, momentum=0.9, wd=5e-4):
    trainer = gluon.Trainer(net.collect_params(), 'nag', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    if use_arcloss:
        loss = ArcLoss(s, m, classes=10, easy_margin=True)
    else:
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    if lr_steps is None:
        lr_steps = [50, 70]

    metric = mtc.Accuracy()
    num_batch = len(train_data)
    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            if lr_counter + 1 < len(lr_steps):
                lr_counter += 1

        train_loss = 0
        metric.reset()
        tic = time.time()
        for batch in train_data:

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            with ag.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]
            for l in losses:
                ag.backward(l)

            trainer.step(batch_size)
            metric.update(labels, outputs)

            train_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)

        _, train_acc = metric.get()
        train_loss /= num_batch
        val_acc, val_loss = validate(net, val_data, ctx)

        toc = time.time()
        print('[epoch % 3d] train accuracy: %.6f, train loss: %.6f | '
              'val accuracy: %.6f, val loss: %.6f, time: %.6f'
              % (epoch, train_acc, train_loss, val_acc, val_loss, toc - tic))
        # if (epoch % 10) == 0 and epoch != 0:
        #     net.save_parameters("./models/attention%d-cifar10-epoch-%d.params" % (args.num_layers, epoch))
    return net


def val(net, data, ctx):
    embeds = []
    labels = []
    for batch in data:
        batch_data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        batch_labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in batch_data]
        for ots, ls in zip(outputs, batch_labels):
            assert len(ots) == len(ls)
            for idx in range(len(ots)):
                embeds.append(ots[idx].asnumpy())
                labels.append(ls[idx].asscalar())
    embeds = np.vstack(embeds)
    labels = np.hstack(labels)
    # vis, plot code from https://github.com/pangyupo/mxnet_center_loss
    num = len(labels)
    names = dict()
    for i in range(10):
        names[i] = str(i)
    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(embeds[:, 0], embeds[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(embeds[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, names[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    if arc:
        fname = 'mnist-arcloss.png'
    else:
        fname = 'mnist-softmax.png'
    plt.savefig("../resources/" + fname)


if __name__ == '__main__':
    arc = False
    context = [mx.gpu(i) for i in range(2)]
    batch_size = 128
    margin_s = 64
    margin_m = 0.3
    #
    train_set = MNIST(train=True, transform=transform_train)
    train_data = gluon.data.DataLoader(train_set, batch_size, True, num_workers=4, last_batch='discard')
    val_set = MNIST(train=False, transform=transform_val)
    val_data = gluon.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=3)

    finetune_net = MnistNet(use_arcloss=True, s=margin_s)
    finetune_net.initialize()
    finetune_net.hybridize()
    finetune_net(nd.ones(shape=[1,64]))
    if arc:
        pretrained_net = MnistNet(use_arcloss=False, s=margin_s)
        pretrained_net.initialize(ctx=context)
        pretrained_net.hybridize()
        pretrained_net = train(pretrained_net, False, context, margin_s, margin_m, epochs=10)

        finetune_net.feature, finetune_net.embedding = pretrained_net.feature, pretrained_net.embedding

    finetune_net.initialize(ctx=context)
    finetune_net.hybridize()

    finetune_net = train(finetune_net, arc, context, margin_s, margin_m, lr=0.1, epochs=50, lr_steps=[25, 35])

    # val and visualization
    val_net = MnistNet(arc, True)
    val_net.feature, val_net.embedding = finetune_net.feature, finetune_net.embedding
    val_net.hybridize()
    val(val_net, val_data, context)
