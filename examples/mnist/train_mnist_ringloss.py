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

from gluonfr.loss import RingLoss
from mxnet.gluon.data.vision import MNIST
from mxnet import nd, gluon, metric as mtc, autograd as ag

from examples.mnist.net.mnist_net import MnistNet
from examples.mnist.utils import plot_result, transform_val, transform_train

os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


def validate(net, val_data, ctx, loss, plot=False):
    metric = mtc.Accuracy()
    val_loss = 0
    ebs = []
    lbs = []
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        ots = [net(X) for X in data]
        embedds = [ot[0] for ot in ots]
        outputs = [ot[1] for ot in ots]

        losses = [loss(yhat, y, emb) for yhat, y, emb in zip(outputs, labels, embedds)]
        metric.update(labels, outputs)
        val_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)
        if plot:
            for es, ls in zip(embedds, labels):
                assert len(es) == len(ls)
                for idx in range(len(es)):
                    ebs.append(es[idx].asnumpy())
                    lbs.append(ls[idx].asscalar())
    if plot:
        ebs = np.vstack(ebs)
        lbs = np.hstack(lbs)

    _, val_acc = metric.get()
    return val_acc, val_loss / len(val_data), ebs, lbs


def train():
    epochs = 100

    lr = 0.01
    lr_steps = [40, 70, np.inf]
    momentum = 0.9
    wd = 5e-4

    plot_period = 5

    ctx = [mx.gpu(i) for i in range(2)]
    batch_size = 256

    lamda = 0.01

    train_set = MNIST(train=True, transform=transform_train)
    train_data = gluon.data.DataLoader(train_set, batch_size, True, num_workers=4, last_batch='discard')
    val_set = MNIST(train=False, transform=transform_val)
    val_data = gluon.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4)

    net = MnistNet(embedding_size=2, feature_norm=False, weight_norm=True)
    # net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
    net.load_parameters("./pretrained_mnist.params", ctx=ctx)
    net.hybridize()

    loss = RingLoss(lamda)
    loss.initialize(ctx=ctx)
    loss.hybridize()

    train_params = net.collect_params()
    train_params.update(loss.params)
    trainer = gluon.Trainer(train_params, 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})

    lr_counter = 0

    metric = mtc.Accuracy()
    num_batch = len(train_data)

    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            lr_counter += 1
        if (epoch % plot_period) == 0:
            plot = True
        else:
            plot = False
        train_loss = 0
        metric.reset()
        tic = time.time()
        ebs = []
        lbs = []
        print("Radius", loss.R.data(ctx=mx.gpu(0)).asscalar())

        for batch in train_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            with ag.record():
                ots = [net(X) for X in data]
                embedds = [ot[0] for ot in ots]
                outputs = [ot[1] for ot in ots]
                losses = [loss(yhat, y, emb) for yhat, y, emb in zip(outputs, labels, embedds)]

            for l in losses:
                ag.backward(l)
            if plot:
                for es, ls in zip(embedds, labels):
                    assert len(es) == len(ls)
                    for idx in range(len(es)):
                        ebs.append(es[idx].asnumpy())
                        lbs.append(ls[idx].asscalar())

            trainer.step(batch_size)
            metric.update(labels, outputs)

            train_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)

        _, train_acc = metric.get()
        train_loss /= num_batch

        val_acc, val_loss, val_ebs, val_lbs = validate(net, val_data, ctx, loss, plot)

        if plot:
            ebs = np.vstack(ebs)
            lbs = np.hstack(lbs)

            plot_result(ebs, lbs, os.path.join("../resources", "ringloss-train-epoch{}.png".format(epoch)))
            plot_result(val_ebs, val_lbs, os.path.join("../resources", "ringloss-val-epoch{}.png".format(epoch)))

        toc = time.time()
        print('[epoch % 3d] train accuracy: %.6f, train loss: %.6f | '
              'val accuracy: %.6f, val loss: %.6f, time: %.6f'
              % (epoch, train_acc, train_loss, val_acc, val_loss, toc - tic))

        # if epoch == 10:
        #     net.save_parameters("./pretrained_mnist.params")
        #     net.save_parameters("./models/attention%d-cifar10-epoch-%d.params" % (args.num_layers, epoch))


if __name__ == '__main__':
    train()
