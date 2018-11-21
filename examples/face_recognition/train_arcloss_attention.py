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
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import time
import logging
import mxnet as mx
import numpy as np
from tqdm import tqdm
from mxnet import gluon, autograd as ag
from mxnet.gluon.data import DataLoader

from gluonfr.loss import *
from gluonfr.model_zoo import *
from gluonfr.data import *
from examples.face_recognition.utils import transform_train, transform_test, validate


def set_grad_add(params):
    for k, param in params.items():
        param.grad_req = "add"


def set_grad_zero(params):
    for param in params:
        params[param].zero_grad()


num_gpu = 4
num_worker = 16
ctx = [mx.gpu(i) for i in range(num_gpu)]
batch_size_per_gpu = 32
batch_size = batch_size_per_gpu * num_gpu

expect_batch_size = 1024

save_period = 1000
iters = 200e3
lr_steps = [30e3, 60e3, 120e3, 180e3, np.inf]

scale = 64
margin = 0.5
embedding_size = 256

lr = 0.0001
momentum = 0.9
wd = 5e-4

train_set = get_recognition_dataset("emore", transform=transform_train)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_worker)

targets = ['lfw']
val_sets = [get_recognition_dataset(name, transform=transform_test) for name in targets]
val_datas = [DataLoader(dataset, batch_size, num_workers=num_worker) for dataset in val_sets]

net = get_attention_face(train_set.num_classes, 56, embedding_size=embedding_size, weight_norm=True, feature_norm=True)
net.load_parameters("../../models/attention-arc-it-65000.params", ctx=ctx)
# net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
net.hybridize(static_alloc=True)

# loss = L2Softmax(train_set.num_classes, alpha=scale, from_normx=True)
loss = ArcLoss(train_set.num_classes, m=margin, s=scale, easy_margin=False)

logger = logging.getLogger('TRAIN')
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("./attention-face-arcloss.log"))

train_params = net.collect_params()
set_grad_add(train_params)
trainer = gluon.Trainer(train_params, 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
lr_counter = 0

logger.info([margin, scale, lr_steps, lr, momentum, wd, batch_size])

it, epoch = 65000, 11
batch_it = 0

loss_mtc, acc_mtc = mx.metric.Loss(), mx.metric.Accuracy()
tic = time.time()
btic = time.time()

while it < iters + 1:
    if it == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate * 0.1)
        lr_counter += 1

    for batch in tqdm(train_data):

        datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        if it == 5000:
            loss = ArcLoss(train_set.num_classes, m=margin, s=scale, easy_margin=False)
        with ag.record():
            ots = [net(X) for X in datas]
            outputs = [ot[1] for ot in ots]
            losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]

        for l in losses:
            ag.backward(l)
        batch_it += 1
        if batch_it == expect_batch_size / batch_size:
            trainer.step(expect_batch_size)

            acc_mtc.update(labels, outputs)
            loss_mtc.update(0, losses)
            batch_it = 0
            set_grad_zero(train_params)

            if (it % 100) == 0 and it != 0:
                _, train_loss = loss_mtc.get()
                _, train_acc = acc_mtc.get()
                toc = time.time()
                logger.info('\n[epoch % 2d] [it % 3d] train loss: %.6f, train_acc: %.6f | '
                            'speed: %.2f samples/s, time: %.6f' %
                            (epoch, it, train_loss, train_acc, expect_batch_size / (toc - btic), toc - tic))

                loss_mtc.reset()
                acc_mtc.reset()
                if (it % save_period) == 0 and it != 0:
                    results = validate(net, ctx, val_datas, targets)
                    for result in results:
                        logger.info('{}'.format(result))
                    net.save_parameters("../../models/attention-arc-it-%d.params" % it)
                tic = time.time()
            btic = time.time()
            it += 1
    epoch += 1
