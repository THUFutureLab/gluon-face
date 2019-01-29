# MIT License
#
# Copyright (c) 2019 Haoxintong
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

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import time
import logging
import mxnet as mx
from tqdm import tqdm
from mxnet import gluon, autograd as ag
from mxnet.gluon.data import DataLoader
from datetime import datetime
from gluoncv.utils import LRScheduler
from gluonfr.loss import *
from gluonfr.model_zoo import *
from gluonfr.data import get_recognition_dataset
from examples.face_recognition.utils import Transform, validate


num_gpu = 4
ctx = [mx.gpu(i) for i in range(num_gpu)]
batch_size = 128 * num_gpu
num_worker = 36

epochs = 15
save_period = 3000
warmup_epochs = 1


scale = 60
margin = 0.5

lr = 0.1
momentum = 0.9
wd = 4e-5

use_float16 = False
trans = Transform(use_float16)

train_set = get_recognition_dataset("emore", transform=trans.transform_train)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_worker, last_batch="discard")

targets = ['lfw']
val_sets = [get_recognition_dataset(name, transform=trans.transform_test) for name in targets]
val_datas = [DataLoader(dataset, batch_size, num_workers=num_worker) for dataset in val_sets]


net = get_mobile_facenet(train_set.num_classes, weight_norm=True, feature_norm=True)
if use_float16:
    net.cast("float16")
net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
net.hybridize(static_alloc=True)

loss = ArcLoss(train_set.num_classes, m=margin, s=scale, easy_margin=False,
               dtype="float16" if use_float16 else "float32")

if not os.path.exists(os.path.join("./log")):
    os.makedirs(os.path.join("./log"))
if not os.path.exists(os.path.join("./models")):
    os.makedirs(os.path.join("./models"))

logger = logging.getLogger('TRAIN')
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("./log/mob-arcloss%s.log" % datetime.strftime(datetime.now(), '%m%d_%H')))

num_batches = len(train_set) // batch_size
train_params = net.collect_params()
train_params.update(loss.params)
lr_scheduler = LRScheduler(mode="cosine", baselr=lr, targetlr=1e-6,
                           niters=num_batches, nepochs=epochs,
                           warmup_lr=lr / 10, warmup_epochs=warmup_epochs, warmup_mode='linear'
                           )
trainer = gluon.Trainer(train_params, 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd,
                                              "lr_scheduler": lr_scheduler, "multi_precision": True})
lr_counter = 0

logger.info([margin, scale, lr, batch_size])
logger.info("Batches per epoch: {}, Classes: {}".format(num_batches, train_set.num_classes))

loss_mtc, acc_mtc = mx.metric.Loss(), mx.metric.Accuracy()
tic = time.time()
btic = time.time()

for epoch in range(epochs):

    for i, batch in enumerate(tqdm(train_data)):
        it = epoch * num_batches + i
        lr_scheduler.update(i, epoch)

        datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        if use_float16:
            labels = [label.astype("float16") for label in labels]

        with ag.record():
            ots = [net(X) for X in datas]
            outputs = [ot[1] for ot in ots]
            losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]

        for l in losses:
            ag.backward(l)

        trainer.step(batch_size)
        acc_mtc.update(labels, outputs)
        loss_mtc.update(0, losses)

        if (it % 200) == 0 and it != 0:
            _, train_loss = loss_mtc.get()
            _, train_acc = acc_mtc.get()
            toc = time.time()

            logger.info('\n[epoch % 2d] [it % 3d] train loss: %.6f, train_acc: %.6f | '
                        ' lr: %.8f, speed: %.2f samples/s, time: %.6f' %
                        (epoch, it, train_loss, train_acc, lr_scheduler.learning_rate,
                         batch_size / (toc - btic), toc - tic))
            loss_mtc.reset()
            acc_mtc.reset()
            tic = time.time()
            if (it % save_period) == 0 or it == 200:
                results = validate(net, ctx, val_datas, targets)
                for result in results:
                    logger.info('{}'.format(result))

                net.save_parameters("./models/mob-arc-fp32-it-%d.params" % it)
        btic = time.time()
