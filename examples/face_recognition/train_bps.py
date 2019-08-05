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
sys.path.append(os.path.dirname(__file__))

import time
import mxnet as mx
import byteps.mxnet as bps
from mxnet import autograd as ag
from gluoncv.utils import LRScheduler
from nvidia.dali.plugin.mxnet import DALIClassificationIterator

from gluonfr.loss import *
from gluonfr.model_zoo import *
from utils import Logger, FacePipe, ParallelValidation

# init BytePS
bps.init()
num_gpu = bps.size()
local_rank = bps.local_rank()
rank = bps.rank()


# hyper parameters
lamda = 0.01
r_init = 20.0
embedding_size = 128

lr = 0.1
momentum = 0.9
wd = 4e-5

ctx = mx.gpu(local_rank)
num_worker = 4
batch_size_per_gpu = 128
batch_size = batch_size_per_gpu * num_gpu

val_targets = ['lfw']
epochs = 60
save_period = 500

# setting logger
if rank == 0:
    if not os.path.exists(os.path.join("./log")):
        os.makedirs(os.path.join("./log"))
    if not os.path.exists(os.path.join("./models")):
        os.makedirs(os.path.join("./models"))

logger = Logger(root="./log", prefix="mobile_facenet", local_rank=local_rank)

# train and val pipeline
train_pipes = [FacePipe(name="emore", batch_size=batch_size_per_gpu, num_threads=num_worker,
                        device_id=local_rank, num_shards=num_gpu, shard_id=rank)]
train_size = train_pipes[0].size
num_classes = train_pipes[0].num_classes
train_iter = DALIClassificationIterator(train_pipes, train_size // num_gpu, auto_reset=True)

validator = ParallelValidation(val_targets, batch_size_per_gpu, rank, local_rank, logger=logger)


# loss, network
net = get_mobile_facenet(num_classes, weight_norm=True)
net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
net.hybridize(static_alloc=True)

loss = RingLoss(lamda, r_init)
loss.initialize(ctx=ctx)
loss.hybridize(static_alloc=True)

# trainer
num_batches = train_size // batch_size
train_params = net.collect_params()
train_params.update(loss.params)
lr_scheduler = LRScheduler(mode="cosine", base_lr=lr, target_lr=1e-8,
                           iters_per_epoch=num_batches, nepochs=epochs)
trainer = bps.DistributedTrainer(train_params, 'sgd',
                                 {'momentum': momentum, 'wd': wd, "lr_scheduler": lr_scheduler,
                                  "multi_precision": True, "learning_rate": lr})

# metrics
loss_mtc, acc_mtc = mx.metric.Loss(), mx.metric.Accuracy()
tic = time.time()
btic = time.time()

# train loop
for epoch in range(epochs):
    for i, batch in enumerate(train_iter):
        it = epoch * num_batches + i
        data = batch[0].data[0]
        label = batch[0].label[0]

        with ag.record():
            embedding, output = net(data)
            batch_loss = loss(output, label, embedding)

        ag.backward(batch_loss)
        trainer.step(batch_size)

        acc_mtc.update([label], [output])
        loss_mtc.update(0, [batch_loss])

        if (it % save_period) == 0 and it != 0:
            _, train_loss = loss_mtc.get()
            _, train_acc = acc_mtc.get()
            toc = time.time()
            logger.info('\n[epoch % 2d] [it % 3d] train loss: %.6f, train_acc: %.6f | '
                        'learning rate: %.8f speed: %.2f samples/s, time: %.6f' %
                        (epoch, it, train_loss, train_acc, trainer.learning_rate,
                         batch_size / (toc - btic), toc - tic), rank)
            logger.info("Radius {}".format(loss.R.data(ctx=mx.gpu(local_rank)).asscalar()), rank)
            validator(net)
            if rank == 0:
                net.save_parameters("./models/mobilefacenet-ring-it-%d.params" % it)
            loss_mtc.reset()
            acc_mtc.reset()
            tic = time.time()
        btic = time.time()
