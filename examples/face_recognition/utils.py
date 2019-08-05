# @File  : utils.py
# @Author: X.Yang&Xt.Hao
# @Contact : pistonyang@gmail.com, haoxintongpku@gmail.com
# @Date  : 18-11-1
import os
import time
import logging
from datetime import datetime

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluonfr.data import *
from gluonfr.metrics.verification import FaceVerification

from mxboard import SummaryWriter

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

__all__ = ["transform_test", "FacePipe", "Transform", "validate", "Logger", "ParallelValidation"]
transform_test = transforms.Compose([
    transforms.ToTensor()
])


class FacePipe(Pipeline):
    """A DALI pipeline for face recordio files.

    """

    def __init__(self, name, batch_size, num_threads, device_id, num_shards, shard_id,
                 root=os.path.expanduser('~/.mxnet/datasets/face'), ):
        super().__init__(batch_size, num_threads, device_id, seed=12)

        idx_files = [os.path.join(root, name, "train.idx")]
        rec_files = [os.path.join(root, name, "train.rec")]
        prop = open(os.path.join(root, name, "property"), "r").read().strip().split(',')
        assert len(prop) == 3
        self.num_classes = int(prop[0])
        self.image_size = [int(prop[1]), int(prop[2])]
        self.size = 0
        for idx_file in idx_files:
            with open(idx_file, "r") as f:
                self.size += len(list(f.readlines()))

        self._input = ops.MXNetReader(path=rec_files, index_path=idx_files, random_shuffle=True,
                                      num_shards=num_shards, shard_id=shard_id, seed=12,
                                      tensor_init_bytes=self.image_size[0] * self.image_size[1] * 8)
        self._decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self._cmnp = ops.CropMirrorNormalize(device="gpu",
                                             output_dtype=types.FLOAT,
                                             output_layout=types.NCHW,
                                             crop=self.image_size,
                                             image_type=types.RGB,
                                             mean=[0., 0., 0.],
                                             std=[255., 255., 255.])
        self._contrast = ops.Contrast(device="gpu")
        self._saturation = ops.Saturation(device="gpu")
        self._brightness = ops.Brightness(device="gpu")

        self._uniform = ops.Uniform(range=(0.7, 1.3))
        self._coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        inputs, labels = self._input(name="Reader")
        images = self._decode(inputs)

        images = self._contrast(images, contrast=self._uniform())
        images = self._saturation(images, saturation=self._uniform())
        images = self._brightness(images, brightness=self._uniform())

        output = self._cmnp(images, mirror=self._coin())
        return output, labels.gpu()


class Transform:
    def __init__(self, use_float16=False):
        self._transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        self._transform_train = transforms.Compose([
            transforms.RandomBrightness(0.3),
            transforms.RandomContrast(0.3),
            transforms.RandomSaturation(0.3),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor()
        ])
        self.use_float16 = use_float16

    def transform_train(self, data, label):
        im = self._transform_train(data)
        if self.use_float16:
            im = im.astype('float16')
        return im, label

    def transform_test(self, data):
        im = self._transform_test(data)
        if self.use_float16:
            im = im.astype('float16')
        return im


class Logger(SummaryWriter):
    r"""A wrapper over Summary Writer, add logger providing text log file.
    The output will start with prefix, end with datetime month-day-hour.


    Parameters
    ----------
    root : str.
        Directory where event file and log file will be written.
    prefix: str
        File's name start with.
    max_queue : int.
        Size of the queue for pending events and summaries.
    flush_secs: Number.
        How often, in seconds, to flush the pending events and summaries to disk.
    filename_suffix : str.
        Every event file's name is suffixed with `filename_suffix` if provided.
    sw_verbose : bool.
        Determines whether to print the logging messages.
    """

    def __init__(self, root, prefix, sw_verbose=False, local_rank=0, root_rank=0, **kwargs):
        if not os.path.exists(root):
            raise FileExistsError("Root not exist: ", root)
        # to avoid create file in the same time.
        time.sleep(0.5 * local_rank)

        self._prefix = os.path.join(root, prefix + "_{}".format(datetime.strftime(datetime.now(), '%m%d%H')))

        super().__init__(logdir=self._prefix, verbose=sw_verbose, **kwargs)
        self._logger = self._init_text_logger()
        self._root_rank = root_rank

    def _init_text_logger(self):
        logger = logging.getLogger('TRAIN')
        logger.setLevel("INFO")
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler("{}.log".format(self._prefix)))
        return logger

    def info(self, text, rank=0):
        if rank == self._root_rank:
            self._logger.info(text)


class ParallelValidation:
    def __init__(self, targets, batch_size, rank, local_rank=0, num_workers=1, logger=None, nfolds=10, norm=True):
        self.metric = FaceVerification(nfolds)
        self.norm = norm
        self.targets = targets
        self.ctx = mx.gpu(local_rank)
        self.logger = logger
        if rank >= len(targets):
            self.skip_validate = True
        else:
            self.skip_validate = False
            # In each process we only do validation for one val set.
            self.name = targets[rank]
            val_set = get_recognition_dataset(self.name, transform=transform_test)

            self.loader = gluon.data.DataLoader(val_set, batch_size, num_workers=num_workers)

    def __call__(self, net, *args, **kwargs):
        if not self.skip_validate:
            self.metric.reset()
            for batch in self.loader:
                data0 = batch[0][0].as_in_context(self.ctx)
                data1 = batch[0][1].as_in_context(self.ctx)
                issame = batch[1].as_in_context(self.ctx)

                embedding0 = net(data0)[0]
                embedding1 = net(data1)[0]
                if self.norm:
                    embedding0 = nd.L2Normalization(embedding0)
                    embedding1 = nd.L2Normalization(embedding1)

                self.metric.update(issame, embedding0, embedding1)

            tpr, fpr, accuracy, val, val_std, far, accuracy_std = self.metric.get()
            text = "{}: {:.6f}+-{:.6f}".format(self.name, accuracy, accuracy_std)
            if self.logger is None:
                print(text)
            else:
                self.logger.info(text)


def validate(net, ctx, val_datas, targets, nfolds=10, norm=True):
    metric = FaceVerification(nfolds)
    results = []
    for loader, name in zip(val_datas, targets):
        metric.reset()
        for i, batch in enumerate(loader):
            data0s = gluon.utils.split_and_load(batch[0][0], ctx, even_split=False)
            data1s = gluon.utils.split_and_load(batch[0][1], ctx, even_split=False)
            issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

            embedding0s = [net(X)[0] for X in data0s]
            embedding1s = [net(X)[0] for X in data1s]
            if norm:
                embedding0s = [nd.L2Normalization(e) for e in embedding0s]
                embedding1s = [nd.L2Normalization(e) for e in embedding1s]

            for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
                metric.update(issame, embedding0, embedding1)

        tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
        results.append("{}: {:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))
    return results
