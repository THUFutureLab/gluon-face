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
"""Use Nvidia Data Loading Library as data provider, this is useful when cpu
cores cannot match gpu nums using Gluon Dataloader."""
import os
import mxnet as mx
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import DALIClassificationIterator

__all__ = ["split_and_load", "DaliDataset", "DALIClassificationIterator"]


def split_and_load(batch_data, num_gpus):
    """Splits Batch from Dali Iterator into `len(ctx_list)` slices and loads
    each slice to one context `mx.gpu(i)`.

    Parameters
    ----------
    batch_data : NDArray
        A batch of data.
    num_gpus : int
        Num of Gpu for computing.

    Returns
    -------
    list of NDArray, list of Labels
        Each corresponds to a context in `ctx_list`.
    """
    return [batch_data[i].data[0] for i in range(num_gpus)], \
           [batch_data[i].label[0].as_in_context(mx.gpu(i)) for i in range(num_gpus)]


class DaliDataset(Pipeline):
    def __init__(self, name, batch_size, num_workers, device_id, num_gpu,
                 root=os.path.expanduser('~/.mxnet/datasets/face')):
        super().__init__(batch_size, num_workers, device_id, seed=12 + device_id)

        idx_files = [os.path.join(root, name, "train.idx")]
        rec_files = [os.path.join(root, name, "train.rec")]
        prop = open(os.path.join(root, name, "property"), "r").read().strip().split(',')
        assert len(prop) == 3
        self.num_classes = int(prop[0])
        self.image_size = [int(prop[1]), int(prop[2])]

        self._input = ops.MXNetReader(path=rec_files, index_path=idx_files, random_shuffle=True,
                                      num_shards=num_gpu, tensor_init_bytes=self.image_size[0] * self.image_size[1] * 8)
        self._decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self._cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW,
                                             crop=self.image_size, image_type=types.RGB,
                                             mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        self._contrast = ops.Contrast(device="gpu", )
        self._saturation = ops.Saturation(device="gpu", )
        self._brightness = ops.Brightness(device="gpu", )

        self._uniform = ops.Uniform(range=(0.7, 1.3))
        self._coin = ops.CoinFlip(probability=0.5)
        self.iter = 0

    def define_graph(self):
        inputs, labels = self._input(name="Reader")
        images = self._decode(inputs)

        images = self._contrast(images, contrast=self._uniform())
        images = self._saturation(images, saturation=self._uniform())
        images = self._brightness(images, brightness=self._uniform())

        output = self._cmnp(images, mirror=self._coin())
        return output, labels

    def iter_setup(self):
        pass
