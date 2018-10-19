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
"""Face Recognition Dataset"""
import os
import pickle
import mxnet as mx
import numpy as np
from mxnet.gluon.data import Dataset
from mxnet.gluon.data import RecordFileDataset
from mxnet import image, recordio

__all__ = ["FRValDataset",
           "FRTrainRecordDataset",
           ]


class FRTrainRecordDataset(RecordFileDataset):
    """A dataset wrapping over a rec serialized file provided by InsightFace Repo.

    Parameters
    ----------
    name : str. Name of val dataset.
    root : str. Path to face folder. Default is '$(HOME)/mxnet/datasets/face'
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::
        transform=lambda data, label: (data.astype(np.float32)/255, label)
    """

    def __init__(self, name, root=os.path.expanduser('~/.mxnet/datasets/face'), flag=1, transform=None):
        super().__init__(os.path.join(root, name, "train.rec"))
        prop = open(os.path.join(root, name, "property"), "r").read().strip().split(',')
        self._flag = flag
        self._transform = transform

        assert len(prop) == 3
        self.num_classes = int(prop[0])
        self.image_size = [int(prop[1]), int(prop[2])]

    def __getitem__(self, idx):
        while True:
            record = super().__getitem__(idx)
            header, img = recordio.unpack(record)
            if _check_valid_image(img):
                decoded_img = image.imdecode(img, self._flag)
            else:
                idx = np.random.randint(low=0, high=len(self))
                continue
            if self._transform is not None:
                return self._transform(decoded_img, header.label)
            return decoded_img, header.label


def _check_valid_image(s):
    return False if len(s) == 0 else True


class FRValDataset(Dataset):
    """A dataset wrapping over a pickle serialized (.bin) file provided by InsightFace Repo.

    Parameters
    ----------
    name : str. Name of val dataset.
    root : str. Path to face folder. Default is '$(HOME)/mxnet/datasets/face'
    transform : callable, default None
        A function that takes data and transforms them:
    ::
        transform = lambda data: data.astype(np.float32)/255

    """

    def __init__(self, name, root=os.path.expanduser('~/.mxnet/datasets/face'), transform=None):
        super().__init__()
        self._transform = transform
        self.name = name
        with open(os.path.join(root, "{}.bin".format(name)), 'rb') as f:
            self.bins, self.issame_list = pickle.load(f, encoding='iso-8859-1')

        self._do_encode = not isinstance(self.bins[0], np.ndarray)

    def __getitem__(self, idx):
        img0 = self._decode(self.bins[2 * idx])
        img1 = self._decode(self.bins[2 * idx + 1])

        issame = 1 if self.issame_list[idx] else 0

        if self._transform is not None:
            img0 = self._transform(img0)
            img1 = self._transform(img1)

        return (img0, img1), issame

    def __len__(self):
        return len(self.issame_list)

    def _decode(self, im):
        if self._do_encode:
            im = im.encode("iso-8859-1")
        return mx.image.imdecode(im)
