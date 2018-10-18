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
import mxnet as mx
import numpy as np
from unittest import TestCase
from gluonfr.data.dataset import FRValDataset
from mxnet.gluon.data import DataLoader

targets = 'lfw,calfw,cplfw,cfp_fp,agedb_30,cfp_ff,vgg2_fp'


class TestDataset(TestCase):
    # def setUpClass(cls):

    def test_val_dataset(self):
        for target in targets.split(","):
            val_set = FRValDataset(target)
            for _ in range(10):
                index = np.random.randint(0, len(val_set))
                _ = val_set[index]

    def test_load_data(self):
        for target in targets.split(","):
            loader = DataLoader(FRValDataset(target), batch_size=8)
            for i, batch in enumerate(loader):
                data = batch[0]
                issame = batch[1]
                print(data[0].shape)
                print(issame)
                # assert isinstance(data, (mx.nd.NDArray, mx.nd.NDArray))
                # assert isinstance(issame, mx.nd.NDArray)
                # assert data[0].shape == data[1].shape == (8, 3, 112, 112)
                # assert issame.shape == (8,)
                if i > 0:
                    break
