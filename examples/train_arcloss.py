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
from gluonfr.model_zoo.residual_attention import get_attention_face
import numpy as np

from mxnet.gluon.data.vision import transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_train = transforms.Compose([
    transforms.RandomBrightness(0.3),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


class AttentionFace(nn.HybridBlock):
    def __init__(self, classes, is_test=False, s=30, **kwargs):
        super().__init__(**kwargs)
        self.feature = get_attention_face(512, 56)
        if is_test:
            self.output = None
        else:
            self.output = NormDense(classes=classes, s=s)

        self._test = is_test

    def hybrid_forward(self, F, x, *args, **kwargs):
        embedding = self.feature(x)
        if self.output is None:
            return embedding
        else:
            return self.output(embedding)


