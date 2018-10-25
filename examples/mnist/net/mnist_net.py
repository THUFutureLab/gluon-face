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
"""MNIST Network"""
from mxnet.gluon import nn
from gluonfr.nn.basic_blocks import NormDense


class MnistNet(nn.HybridBlock):
    def __init__(self, weight_norm=True, feature_norm=True, embedding_size=2, s=1, **kwargs):
        super().__init__(**kwargs)
        self.feature = nn.HybridSequential()
        self.feature.add(
            nn.Conv2D(32, 5, activation="relu"),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(64, 5, activation="relu"),
            nn.MaxPool2D(2, 2),
            nn.Flatten(),
            nn.Dense(128, activation="relu"),
            nn.Dense(256, activation="relu"),
            nn.Dense(embedding_size, use_bias=False)
        )
        self.output = NormDense(classes=10, s=s, weight_norm=weight_norm,
                                feature_norm=feature_norm, in_units=embedding_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        embedding = self.feature(x)
        output = self.output(embedding)
        return embedding, output
