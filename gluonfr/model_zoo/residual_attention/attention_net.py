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
"""Residual Attention network, implemented in Gluon."""

from mxnet.gluon import nn
from .attention_block import BottleneckV2, AttentionBlock

__all__ = ["AttentionNet", "AttentionNetFace",
           "get_attention_net", "get_attention_face",
           "attention_net56", "attention_net92", "attention_net128",
           "attention_net164", "attention_net236", "attention_net452"
           ]


class AttentionNet(nn.HybridBlock):
    r"""AttentionNet Model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/abs/1704.06904>`_ paper.

    Parameters
    ----------
    :param classes: int. Number of classification classes.
    :param modules: list. The number of Attention Module in each stage.
    :param p: int. Number of pre-processing Residual Units before split into trunk branch and mask branch.
    :param t: int. Number of Residual Units in trunk branch.
    :param r: int. Number of Residual Units between adjacent pooling layer in the mask branch.
    :param kwargs:

    """

    def __init__(self, classes, modules, p, t, r, **kwargs):
        super().__init__(**kwargs)
        assert len(modules) == 3
        with self.name_scope():
            self.features = nn.HybridSequential()
            # 112x112
            self.features.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            # 56x56
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.features.add(BottleneckV2(256, 1, True, 64))
            for _ in range(modules[0]):
                self.features.add(AttentionBlock(256, 56, 1, p, t, r))

            # 28x28
            self.features.add(BottleneckV2(512, 2, True, 256))
            for _ in range(modules[1]):
                self.features.add(AttentionBlock(512, 28, 2, p, t, r))

            # 14x14
            self.features.add(BottleneckV2(1024, 2, True, 512))
            for _ in range(modules[2]):
                self.features.add(AttentionBlock(1024, 14, 3, p, t, r))

            # 7x7
            self.features.add(BottleneckV2(2048, 2, True, 1024),
                              BottleneckV2(2048, 1),
                              BottleneckV2(2048, 1))

            # 2048
            self.features.add(nn.BatchNorm(),
                              nn.Activation('relu'),
                              nn.GlobalAvgPool2D(),
                              nn.Flatten())

            # classes
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class AttentionNetFace(nn.HybridBlock):
    r"""
    AttentionNet Model for input 112x112.

    Parameters
    ----------
    :param classes: int. Number of classification classes.
    :param kwargs:

    """

    def __init__(self, classes, modules, p, t, r, **kwargs):
        super().__init__(**kwargs)
        assert len(modules) == 3
        with self.name_scope():
            self.features = nn.HybridSequential()
            # 112x112
            self.features.add(nn.Conv2D(64, 3, 1, 1, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            # 56x56
            self.features.add(BottleneckV2(256, 2, True, 64))
            for _ in range(modules[0]):
                self.features.add(AttentionBlock(256, 56, 1, p, t, r))

            # 28x28
            self.features.add(BottleneckV2(512, 2, True, 256))
            for _ in range(modules[1]):
                self.features.add(AttentionBlock(512, 28, 2, p, t, r))

            # 14x14
            self.features.add(BottleneckV2(1024, 2, True, 512))
            for _ in range(modules[2]):
                self.features.add(AttentionBlock(1024, 14, 3, p, t, r))

            # 8x8
            self.features.add(BottleneckV2(2048, 2, True, 1024),
                              BottleneckV2(2048, 1),
                              BottleneckV2(2048, 1))

            # 2048
            self.features.add(nn.BatchNorm(),
                              nn.Activation('relu'),
                              nn.GlobalAvgPool2D(),
                              nn.Flatten())

            # classes
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification ([p, t, r], [stage1, stage2, stage3])
# The hyper-parameters are based on paper section 4.1. The number of layers can be calculated by 36m+20
# where m is the number of Attention Module in each stage when `p, t, r = 1, 2, 1`.
attention_net_spec = {56: ([1, 2, 1], [1, 1, 1]),
                      92: ([1, 2, 1], [1, 2, 3]),
                      128: ([1, 2, 1], [3, 3, 3]),
                      164: ([1, 2, 1], [4, 4, 4]),
                      236: ([1, 2, 1], [6, 6, 6]),
                      452: ([2, 4, 3], [6, 6, 6])}


# Constructor
def get_attention_net(classes, num_layers, **kwargs):
    r"""AttentionNet Model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/abs/1704.06904>`_ paper.

    Parameters
    ----------
    classes : int,
        Number of classification classes.
    num_layers : int
        Numbers of layers. Options are 56, 92, 128, 164, 236, 452.
    """
    assert num_layers in attention_net_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(attention_net_spec.keys()))
    ptr, modules = attention_net_spec[num_layers]
    assert len(ptr) == len(modules) == 3
    p, t, r = ptr
    net = AttentionNet(classes, modules, p, t, r, **kwargs)
    return net


def get_attention_face(classes, num_layers, **kwargs):
    r"""AttentionNet Model for 112x112 face images from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/abs/1704.06904>`_ paper.

    Parameters
    ----------
    classes : int,
        Number of classification classes.
    num_layers : int
        Numbers of layers. Options are 56, 92, 128, 164, 236, 452.
    """
    assert num_layers in attention_net_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(attention_net_spec.keys()))
    ptr, modules = attention_net_spec[num_layers]
    assert len(ptr) == len(modules) == 3
    p, t, r = ptr
    net = AttentionNetFace(classes, modules, p, t, r, **kwargs)
    return net


def attention_net56(classes, **kwargs):
    r"""AttentionNet 56 Model from
       `"Residual Attention Network for Image Classification"
       <https://arxiv.org/abs/1704.06904>`_ paper.

       Parameters
       ----------
       classes : int,
           Number of classification classes.
       """
    return get_attention_net(classes, 56, **kwargs)


def attention_net92(classes, **kwargs):
    r"""AttentionNet 92 Model from
       `"Residual Attention Network for Image Classification"
       <https://arxiv.org/abs/1704.06904>`_ paper.

       Parameters
       ----------
       classes : int,
           Number of classification classes.
       """
    return get_attention_net(classes, 92, **kwargs)


def attention_net128(classes, **kwargs):
    r"""AttentionNet 128 Model from
       `"Residual Attention Network for Image Classification"
       <https://arxiv.org/abs/1704.06904>`_ paper.

       Parameters
       ----------
       classes : int,
           Number of classification classes.
       """
    return get_attention_net(classes, 128, **kwargs)


def attention_net164(classes, **kwargs):
    r"""AttentionNet 164 Model from
       `"Residual Attention Network for Image Classification"
       <https://arxiv.org/abs/1704.06904>`_ paper.

       Parameters
       ----------
       classes : int,
           Number of classification classes.
       """
    return get_attention_net(classes, 164, **kwargs)


def attention_net236(classes, **kwargs):
    r"""AttentionNet 236 Model from
       `"Residual Attention Network for Image Classification"
       <https://arxiv.org/abs/1704.06904>`_ paper.

       Parameters
       ----------
       classes : int,
           Number of classification classes.
       """
    return get_attention_net(classes, 236, **kwargs)


def attention_net452(classes, **kwargs):
    r"""AttentionNet 452 Model from
       `"Residual Attention Network for Image Classification"
       <https://arxiv.org/abs/1704.06904>`_ paper.

       Parameters
       ----------
       classes : int,
           Number of classification classes.
       """
    return get_attention_net(classes, 452, **kwargs)
