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
from mxnet.gluon import nn
from ..nn.basic_blocks import NormDense

__all__ = ["MobileFaceNet",
           "get_mobile_facenet",
           ]


def _make_conv(stage_index, channels=1, kernel=1, stride=1, pad=0,
               num_group=1, active=True):
    out = nn.HybridSequential(prefix='stage%d_' % stage_index)
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(nn.PReLU())
    return out


def _make_bottleneck(stage_index, layers, channels, stride, t, in_channels=0, use_se=False):
    layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
    with layer.name_scope():
        layer.add(Bottleneck(in_channels=in_channels, channels=channels, t=t, stride=stride, use_se=use_se))
        for _ in range(layers - 1):
            layer.add(Bottleneck(channels, channels, t, 1, use_se=use_se))
    return layer


class Bottleneck(nn.HybridBlock):

    def __init__(self, in_channels, channels, t, stride, use_se=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()
            self.out.add(_make_conv(0, in_channels * t),
                         _make_conv(1, in_channels * t, kernel=3, stride=stride,
                                    pad=1, num_group=in_channels * t),
                         _make_conv(2, channels, active=False))
            if use_se:
                self.out.add(SELayer(channels, channels))

    def hybrid_forward(self, F, x, **kwargs):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class MobileFaceNet(nn.HybridBlock):
    def __init__(self, classes, weight_norm=False, feature_norm=False,
                 use_se=False, embedding_size=128, **kwargs):
        super(MobileFaceNet, self).__init__(**kwargs)
        with self.name_scope():
            self.feature = nn.HybridSequential(prefix='feature_')
            with self.feature.name_scope():
                self.feature.add(_make_conv(0, 64, kernel=3, stride=2, pad=1),
                                 _make_conv(0, 64, kernel=3, stride=1, pad=1, num_group=64))

                self.feature.add(
                    _make_bottleneck(1, layers=5, channels=64, stride=2, t=2, in_channels=64, use_se=use_se),
                    _make_bottleneck(2, layers=1, channels=128, stride=2, t=4, in_channels=64, use_se=use_se),
                    _make_bottleneck(3, layers=6, channels=128, stride=1, t=2, in_channels=128, use_se=use_se),
                    _make_bottleneck(4, layers=1, channels=128, stride=2, t=4, in_channels=128, use_se=use_se),
                    _make_bottleneck(5, layers=2, channels=128, stride=1, t=2, in_channels=128, use_se=use_se))

                self.feature.add(_make_conv(6, 512),
                                 _make_conv(6, 512, kernel=7, num_group=512, active=False),
                                 nn.Conv2D(embedding_size, 1, use_bias=False),
                                 nn.BatchNorm(scale=False, center=False),
                                 nn.PReLU(),
                                 nn.Flatten())

            self.output = NormDense(classes, weight_norm=weight_norm, feature_norm=feature_norm,
                                    in_units=embedding_size, prefix='output_')

    def hybrid_forward(self, F, x, **kwargs):
        embedding = self.feature(x)
        out = self.output(embedding)
        return embedding, out


def get_mobile_facenet(classes, **kwargs):
    return MobileFaceNet(classes, **kwargs)
