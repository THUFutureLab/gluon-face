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
"""Basic Blocks used in GluonFR."""

__all__ = ['NormDense', 'SELayer']

from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock


class NormDense(HybridBlock):
    def __init__(self, classes, weight_norm=False, feature_norm=False,
                 dtype='float32', weight_initializer=None, in_units=0, **kwargs):
        super().__init__(**kwargs)
        self._weight_norm = weight_norm
        self._feature_norm = feature_norm

        self._classes = classes
        self._in_units = in_units
        if weight_norm:
            assert in_units > 0, "Weight shape cannot be inferred auto when use weight norm, " \
                                 "in_units should be given."
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(classes, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x, weight, *args, **kwargs):
        if self._weight_norm:
            weight = F.L2Normalization(weight, mode='instance')
        if self._feature_norm:
            x = F.L2Normalization(x, mode='instance', name='fc1n')
        return F.FullyConnected(data=x, weight=weight, no_bias=True,
                                num_hidden=self._classes, name='fc7')

    def __repr__(self):
        s = '{name}({layout})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


class SELayer(HybridBlock):
    def __init__(self, channel, in_channel, reduction=16, **kwargs):
        super(SELayer, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.fc = nn.HybridSequential()
            with self.fc.name_scope():
                self.fc.add(nn.Conv2D(channel // reduction, kernel_size=1, in_channels=in_channel))
                self.fc.add(nn.PReLU())
                self.fc.add(nn.Conv2D(channel, kernel_size=1, in_channels=channel // reduction))
                self.fc.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self.avg_pool(x)
        y = self.fc(y)
        return F.broadcast_mul(x, y)
