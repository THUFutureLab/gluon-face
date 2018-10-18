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
"""Attention Block, implemented in Gluon."""

from mxnet.gluon import nn
from mxnet.gluon.model_zoo.vision.resnet import BottleneckV2

__all__ = ["AttentionBlock", "BottleneckV2"]


class AttentionBlock(nn.HybridBlock):
    def __init__(self, channels, out_size, stage, p=1, t=2, r=1, **kwargs):
        r"""Residual Attention Block from
        `"Residual Attention Network for Image Classification"
        <https://arxiv.org/abs/1704.06904>`_ paper.

        Parameters
        ----------
        :param channels: int. Number of output channels.
        :param out_size: int. Size of the output feature map, now it only supports square shape.
        :param stage: int. Stage described in Figure 2.
        :param p: int. Number of pre-processing Residual Units before split into trunk branch and mask branch.
        :param t: int. Number of Residual Units in trunk branch.
        :param r: int. Number of Residual Units between adjacent pooling layer in the mask branch.
        :param kwargs:
        """
        super().__init__(**kwargs)
        with self.name_scope():
            self.pre = nn.HybridSequential()
            for i in range(p):
                self.pre.add(BottleneckV2(channels, 1, prefix='pre_%d_' % i))

            self.trunk_branch = nn.HybridSequential()
            for i in range(t):
                self.trunk_branch.add(BottleneckV2(channels, 1, prefix='trunk_%d_' % i))

            self.mask_branch = _MaskBlock(channels, r, out_size, stage, prefix='mask_')

            self.post = nn.HybridSequential()
            for i in range(p):
                self.post.add(BottleneckV2(channels, 1, prefix='post_%d_' % i))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.pre(x)
        mask = self.mask_branch(x)
        trunk = self.trunk_branch(x)
        out = (1 + mask) * trunk
        out = self.post(out)
        return out


class _UpSampleBlock(nn.HybridBlock):
    def __init__(self, out_size, **kwargs):
        super().__init__(**kwargs)
        self._size = out_size

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.contrib.BilinearResize2D(x, height=self._size, width=self._size)


class _MaskBlock(nn.HybridBlock):
    def __init__(self, channels, r, out_size, stage, **kwargs):
        super().__init__(**kwargs)
        self._stage = stage
        with self.name_scope():
            self._make_layers(channels, r, stage, out_size)

    def _make_layers(self, channels, r, stage, out_size):
        if stage <= 1:
            self.down_sample_1 = nn.MaxPool2D(3, 2, 1)
            self.down_res_unit_1 = nn.HybridSequential()
            for i in range(r):
                self.down_res_unit_1.add(BottleneckV2(channels, 1, prefix="down_res1_%d_" % i))
            self.skip_connection_1 = BottleneckV2(channels, 1)

            self.up_res_unit_1 = nn.HybridSequential()
            for i in range(r):
                self.up_res_unit_1.add(BottleneckV2(channels, 1, prefix="up_res1_%d_" % i))
            self.up_sample_1 = _UpSampleBlock(out_size)
            out_size = out_size // 2

        if stage <= 2:
            self.down_sample_2 = nn.MaxPool2D(3, 2, 1)
            self.down_res_unit_2 = nn.HybridSequential()
            for i in range(r):
                self.down_res_unit_2.add(BottleneckV2(channels, 1, prefix="down_res2_%d_" % i))
            self.skip_connection_2 = BottleneckV2(channels, 1)

            self.up_res_unit_2 = nn.HybridSequential()
            for i in range(r):
                self.up_res_unit_2.add(BottleneckV2(channels, 1, prefix="up_res2_%d_" % i))
            self.up_sample_2 = _UpSampleBlock(out_size)
            out_size = out_size // 2

        if stage <= 3:
            self.down_sample_3 = nn.MaxPool2D(3, 2, 1)
            self.down_res_unit_3 = nn.HybridSequential()
            for i in range(r):
                self.down_res_unit_3.add(BottleneckV2(channels, 1, prefix="down_res3_%d_" % i))

            self.up_res_unit_3 = nn.HybridSequential()
            for i in range(r):
                self.up_res_unit_3.add(BottleneckV2(channels, 1, prefix="up_res3_%d_" % i))
            self.up_sample_3 = _UpSampleBlock(out_size)

        self.output = nn.HybridSequential()
        self.output.add(nn.BatchNorm(),
                        nn.Activation('relu'),
                        nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False),
                        nn.BatchNorm(),
                        nn.Activation('relu'),
                        nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False),
                        nn.Activation('sigmoid')
                        )

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self._stage <= 1:
            x_down1 = self.down_sample_1(x)
            x_down1 = self.down_res_unit_1(x_down1)
            residual_1 = self.skip_connection_1(x_down1)
        else:
            x_down1 = x
            residual_1 = 0

        if self._stage <= 2:
            x_down2 = self.down_sample_2(x_down1)
            x_down2 = self.down_res_unit_2(x_down2)
            residual_2 = self.skip_connection_2(x_down2)
        else:
            x_down2 = x
            residual_2 = 0

        if self._stage <= 3:
            x_down3 = self.down_sample_3(x_down2)
            x_down3 = self.down_res_unit_3(x_down3)

            x_up3 = self.up_res_unit_3(x_down3)
            x_up3 = self.up_sample_3(x_up3)
        else:
            raise ValueError("param stage should be a number not larger than 3!")

        if self._stage <= 2:
            x_up2 = x_up3 + residual_2
            x_up2 = self.up_res_unit_2(x_up2)
            x_up2 = self.up_sample_2(x_up2)
        else:
            x_up2 = x_up3

        if self._stage <= 1:
            x_up1 = x_up2 + residual_1
            x_up1 = self.up_res_unit_1(x_up1)
            x_up1 = self.up_sample_1(x_up1)
        else:
            x_up1 = x_up2
        out = self.output(x_up1)
        return out
