# @File  : se_resnet.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 18-11-2
"""SE_ResNetV2 is not from origin paper.
It's proposed from  ArcFace paper as SE-LResNet.
`"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
<https://arxiv.org/pdf/1801.07698.pdf>`_ paper.
"""
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.model_zoo.vision.resnet import _conv3x3
from ..nn.basic_blocks import FrBase, SELayer


__all__ = ['get_se_resnet', 'se_resnet18_v2', 'se_resnet34_v2',
           'se_resnet50_v2', 'se_resnet101_v2', 'se_resnet152_v2']


class SE_BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(SE_BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.prelu1 = nn.PReLU()
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.prelu2 = nn.PReLU()
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4)
        self.bn3 = nn.BatchNorm()
        self.prelu3 = nn.PReLU()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

        self.se = SELayer(channels, channels)

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = self.prelu1(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.conv3(x)

        x = self.se(x)
        return x + residual


class SE_ResNetV2(FrBase):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """

    def __init__(self, block, layers, channels, classes=-1, thumbnail=False,
                 embedding_size=512, weight_norm=False, feature_norm=False,
                 need_cls_layer=True, **kwargs):
        super(SE_ResNetV2, self).__init__(classes, embedding_size, weight_norm,
                                          feature_norm, need_cls_layer, **kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.PReLU())
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i + 1],
                                                   stride, i + 1, in_channels=in_channels))
                in_channels = channels[i + 1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.PReLU())
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())

            self.features.add(nn.Dense(embedding_size, use_bias=False))
            self.features.add(nn.BatchNorm(scale=False, center=False))
            self.features.add(nn.PReLU())

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers - 1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer


resnet_spec = {18: (SE_BottleneckV2, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: (SE_BottleneckV2, [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: (SE_BottleneckV2, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: (SE_BottleneckV2, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: (SE_BottleneckV2, [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}


def get_se_resnet(num_layers, **kwargs):
    r"""SE_ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    SE_ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))

    block_type, layers, channels = resnet_spec[num_layers]
    net = SE_ResNetV2(block_type, layers, channels, **kwargs)
    return net


def se_resnet18_v2(**kwargs):
    r"""SE_ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_se_resnet(18, **kwargs)


def se_resnet34_v2(**kwargs):
    r"""SE_ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_se_resnet(34, **kwargs)


def se_resnet50_v2(**kwargs):
    r"""SE_ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_se_resnet(50, **kwargs)


def se_resnet101_v2(**kwargs):
    r"""SE_ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_se_resnet(101, **kwargs)


def se_resnet152_v2(**kwargs):
    r"""SE_ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_se_resnet(152, **kwargs)

