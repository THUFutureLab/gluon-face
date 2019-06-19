# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

import pytest
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from gluonfr.nn.basic_blocks import FrBase
from tests.unittests.common import try_gpu


class Model(FrBase):
    def __init__(self, embedding_size=1, weight_norm=False, feature_norm=False,
                 need_cls_layer=True):
        super(Model, self).__init__(1, embedding_size, weight_norm, feature_norm, need_cls_layer)
        self.features = nn.HybridSequential()
        self.features.add(
            nn.Conv2D(16, 3, 1, 1, use_bias=False),
            nn.BatchNorm(),
            nn.Dense(embedding_size, use_bias=False),
            nn.BatchNorm(center=False, scale=False),
        )


@pytest.fixture(params=[[16, False, False, False],
                        [16, True, True, False],
                        [16, True, True, True]])
def get_model(request):
    return Model(*request.param)


@try_gpu(0)
def test_model(get_model):
    ctx = mx.context.current_context()
    x = nd.random.normal(shape=(2, 3, 112, 112), ctx=ctx)
    model = get_model
    model.initialize(ctx=ctx)
    model(x)
