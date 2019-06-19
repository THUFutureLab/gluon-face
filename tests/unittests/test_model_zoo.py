# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

import mxnet as mx
from gluonfr.model_zoo.model_zoo import *
from tests.unittests.common import try_gpu


@try_gpu(0)
def test_model_zoo():
    ctx = mx.context.current_context()
    models = get_model_list()
    data = mx.random.normal(shape=(2, 3, 112, 112), ctx=ctx)
    for model_name in models:
        model = get_model(model_name, classes=10, weight_norm=True, feature_norm=True)
        model.initialize(ctx=ctx)
        model(data)
        mx.nd.waitall()
