# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)


import mxnet as mx
import functools


def try_gpu(gpu_id=0):
    """Try execute on gpu, if not fallback to cpu"""

    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def test_wapper(*args, **kwargs):
            try:
                a = mx.nd.zeros((1,), ctx=mx.gpu(gpu_id))
                ctx = mx.gpu(gpu_id)
            except Exception:
                ctx = mx.cpu()
            with ctx:
                orig_test(*args, **kwargs)

        return test_wapper

    return test_helper


def try_cpu(cpu_id=0):
    """Try execute on gpu, if not fallback to cpu"""

    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def test_wapper(*args, **kwargs):
            try:
                a = mx.nd.zeros((1,), ctx=mx.cpu(cpu_id))
                ctx = mx.gpu(cpu_id)
            except Exception:
                ctx = mx.cpu(0)
            with ctx:
                orig_test(*args, **kwargs)

        return test_wapper

    return test_helper
