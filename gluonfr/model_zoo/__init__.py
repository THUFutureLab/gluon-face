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
""" Models for face recognition"""
from .mobile_facenet import *
from .attention_net import *
from .se_resnet import *


_models = {
    'l_se_resnet18v2': se_resnet18_v2,
    'l_se_resnet50v2': se_resnet50_v2,
    'l_se_resnet101v2': se_resnet101_v2,
    'mobilefacenet': get_mobile_facenet,
    'attention_facenet': get_attention_face,
    'attentionfacenet56': attention_net56,
    'attentionfacenet92': attention_net92,
}


def get_model(name, **kwargs):
    """Returns a model by name.

    Parameters
    ----------
    name : str
        Name of the model.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.


    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return _models.keys()

