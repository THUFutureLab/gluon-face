# @File  : model_zoo.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 19-2-19

from .mobile_facenet import *
from .attention_net import *
from .se_resnet import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'l_se_resnet18v2': se_resnet18_v2,
    'l_se_resnet50v2': se_resnet50_v2,
    'l_se_resnet101v2': se_resnet101_v2,
    'mobilefacenet': get_mobile_facenet,
    'attentionfacenet56': attention_net56,
    'attentionfacenet92': attention_net92,
}


def get_model(name, **kwargs):
    """Returns a model by name
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
