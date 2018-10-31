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
"""GluonFR: a deep learning face recognition toolkit powered by Gluon."""

# mxnet version check
mx_version = '1.3.0'
try:
    import mxnet as mx
    from distutils.version import LooseVersion
    if LooseVersion(mx.__version__) < LooseVersion(mx_version):
        msg = (
            "Legacy mxnet=={} detected, some new modules may not work properly. "
            "mxnet>={} is required. You can use pip to upgrade mxnet "
            "`pip install mxnet/mxnet-cu92 --pre --upgrade`").format(mx.__version__, mx_version)
        raise ImportError(msg)
except ImportError:
    raise ImportError(
        "Unable to import dependency mxnet. "
        "A quick tip is to install via `pip install mxnet/mxnet-cu92 --pre`. "
        )

__version__ = '0.1.0'

from . import data
# from . import model_zoo
from . import nn
from . import utils
from . import loss
