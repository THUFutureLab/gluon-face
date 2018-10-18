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
"""Custom losses"""
import math
import numpy as np
from mxnet import nd
from mxnet.gluon.loss import Loss

__all__ = ["ArcLoss",  "TripletLoss"]
numeric_types = (float, int, np.generic)


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is nd.ndarray else F.reshape_like(x, y)


def _softmax_loss(F, pred, gt_label, weight, sample_weight, sparse_label, axis, batch_axis):
    pred = F.log_softmax(pred, axis)
    if sparse_label:
        loss = -F.pick(pred, gt_label, axis=axis, keepdims=True)
    else:
        label = _reshape_like(F, gt_label, pred)
        loss = -F.sum(pred * label, axis=axis, keepdims=True)
    loss = _apply_weighting(F, loss, weight, sample_weight)
    return F.mean(loss, axis=batch_axis, exclude=True)


# Angular/cosine margin based loss
class ArcLoss(Loss):
    r"""ArcLoss from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    <https://arxiv.org/abs/1801.07698>`_ paper.

    Parameters
    ----------
    :param s: int. Scale parameter for loss.
    :param m:

    """

    def __init__(self, s, m, classes, easy_margin=False, margin_verbose=False,
                 # in_units=0, weight_initializer=None, dtype='float32',
                 axis=-1, sparse_label=True,
                 weight=None, batch_axis=0, **kwargs):
        super().__init__(weight, batch_axis, **kwargs)
        assert s > 0.0
        assert 0.0 <= m < (math.pi / 2)

        self.margin_s = s
        self.margin_m = m
        self.margin_verbose = True if margin_verbose > 0 else False
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        # threshold = 0.0
        self.threshold = math.cos(math.pi - m)
        self._classes = classes
        self._axis = axis
        self._sparse_label = sparse_label

    def hybrid_forward(self, F, x, gt_label, sample_weight=None, *args, **kwargs):

        zy = F.pick(x, gt_label, axis=1)  # 得到fc7中gt_label位置的值。(B,1)或者(B)，即当前batch中yi处的scos(theta)
        cos_theta = zy / self.margin_s

        if self.easy_margin:
            cond = F.Activation(data=cos_theta, act_type='relu')

        else:
            cond_v = cos_theta - self.threshold
            cond = F.Activation(data=cond_v, act_type='relu')

        sin_theta = F.sqrt(1.0 - cos_theta * cos_theta)
        new_zy = cos_theta * self.cos_m
        b = sin_theta * self.sin_m
        new_zy = new_zy - b
        new_zy = new_zy * self.margin_s
        if self.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - self.margin_s * self.mm
        new_zy = F.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = F.expand_dims(diff, 1)
        gt_one_hot = F.one_hot(gt_label, depth=self._classes, on_value=1.0, off_value=0.0)
        body = F.broadcast_mul(gt_one_hot, diff)
        fc7 = x + body
        return _softmax_loss(F, fc7, gt_label, self._weight, sample_weight,
                             self._sparse_label, self._axis, self._batch_axis)


# Euclidean distance based loss
class TripletLoss(Loss):
    r"""Calculates triplet loss given three input tensors and a positive margin.
    Triplet loss measures the relative similarity between prediction, a positive
    example and a negative example:

    .. math::
        L = \sum_i \max(\Vert {pred}_i - {pos_i} \Vert_2^2 -
                        \Vert {pred}_i - {neg_i} \Vert_2^2 + {margin}, 0)

    `pred`, `positive` and `negative` can have arbitrary shape as long as they
    have the same number of elements.

    Parameters
    ----------
    margin : float
        Margin of separation between correct and incorrect pair.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **positive**: positive example tensor with arbitrary shape. Must have
          the same size as pred.
        - **negative**: negative example tensor with arbitrary shape Must have
          the same size as pred.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(TripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, pred, positive, negative):
        positive = _reshape_like(F, positive, pred)
        negative = _reshape_like(F, negative, pred)
        loss = F.sum(F.square(pred - positive) - F.square(pred - negative),
                     axis=self._batch_axis, exclude=True)
        loss = F.relu(loss + self._margin)
        return _apply_weighting(F, loss, self._weight, None)

