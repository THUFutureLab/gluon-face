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
from mxnet import nd, init
from mxnet.gluon.loss import Loss

__all__ = ["ArcLoss", "TripletLoss", "RingLoss"]
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

    def hybrid_forward(self, F, x, label, sample_weight=None, *args, **kwargs):
        # nd.where()
        cos_theta = F.pick(x, label, axis=1)  # 得到fc7中gt_label位置的值。(B,1)或者(B)，即当前batch中yi处的scos(theta)

        #
        # if self.easy_margin:
        #     cond = F.Activation(data=cos_theta, act_type='relu')
        #
        # else:
        #     cond_v = cos_theta - self.threshold
        #     cond = F.Activation(data=cond_v, act_type='relu')

        sin_theta = F.sqrt(1.0 - cos_theta * cos_theta)

        cos_theta_add_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        # new_zy = new_zy * self.margin_s
        # if self.easy_margin:
        #     zy_keep = zy
        # else:
        #     zy_keep = zy - self.margin_s * self.mm
        # new_zy = F.where(cond, new_zy, zy_keep)

        diff = F.expand_dims(cos_theta_add_m - cos_theta, 1)
        gt_one_hot = F.one_hot(label, depth=self._classes, on_value=1.0, off_value=0.0)
        body = F.broadcast_mul(gt_one_hot, diff)
        fc7 = (x + body) * self.margin_s
        return _softmax_loss(F, fc7, label, self._weight, sample_weight,
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


class RingLoss(Loss):
    """Computes the Ring Loss from
    `"Ring loss: Convex Feature Normalization for Face Recognition"
    <https://arxiv.org/abs/1803.00130>`_paper.

    .. math::
        p = \softmax({pred})

        L_SM = -\sum_i \log p_{i,{label}_i}

        L_R = \frac{\lambda}{2m} \sum_{i=1}^{m} (\Vert \mathcal{F}({x}_i)\Vert_2 - R )^2

     Parameters
    ----------
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and ring loss.
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **embedding**: the output of embedding layer before classification.
          It should be (batch size, feature size) shape.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.

    """

    def __init__(self, lamda, weight=None, batch_axis=0,
                 axis=-1, sparse_label=True,
                 weight_initializer=init.Constant(1.0), dtype='float32', **kwargs):
        super().__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        # Softmax
        self._axis = axis
        self._sparse_label = sparse_label

        # RingLoss
        self._lamda = lamda
        self.R = self.params.get('R', shape=(1,),
                                 init=weight_initializer, dtype=dtype,
                                 allow_deferred_init=True)

    def hybrid_forward(self, F, pred, label, embedding, R, sample_weight=None):
        # RingLoss
        emb_norm = F.norm(embedding, axis=1)
        loss_r = F.square(F.broadcast_sub(emb_norm, R))
        loss_r = F.mean(loss_r, keepdims=True) * 0.5

        # Softmax
        pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss_sm = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss_sm = -F.sum(pred * label, axis=self._axis, keepdims=True)
        loss_sm = F.mean(loss_sm, axis=self._batch_axis, exclude=True)

        loss = F.broadcast_add(loss_sm, self._lamda * loss_r)
        return _apply_weighting(F, loss, self._weight, sample_weight)
