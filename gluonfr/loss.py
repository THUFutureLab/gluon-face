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
"""Custom losses"""
import math
import numpy as np
from mxnet import nd, init
from mxnet.gluon.loss import Loss, SoftmaxCrossEntropyLoss

__all__ = ["SoftmaxCrossEntropyLoss", "ArcLoss", "TripletLoss", "RingLoss", "CosLoss",
           "L2Softmax", "ASoftmax", "CenterLoss", "ContrastiveLoss", "LGMLoss"]
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


class L2Softmax(SoftmaxCrossEntropyLoss):
    r"""L2Softmax from
    `"L2-constrained Softmax Loss for Discriminative Face Verification"
    <https://arxiv.org/abs/1703.09507>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    alpha: float.
        The scaling parameter, a hypersphere with small alpha
        will limit surface area for embedding features.
    p: float, default is 0.9.
        The expected average softmax probability for correctly
        classifying a feature.
    from_normx: bool, default is False.
         Whether input has already been normalized.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, alpha, p=0.9, from_normx=False,
                 axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, "For given probability of p={}, alpha should higher than {}.".format(p, alpha_low)
        self.alpha = alpha
        self._from_normx = from_normx

    def hybrid_forward(self, F, x, label, sample_weight=None):
        if not self._from_normx:
            x = F.L2Normalization(x, mode='instance', name='fc1n')
        fc7 = x * self.alpha
        return super().hybrid_forward(F, pred=fc7, label=label, sample_weight=sample_weight)


class CosLoss(SoftmaxCrossEntropyLoss):
    r"""CosLoss from
       `"CosFace: Large Margin Cosine Loss for Deep Face Recognition"
       <https://arxiv.org/abs/1801.09414>`_ paper.

       It is also AM-Softmax from
       `"Additive Margin Softmax for Face Verification"
       <https://arxiv.org/abs/1801.05599>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float, default 0.4
        Margin parameter for loss.
    s: int, default 64
        Scale parameter for loss.


    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m, s, axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)
        self._classes = classes
        self._scale = s
        self._margin = m

    def hybrid_forward(self, F, x, label, sample_weight=None):
        if self._sparse_label:
            one_hot_label = F.one_hot(label, depth=self._classes, on_value=1.0, off_value=0.0)
        else:
            one_hot_label = label

        body = one_hot_label * self._margin
        fc7 = (x - body) * self._scale

        return super().hybrid_forward(F, pred=fc7, label=label, sample_weight=sample_weight)


class ArcLoss(SoftmaxCrossEntropyLoss):
    r"""ArcLoss from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    <https://arxiv.org/abs/1801.07698>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float.
        Margin parameter for loss.
    s: int.
        Scale parameter for loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m=0.5, s=64,
                 axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label,
                         weight=weight, batch_axis=batch_axis, **kwargs)
        assert s > 0.
        assert 0 <= m < (math.pi / 2)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self._classes = classes

    def hybrid_forward(self, F, pred, label, sample_weight=None, *args, **kwargs):

        cos_t = F.pick(pred, label, axis=1)  # cos(theta_yi)

        cond_v = cos_t - self.threshold
        cond = F.Activation(data=cond_v, act_type='relu')
        sin_t = 1.0 - F.sqrt(cos_t * cos_t)  # sin(theta)
        new_zy = cos_t * self.cos_m - sin_t * self.sin_m  # cos(theta_yi + m)

        zy_keep = cos_t - self.mm  # (cos(theta_yi) - sin(pi - m)*m)
        new_zy = F.where(cond, new_zy, zy_keep)
        diff = new_zy - cos_t  # cos(theta_yi + m) - cos(theta_yi)
        diff = F.expand_dims(diff, 1)  # shape=(b, 1)
        gt_one_hot = F.one_hot(label, depth=self._classes, on_value=1.0, off_value=0.0)  # shape=(b,classes)
        body = F.broadcast_mul(gt_one_hot, diff)
        pred = pred + body
        pred = pred * self.s

        return super().hybrid_forward(F, pred=pred, label=label, sample_weight=sample_weight)


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
    margin: float
        Margin of separation between correct and incorrect pair.
    weight: float or None
        Global scalar weight for loss.
    batch_axis: int, default 0
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


class ContrastiveLoss(Loss):
    r"""Computes the contrastive loss.
    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Parameters
    ----------
    margin: float, default is 1.
        Margin term in the loss definition.

    Inputs:

        - **anchor**: prediction tensor. Embeddings should be l2 normalized.
        - **positive**: positive example tensor with arbitrary shape. Must have
          the same size as anchor. Embeddings should be l2 normalized.
        - **labels**: array with shape (batch_size,) of
          binary labels indicating positive vs negative pair.

    Outputs:
        - **loss**:  loss tensor with shape (batch_size,).Dimensions other than
          batch_axis are averaged out.
      """

    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super().__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, anchor, positive, labels):
        positive = _reshape_like(F, positive, anchor)
        dists = F.norm(F.square(anchor - positive), axis=1)
        loss = labels * F.square(dists) + (1 - labels) * F.square(F.maximum(self._margin - dists, 0))
        return _apply_weighting(F, loss, self._weight, None)


class RingLoss(SoftmaxCrossEntropyLoss):
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

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.

    """

    def __init__(self, lamda, weight_initializer=None, dtype='float32',
                 axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)

        self._lamda = lamda
        self.R = self.params.get('R', shape=(1,), init=weight_initializer,
                                 dtype=dtype, allow_deferred_init=True)

    def hybrid_forward(self, F, pred, label, embedding, R, sample_weight=None):
        # RingLoss
        emb_norm = F.norm(embedding, axis=1)
        loss_r = F.square(F.broadcast_sub(emb_norm, R)) * 0.5
        loss_r = _apply_weighting(F, loss_r, self._weight, sample_weight)

        # Softmax
        loss_sm = super().hybrid_forward(F, pred, label, sample_weight)

        return loss_sm + self._lamda * loss_r


class ASoftmax(SoftmaxCrossEntropyLoss):
    r"""ASoftmax from
    `"SphereFace: Deep Hypersphere Embedding for Face Recognition"
    <https://arxiv.org/pdf/1704.08063.pdf>`_ paper.
    input(weight, x) has already been normalized

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float.
        Margin parameter for loss.
    s: int.
        Scale parameter for loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m, s, axis=-1, phiflag=True,
                 sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)
        self._classes = classes
        self._scale = s
        self._margin = m
        self._phiflag = phiflag
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    @staticmethod
    def _myphi(x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
            x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

    def hybrid_forward(self, F, x, label, sample_weight=None):
        cos_theta = F.clip(x, -1, 1)

        if self._phiflag:
            cos_m_theta = self.mlambda[int(self._margin)](cos_theta)
            theta = cos_theta.arccos()
            k = (self._margin * theta / math.pi).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.arccos()
            phi_theta = self._myphi(theta, self._margin)
            phi_theta = phi_theta.clip(-1 * self._margin, 1)

        if self._sparse_label:
            one_hot_label = F.one_hot(label, depth=self._classes, on_value=1.0, off_value=0.0)
        else:
            one_hot_label = label

        self.it += 1
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        diff = (phi_theta - x) * 1.0 / (1 + self.lamb)

        body = one_hot_label * diff
        fc7 = (x + body) * self._scale

        return super().hybrid_forward(F, pred=fc7, label=label, sample_weight=sample_weight)


class CenterLoss(SoftmaxCrossEntropyLoss):
    """Computes the Center Loss from
    `"A Discriminative Feature Learning Approach for Deep Face Recognition"
    <http://ydwen.github.io/papers/WenECCV16.pdf>`_paper.
    Implementation is refer to
    "https://github.com/ShownX/mxnet-center-loss/blob/master/center_loss.py"

    Parameters
    ----------
    classes: int.
        Number of classes.

    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and center loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.

    """

    def __init__(self, classes, embedding_size, lamda,
                 weight_initializer=init.Xavier(magnitude=2.24), dtype='float32',
                 axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)
        self._lmda = lamda
        self.centers = self.params.get('centers', shape=(classes, embedding_size), init=weight_initializer,
                                       dtype=dtype, allow_deferred_init=True)

    def hybrid_forward(self, F, x, label, embeddings, centers, sample_weight=None):
        hist = F.array(np.bincount(label.asnumpy().astype(int)))

        centers_count = F.take(hist, label)
        centers_selected = F.take(centers, label)
        loss_c = self._lmda * 0.5 * F.sum(F.square(embeddings - centers_selected), 1) / centers_count

        # Softmax
        loss_sm = super().hybrid_forward(F, x, label, sample_weight)
        return loss_sm + loss_c


class LGMLoss(Loss):
    """LGM Loss from
    `"Rethinking Feature Distribution for Loss Functions in Image Classification"
    <https://arxiv.org/abs/1803.02988>`_paper.
    Implementation is refer to
    https://github.com/LeeJuly30/L-GM-Loss-For-Gluon/blob/master/L_GM.py

    Parameters
    ----------
    num_classes: int.
        The num of classes.
    embedding_size: int.
        The size of embedding feature.
    alpha: float.
        A non-negative parameter controlling the size of the expected margin between
        two classes on the training set.
    lamda: float.
        A non-negative weighting coefficient.
    lr_mult: float.
        Var updating need a relatively low learning rate compared to the overall learning rate.
    """

    def __init__(self, num_classes, embedding_size, alpha, lamda, lr_mult, **kwargs):
        super().__init__(weight=None, batch_axis=0, **kwargs)
        self._num_class = num_classes
        self._feature_dim = embedding_size
        self._alpha = alpha
        self._lamda = lamda
        self.mean = self.params.get('mean', shape=(num_classes, embedding_size), init=init.Xavier())
        self.var = self.params.get('var', shape=(num_classes, embedding_size), init=init.Constant(1), lr_mult=lr_mult)

    def _classification_probability(self, F, x, label, mean, var):
        reshape_var = F.reshape(var, (-1, 1, self._feature_dim))
        reshape_mean = F.reshape(mean, (-1, 1, self._feature_dim))
        x = F.expand_dims(x, 0)
        x = F.broadcast_minus(x, reshape_mean)
        d_z = F.elemwise_mul(F.broadcast_div(x, (reshape_var + 1e-8)), x)
        d_z = F.transpose(F.sum(d_z, axis=2) / 2)

        mask = F.one_hot(label, self._num_class) * self._alpha + 1
        margin_d_z = d_z * mask
        probability = F.broadcast_div(F.exp(-margin_d_z), (F.sqrt(F.prod(var, 1)) + 1e-8))
        return probability, d_z

    def hybrid_forward(self, F, x, label, mean, var):
        probability, m_distance = self._classification_probability(F, x, label, mean, var)

        # classification loss
        class_probability = F.pick(probability, label, axis=1)
        loss_cls = -F.log(class_probability / (F.sum(probability, 1) + 1e-8) + 1e-8)

        # likehood loss
        loss_lkd = F.pick(m_distance, label, axis=1)
        l_gm_loss = loss_cls + self._lamda * loss_lkd
        return l_gm_loss, probability
