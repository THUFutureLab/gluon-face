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
""""""
import mxnet as mx
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate

__all__ = ["FaceVerification"]


class FaceVerification(mx.metric.EvalMetric):
    """ Compute confusion matrix of 1:1 problem in face verification or other fields.
    Use update() to collect the outputs and compute distance in each batch, then use get() to compute the
    confusion matrix and accuracy of the val dataset.

    Parameters
    ----------
    nfolds: int, default is 10

    thresholds: ndarray, default is None.
        Use np.arange to generate thresholds. If thresholds=None, np.arange(0, 2, 0.01) will be used for
        euclidean distance.

    far_target: float, default is 1e-3.
        This is used to get the verification accuracy of expected far.

    dist_type: int, default is 0.
        Option value is {0, 1}, 0 for euclidean distance, 1 for cosine similarity. Here for cosine distance,
        we use `1 - cosine` as the final distances.

    """

    def __init__(self, nfolds=10, thresholds=None, far_target=1e-3, dist_type=0):
        super().__init__("FaceVerification")
        self.far_target = far_target
        self._nfolds = nfolds
        self._dists = []
        self._issame = []
        default_thresholds = np.arange(0, 2, 0.01) if dist_type == 0 else np.arange(0, 1, 0.01)
        self._thresholds = thresholds if thresholds is not None else default_thresholds
        self.reset()
        self._dist_type = dist_type

    # noinspection PyMethodOverriding
    def update(self, labels: mx.nd.NDArray, embeddings0: mx.nd.NDArray, embeddings1: mx.nd.NDArray):
        """

        :param labels: NDArray.
        :param embeddings0: NDArray.
        :param embeddings1: NDArray.
        :return:
        """

        embeddings0 = embeddings0.asnumpy() if not isinstance(embeddings0, np.ndarray) else embeddings0
        embeddings1 = embeddings1.asnumpy() if not isinstance(embeddings1, np.ndarray) else embeddings1
        labels = labels.asnumpy() if not isinstance(labels, np.ndarray) else labels

        if self._dist_type == 0:
            diff = np.subtract(embeddings0, embeddings1)
            dists = np.sqrt(np.sum(np.square(diff), 1))
        else:
            dists = 1 - np.sum(np.multiply(embeddings0, embeddings1), axis=1) / \
                    (np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1))

        self._dists += [d for d in dists]
        self._issame += [l for l in labels]

    def get(self):
        tpr, fpr, accuracy = calculate_roc(self._thresholds, np.asarray(self._dists),
                                           np.asarray(self._issame), self._nfolds)

        val, val_std, far = calculate_val(self._thresholds, np.asarray(self._dists),
                                          np.asarray(self._issame), self.far_target, self._nfolds)
        acc, acc_std = np.mean(accuracy), np.std(accuracy)
        return tpr, fpr, acc, val, val_std, far, acc_std

    def reset(self):
        self._dists = []
        self._issame = []


# code below is modified from project <Facenet (David Sandberg)>
class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, dist, actual_issame, nrof_folds=10):
    assert len(dist) == len(actual_issame), "Shape of predicts and labels mismatch!"

    nrof_pairs = len(dist)
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds,))
    indices = np.arange(nrof_pairs)
    dist = np.array(dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds,))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])

        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], \
            fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set],
                                                                  actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, dist, actual_issame, far_target, nrof_folds=10):
    assert len(dist) == len(actual_issame), "Shape of predicts and labels mismatch!"

    nrof_pairs = len(dist)
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)
    dist = np.array(dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])

        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    val_std = np.std(val)
    far_mean = np.mean(far)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
