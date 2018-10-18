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

import sklearn
from tqdm import tqdm
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import split_and_load
from .verification import FaceVerification


def calculate_accuracy(net, dataset, batch_size, ctx, nfolds=10, norm=True):
    metric = FaceVerification(nfolds)
    loader = DataLoader(dataset, batch_size)
    for i, batch in tqdm(enumerate(loader)):
        data0s = split_and_load(batch[0][0], ctx, even_split=False)
        data1s = split_and_load(batch[0][1], ctx, even_split=False)
        issame_list = split_and_load(batch[1], ctx, even_split=False)

        embedding0s = [net(X) for X in data0s]
        embedding1s = [net(X) for X in data1s]
        if norm:
            embedding0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
            embedding1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]

        for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
            metric.update(issame, embedding0, embedding1)

    tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
    return accuracy, accuracy_std
