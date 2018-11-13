# @File  : utils.py
# @Author: X.Yang&Xt.Hao
# @Contact : pistonyang@gmail.com, haoxintongpku@gmail.com
# @Date  : 18-11-1
import sklearn
from mxnet import gluon, autograd as ag
from mxnet.gluon.data.vision import transforms
from gluonfr.metrics.verification import FaceVerification


def inf_train_gen(loader):
    """
    Using iterations train network.
    :param loader: Dataloader
    :return: batch of data
    """
    while True:
        for batch in loader:
            yield batch


transform_test = transforms.Compose([
    transforms.ToTensor()
])

_transform_train = transforms.Compose([
    transforms.RandomBrightness(0.3),
    transforms.RandomContrast(0.3),
    transforms.RandomSaturation(0.3),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor()
])


def transform_train(data, label):
    im = _transform_train(data)
    return im, label


def validate(net, ctx, val_datas, targets, nfolds=10, norm=True):
    metric = FaceVerification(nfolds)
    results = []
    for loader, name in zip(val_datas, targets):
        metric.reset()
        for i, batch in enumerate(loader):
            data0s = gluon.utils.split_and_load(batch[0][0], ctx, even_split=False)
            data1s = gluon.utils.split_and_load(batch[0][1], ctx, even_split=False)
            issame_list = gluon.utils.split_and_load(batch[1], ctx, even_split=False)

            embedding0s = [net(X)[0] for X in data0s]
            embedding1s = [net(X)[0] for X in data1s]
            if norm:
                embedding0s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding0s]
                embedding1s = [sklearn.preprocessing.normalize(e.asnumpy()) for e in embedding1s]

            for embedding0, embedding1, issame in zip(embedding0s, embedding1s, issame_list):
                metric.update(issame, embedding0, embedding1)

        tpr, fpr, accuracy, val, val_std, far, accuracy_std = metric.get()
        results.append("{}: {:.6f}+-{:.6f}".format(name, accuracy, accuracy_std))
    return results
