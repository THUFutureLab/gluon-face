# -*- coding: utf-8 -*-
# @Author  : PistonYang(pistonyang@gmail.com)

import pytest
import numpy as np
from gluonfr.data.dataset import FRValDataset, FRTrainRecordDataset

train_sets = ['vgg', 'webface', 'emore']
val_sets = ['agedb_30', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'lfw']


def parse_sets(datasets):
    return [[dataset] for dataset in datasets]


@pytest.fixture(params=parse_sets(train_sets))
def get_train_datasets(request):
    dt = FRTrainRecordDataset(*request.param)
    return dt


@pytest.fixture(params=parse_sets(val_sets))
def get_val_datasets(request):
    dt = FRValDataset(*request.param)
    return dt


def test_train_datasets(get_train_datasets):
    dt = get_train_datasets
    for _ in range(10):
        index = np.random.randint(0, len(dt))
        img, label = dt[index]
        assert img.shape == (112, 112, 3)
        assert type(label) is float


def test_val_datasets(get_val_datasets):
    dt = get_val_datasets
    for _ in range(10):
        index = np.random.randint(0, len(dt))
        imgs, label = dt[index]
        assert imgs[0].shape == imgs[1].shape == (112, 112, 3)
        assert label in (0, 1)
