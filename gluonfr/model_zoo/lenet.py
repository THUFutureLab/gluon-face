# @File  : lenet.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 18-11-2
"""This net is proposed form center loss.
We also use it to test other losses."""
from mxnet.gluon import nn
from ..nn.basic_blocks import FR_Base


class LeNet_m(FR_Base):
    r"""LeNet_m model is lenet++ from
    `"A Discriminative Feature Learning Approach for Deep Face Recognition"
    <https://ydwen.github.io/papers/WenECCV16.pdf>`_ paper.

    Parameters
    ----------
    embedding_size : int
        Number units of embedding layer.
    """
    def __init__(self, embedding_size=2, weight_norm=True, feature_norm=True, **kwargs):
        super().__init__(10, embedding_size, weight_norm, feature_norm, **kwargs)
        self.features = nn.HybridSequential()
        self.features.add(
            nn.Conv2D(32, 5, padding=2, strides=1),
            nn.PReLU(),
            nn.Conv2D(32, 5, padding=2, strides=1),
            nn.PReLU(),
            nn.MaxPool2D(2, strides=2),
            nn.Conv2D(64, 5, padding=2, strides=1),
            nn.PReLU(),
            nn.Conv2D(64, 5, padding=2, strides=1),
            nn.PReLU(),
            nn.MaxPool2D(2, strides=2),
            nn.Conv2D(128, 5, padding=2, strides=1),
            nn.PReLU(),
            nn.Conv2D(128, 5, padding=2, strides=1),
            nn.PReLU(),
            nn.MaxPool2D(2, strides=2),
            nn.Flatten(),
            nn.Dense(embedding_size, use_bias=False),
            nn.PReLU()
        )
