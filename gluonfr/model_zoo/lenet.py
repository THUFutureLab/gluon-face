# @File  : lenet.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 18-11-2
"""This net is proposed form center loss.
We also use it to test other losses."""
from mxnet.gluon import nn
from gluonfr.nn.basic_blocks import NormDense


class LeNet_m(nn.HybridBlock):
    r"""LeNet_m model is lenet++ from
    `"A Discriminative Feature Learning Approach for Deep Face Recognition"
    <https://ydwen.github.io/papers/WenECCV16.pdf>`_ paper.

    Parameters
    ----------
    embedding_size : int
        Number units of embedding layer.
    """
    def __init__(self, embedding_size=2, **kwargs):
        super().__init__(**kwargs)
        self.feature = nn.HybridSequential()
        self.feature.add(
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
        self.output = NormDense(10, weight_norm=True, feature_norm=True, in_units=embedding_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        embedding = self.feature(x)
        output = self.output(embedding)
        return embedding, output
