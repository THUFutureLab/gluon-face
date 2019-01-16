# Gluon FR Toolkit

GluonFR is a toolkit based on MXnet-Gluon, provides SOTA deep learning algorithm and models in face recognition.

此项目灵感来自GluonCV, 并按照其结构组织. 除了帮助研究者和开发者们迅速上手目前最前沿的人脸识别算法, 
也希望能够让更多的人了解Gluon这一好用的工具, 使用MXnet-Gluon进行深度学习算法的研究.

## Installation
GluonFR supports Python 3.5 or later. 
To install this package you need install GluonCV and MXNet first:
```shell
pip install gluoncv --pre
pip install mxnet-mkl --pre --upgrade
# if cuda XX is installed
pip install mxnet-cuXXmkl --pre --upgrade
```
Then install gluonfr:

- From Socure

```shell
git clone https://github.com/THUFutureLab/gluon-face
cd gluon-face/
python3 setup.py install
```

- Pip

```shell
pip install gluonfr
```

## GluonFR Introduction:
GluonFR is based on MXnet-Gluon, if you are new to it, please check out [dmlc 60-minute crash course](http://gluon-crash-course.mxnet.io/).
#### Data: 
这一部分主要提供训练和验证数据的输入. GluonFR目前使用的训练集是由DeepInsight提供, 使用mtcnn进行关键点检测并对齐至(112, 112)大小,
 详情参考[[insightface/Dataset-Zoo]](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). 
另外, data/中还包括nvidia-dali库的使用样例, 在CPU预处理数据成为训练瓶颈时可以考虑试用, 目前dali库中坑还比较多.

This part provides input pipeline for training and validation, 
all datasets is aligned by mtcnn and cropped to (112, 112) by DeepInsight, 
they converted images to `train.rec`, `train.idx` and `val_data.bin` files,  please check out 
[[insightface/Dataset-Zoo]](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) for more information. 
In `data/dali_utils.py`, there is a simple example of Nvidia-DALI. It is worth trying when data augmentation with cpu 
can not satisfy the speed of gpu training,  

The files should be prepared like:
```
face/
    emore/
        train.rec
        train.idx
        property
    ms1m/
        train.rec
        train.idx
        property
    lfw.bin
    agedb_30.bin
    ...
    vgg2_fp.bin
```
We use `~/.mxnet/datasets` as default dataset root to match mxnet setting.

#### Model:
mobile_facenet, res_attention_net, se_resnet...

#### Loss:
GluonFR provides implement of losses in recent, including SoftmaxCrossEntropyLoss, ArcLoss, TripletLoss, 
RingLoss, CosLoss, L2Softmax, ASoftmax, CenterLoss, ContrastiveLoss, ... , and we will keep updating in future.  
If there is any method we overlooked, please open an [issue](https://github.com/THUFutureLab/gluon-face/issues).

#### Example:
GluonFR提供了Mnist手写数字识别的训练和可视化代码, 用于验证损失函数的有效性;在人脸识别数据集上基于model-zoo模型完成训练.  
`examples/` shows how to use gluonfr to train a face recognition model, and how to get Mnist 2-D 
feature embedding visualization.  
  
## Losses in GluonFR:  
下表中最后一列是论文中在LFW上的最优结果, 数据、网络结构都可能不同, 仅供参考.  
The last column of this chart is the best LFW accuracy reported in paper, they are trained with different data and networks, 
later we will give our results of these method with same train data and network. 

|Method| Paper |Visualization of MNIST|LFW|
|:---|:---:| :---:|:---:|
|Contrastive Loss|[ContrastiveLoss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)|-|-|
|Triplet|[1503.03832](https://arxiv.org/abs/1503.03832)|-|99.63±0.09|
|Center Loss|[CenterLoss](https://ydwen.github.io/papers/WenECCV16.pdf)|<img src="resources/mnist-euclidean/center-train-epoch100.png"/>|99.28 |
|L2-Softmax|[1703.09507](https://arxiv.org/abs/1703.09507)|-|99.33|
|A-Softmax|[1704.08063](https://arxiv.org/abs/1704.08063)|-|99.42|
|CosLoss/AMSoftmax|[1801.05599](https://arxiv.org/abs/1801.05599)/[1801.05599](https://arxiv.org/abs/1801.05599)|<img src="resources/minst-angular/cosloss-train-epoch95.png"/>|99.17|
|Arcloss|[1801.07698](https://arxiv.org/abs/1801.07698)|<img src="resources/minst-angular/arcloss-train-epoch100.png"/>|99.82|
|Ring loss|[1803.00130](https://arxiv.org/abs/1803.00130)|<img src="resources/mnist-euclidean/ringloss-train-epoch95-0.1.png"/>|99.52|
|LGM Loss|[1803.02988](https://arxiv.org/abs/1803.02988)|<img src="resources/mnist-euclidean/LGMloss-train-epoch100.png"/>|99.20±0.03|

## Pretrained Models
To be continued.

## Todo

- More pretrained models
- IJB and Megaface Results
- Other losses
- Dataloader for loss depend on how to provide batches like Triplet, ContrastiveLoss, RangeLoss...
- Try GluonCV resnetV1b/c/d/ to improve performance
- Create hosted docs
- Test module
- [x] Pypi package


## Docs

GluonFR documentation is not available now. 

## Authors
{ [haoxintong](https://github.com/haoxintong) [Yangxv](https://github.com/PistonY) [Haoyadong](https://github.com/jiqirenno1) [Sunhao](https://github.com/smartadpole) }

## Discussion
[中文社区Gluon-Forum](https://discuss.gluon.ai/t/topic/9959) Feel free to use English here :D.

## References

1. MXNet Documentation and Tutorials [https://zh.diveintodeeplearning.org/](https://zh.diveintodeeplearning.org/)

1. NVIDIA DALI documentation[NVIDIA DALI documentation](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)

1. Deepinsight [insightface](https://github.com/deepinsight/insightface)

