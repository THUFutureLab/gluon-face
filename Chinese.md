# Gluon FR Toolkit

此项目灵感来自GluonCV, 并按照其结构组织. 除了帮助研究者和开发者们迅速上手目前最前沿的人脸识别算法, 
也希望能够让更多的人了解Gluon这一好用的工具, 使用MXnet-Gluon进行深度学习算法的研究.

## 安装指南
Gluon-fr支持python3.5或以上版本.
同时需要Mxnet和gluon-cv,可以用以下命令安装它们.
- Mxnet建议安装稳定版或者nightly版,gluon-cv必须安装nightly版.
- 如果你想要训练必须要GPUs.
```shell
pip install gluoncv --pre
pip install mxnet-mkl --pre --upgrade
# if cuda XX is installed
pip install mxnet-cuXXmkl --pre --upgrade
```
之后安装gluon-fr
- 从源码安装(建议)
```shell
pip install git+https://github.com/THUFutureLab/gluon-face.git@master
```
- pip安装
```shell
pip install gluonfr
```

## GluonFR 介绍:
GluonFR 基于gluon-cv, 如果你不太熟悉它可以阅读一个简短的教程.[dmlc 60-minute crash course](http://gluon-crash-course.mxnet.io/).
#### 训练数据: 
这一部分主要提供训练和验证数据的输入. GluonFR目前使用的训练集是由DeepInsight提供, 使用mtcnn进行关键点检测并对齐至(112, 112)大小,
 详情参考[[insightface/Dataset-Zoo]](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). 
另外, data/中还包括nvidia-dali库的使用样例, 在CPU预处理数据成为训练瓶颈时可以考虑试用, 目前dali库中坑还比较多.

训练数据目录结构如下所示:
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
我们使用 `~/.mxnet/datasets` 作为根目录为了保持和mxnet一致.

#### 模型:
mobile_facenet, res_attention_net, se_resnet...

#### Loss函数:
GluonFR 提供了在人脸识别中主流的loss函数, 包括 SoftmaxCrossEntropyLoss, ArcLoss, TripletLoss, 
RingLoss, CosLoss, L2Softmax, ASoftmax, CenterLoss, ContrastiveLoss, ... , 并且我们还会随时更新它们.  
如果有任何遗漏的loss我们没有实现,你可以提交一个 [issue](https://github.com/THUFutureLab/gluon-face/issues) 告知我们.

#### Example:
GluonFR提供了Mnist手写数字识别的训练和可视化代码, 用于验证损失函数的有效性;在人脸识别数据集上基于model-zoo模型完成训练.  
  
## Losses in GluonFR:  
下表中最后一列是论文中在LFW上的最优结果, 数据、网络结构都可能不同, 仅供参考.  

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

## 预训练模型
我们在在WiKi中的[Model_Zoo](wiki/Model_Zoo)提供预训练模型的更多细节.

## Todo

- More pretrained models
- IJB and Megaface Results
- Other losses
- Dataloader for loss depend on how to provide batches like Triplet, ContrastiveLoss, RangeLoss...
- Try GluonCV resnetV1b/c/d/ to improve performance
- Create hosted docs
- Test module
- [x] Pypi package


## 文档

目前还没有文档可供浏览.我们可能会在后期提供它们.

## 作者
{ [haoxintong](https://github.com/haoxintong) [Yangxv](https://github.com/PistonY) [Haoyadong](https://github.com/jiqirenno1) [Sunhao](https://github.com/smartadpole) }

## 讨论区
我们的中文讨论区在 [中文社区Gluon-Forum](https://discuss.gluon.ai/t/topic/9959)

## 参考文献

1. MXNet Documentation and Tutorials [https://zh.diveintodeeplearning.org/](https://zh.diveintodeeplearning.org/)

1. NVIDIA DALI documentation[NVIDIA DALI documentation](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)

1. Deepinsight [insightface](https://github.com/deepinsight/insightface)