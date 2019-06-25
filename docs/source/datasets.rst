Datasets
========

``gluonfr.data`` provides input pipeline for training and validation, all
datasets is aligned by mtcnn and cropped to (112, 112) by DeepInsight,
they converted images to ``train.rec``, ``train.idx`` and
``val_data.bin`` files, please check out
`[insightface/Dataset-Zoo] <https://github.com/deepinsight/insightface/wiki/Dataset-Zoo>`__
for more information. In ``examples/dali_utils.py``, there is a simple
example of Nvidia-DALI. It is worth trying when data augmentation with
cpu can not satisfy the speed of gpu training,

The files should be prepared like:

::

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

We use ``~/.mxnet/datasets`` as default dataset root to match mxnet setting.

References
----------

- LFW
    `"Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments"
    <http://vis-www.cs.umass.edu/lfw/lfw.pdf>`__

- CALFW
    `"A Database for Studying Cross-Age Face Recognition in Unconstrained Environments"
    <http://arxiv.org/abs/1708.08197>`__

- CPLFW
    `"Cross-pose LFW: A database for studying cross-pose face recognition in unconstrained environments"
    <http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf>`__

- CFP_fp, CFP_ff
    `"Frontal to Profile Face Verification in the Wild" <http://www.cfpw.io/paper.pdf>`__

- AgeDB_30
    `"AgeDB: the first manually collected, in-the-wild age database"
    <https://ibug.doc.ic.ac.uk/media/uploads/documents/agedb.pdf>`__

- VGG2_fp
    `"VGGFace2: A dataset for recognising faces across pose and age"
    <https://arxiv.org/abs/1710.08092>`__
