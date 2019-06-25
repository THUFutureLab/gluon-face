Model Zoo
=========

Mobilefacenet Result.
~~~~~~~~~~~~~~~~~~~~~

+------------+-------------+---------------+-------------+
| TestSet    | Ours        | Insightface   | Proposed    |
+============+=============+===============+=============+
| LFW:       | **99.56**   | 99.50         | 99.55       |
+------------+-------------+---------------+-------------+
| CFP\_FP:   | **92.98**   | 88.94         | -           |
+------------+-------------+---------------+-------------+
| AgeDB30:   | 95.86       | 95.91         | **96.07**   |
+------------+-------------+---------------+-------------+

Reference:

1. Our `code <https://github.com/THUFutureLab/gluon-face>`__ train
`script <https://github.com/THUFutureLab/gluon-face/blob/master/scripts/mobilefacenet-arcface.sh>`__
and log/model in (\ `Baidu:y5zh <https://pan.baidu.com/s/13Diy2jS1rkbWEZuQ5J8wjg>`__,
`Google Drive <https://drive.google.com/file/d/1RXXb19GhjX04ZjmYhsFW1CSMpRyU9CiP/view?usp=sharing>`__).

2. Insightface `result <https://github.com/deepinsight/insightface/issues/214>`__.
3. Mobilefacenets `papers <https://arxiv.org/pdf/1804.07573.pdf>`__.(No open project)

Details
^^^^^^^

+--------------+--------------------------+--------------------------+
| Flip         | False                    | True                     |
+==============+==========================+==========================+
| lfw:         | 0.995500+-0.003337       | **0.995667+-0.003432**   |
+--------------+--------------------------+--------------------------+
| calfw:       | 0.951000+-0.012069       | **0.973083+-0.022889**   |
+--------------+--------------------------+--------------------------+
| cplfw:       | 0.882000+-0.014295       | **0.938556+-0.045234**   |
+--------------+--------------------------+--------------------------+
| cfp\_fp:     | 0.927714+-0.015309       | **0.929880+-0.035907**   |
+--------------+--------------------------+--------------------------+
| agedb\_30:   | **0.958667+-0.008492**   | 0.934903+-0.033667       |
+--------------+--------------------------+--------------------------+
| cfp\_ff:     | **0.995571+-0.002744**   | 0.944868+-0.037657       |
+--------------+--------------------------+--------------------------+
| vgg2\_fp:    | 0.920600+-0.010920       | 0.940581+-0.032677       |
+--------------+--------------------------+--------------------------+

Information
^^^^^^^^^^^

1. Github has some
   `projects <https://github.com/qidiso/mobilefacenet-V2>`__ train to a
   high level, but with embedding\_size of 512,compare with them we use
   embedding\_size of 128 which origin paper proposed, model size is
   only 4.1M.
2. Welcome to use our train script to do more exploration, and if you
   get better results you could make a pr to us.
3. We Pre-trained model through L2-Regularization, output is cos(theta).