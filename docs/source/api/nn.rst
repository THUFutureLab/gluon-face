gluonfr.nn
==========

Neural Network Components.

.. hint::

  Not every component listed here is `HybridBlock <https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html>`_,
  which means some of them are not hybridizable.
  However, we are trying our best to make sure components required during inference are hybridizable
  so the entire network can be exported and run in other languages.

  For example, encoders are usually non-hybridizable but are only required during training.
  In contrast, decoders are mostly `HybridBlock`_ s.

Basic Blocks
------------

Blocks that usually used in face recognition.

.. currentmodule:: gluonfr.nn.basic_blocks

.. autosummary::
    :nosignatures:



API Reference
-------------

.. automodule:: gluonfr.nn.basic_blocks
    :members:
    :imported-members:
