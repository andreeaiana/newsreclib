Training Objectives
===================

NewsRecLib supports 3 training objectives:
**point-wise classification**, **contrastive learning objectives**, and
**dual training objectives**.

Point-wise classification objectives
------------------------------------
NewsRecLib implements model training with `Cross-Entropy Loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_.
as the most standard classification objective.

Contrastive-learning objectives
-------------------------------
NewsRecLib implements `Supervised Contrastive Loss <https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html>`_
as contrastive-learning objective.

Reference: Khosla, Prannay, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. "Supervised contrastive learning." Advances in neural information processing systems 33 (2020): 18661-18673.

Dual training objectives
------------------------
Models can also be trained with a dual learning objective, which combines
cross-entropy loss :math:`\mathcal{L}_{CE}` and supervised contrastive
loss :math:`\mathcal{L}_{SCL}` with a weighted average.

:math:`\mathcal{L} = (1-\lambda) \mathcal{L}_{CE} + \lambda \mathcal{L}_{SCL}`

Reference: Gunel, Beliz, Jingfei Du, Alexis Conneau, and Ves Stoyanov. "Supervised contrastive learning for pre-trained language model fine-tuning." arXiv preprint arXiv:2011.01403 (2020).
