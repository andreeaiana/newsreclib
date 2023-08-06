Summary of the Datasets
=======================

NewsRecLib integrates, to date, 2 benchmark datasets:
*MIND* and *Adressa*. Each is supported in two variants,
depending on the dataset size.

.. py:module:: newsreclib.data

MIND Dataset
------------

NewsRecLib provides downloading, parsing, annotation, and loading functionalities for two variants
of the `MIND <https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md>`_:
MINDsmall and MINDlarge.

Reference: Wu, Fangzhao, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu et al. "Mind: A large-scale dataset for news recommendation." In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 3597-3606. 2020.

For further details, please refer to the `paper <https://aclanthology.org/2020.acl-main.331/>`_

.. autosummary::
   mind_rec_datamodule.MINDRecDataModule
   mind_news_datamodule.MINDNewsDataModule

Adreesa Dataset
---------------

NewsRecLib provides downloading, parsing, annotation, and loading functionalities for two variants
of the `Adressa <https://reclab.idi.ntnu.no/dataset/>`_: 1-week and 3-month.

Reference: Gulla, Jon Atle, Lemei Zhang, Peng Liu, Özlem Özgöbek, and Xiaomeng Su. "The adressa dataset for news recommendation." In Proceedings of the international conference on web intelligence, pp. 1042-1048. 2017.

For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3106426.3109436>`_

.. autosummary::
   adressa_rec_datamodule.AdressaRecDataModule
   adressa_news_datamodule.AdressaNewsDataModule
