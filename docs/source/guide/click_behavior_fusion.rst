Click Behavior Fusion
=====================

NewsRecLib supports 2 strategies for aggregating users' click behaviors:
**early fusion** and **late fusion**.

Early Fusion
------------
This is the predominant paradigm used in all recommendation models.
It involves aggregating the representations of clicked news
(i.e., building an explicit user representation) before comparison
with the recommendation candidate.

When choosing this option, users will have to select one of the
available user encoders or implement a new one.

Late Fusion
-----------
This light-weight approach replaces user encoders with the mean-pooling
of dot-product scores between the embedding of the candidate :math:`n^c`
and the embeddings of the clicked news :math:`n_i^u`.

Given a candidate news :math:`n^c` and a sequence of news clicked by the
user :math:`H = n_1^u, ..., n_N^u`, the relevance score of the candidate news
with regards to the user :math:`u`'s history is computed as:
:math:`s(\mathbf{n}^c, u) = \frac{1}{N} \sum_{i=1}^N \mathbf{n}^c \cdot \mathbf{n}_i^u`,
where :math:`\mathbf{n}` denotes the embedding of a news and :math:`N` the history length.

For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3539618.3592062>`_

Reference: Iana, Andreea, Goran Glavas, and Heiko Paulheim. "Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives." In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2384-2388. 2023.
