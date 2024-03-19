from typing import Any, Dict, TypedDict

import torch


class RecommendationBatch(TypedDict):
    """Batch used for recommendation.

    Attributes:
        batch_hist:
            Batch of histories of users.
        batch_cand:
            Batch of candidates for each user.
        x_hist:
            Dictionary of news from a the users' history, mapping news features to values.
        x_cand
            Dictionary of news from a the users' candidates, mapping news features to values.
        labels:
            Ground truth specifying whether the news is relevant to the user.
        user_ids:
            Original user IDs of the users included in the batch.
        user_idx:
            Indices of users included in the batch (e.g., for creating embedding matrix).
    """

    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Dict[str, Any]
    x_cand: Dict[str, Any]
    labels: torch.Tensor
    user_ids: torch.Tensor
    user_idx: torch.Tensor


class NewsBatch(TypedDict):
    """Batch used for reshaping the embedding space based on an aspect of the news.

    Reference: Iana, Andreea, Goran Glavaš, and Heiko Paulheim.
    "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation."
    arXiv preprint arXiv:2307.16089 (2023).
    `https://arxiv.org/pdf/2307.16089.pdf`

    Attributes:
        news:
            Dictionary mapping features of news to values.
        labels:
            Labels of news based on the specified aspect.
    """

    news: Dict[str, Any]
    labels: torch.Tensor
