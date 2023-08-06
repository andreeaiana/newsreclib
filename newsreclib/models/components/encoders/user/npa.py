# Adapted from https://github.com/info-ruc/ai20projects/blob/ca6f993cfa569250b3116921f4b481d01de36197/2018202180/src/scripts/models/NPA.py

import torch
import torch.nn as nn

from newsreclib.models.components.layers.attention import PersonalizedAttention
from newsreclib.models.components.layers.projection import UserPreferenceQueryProjection


class UserEncoder(nn.Module):
    """Implements the user encoder of NPA.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "NPA: neural news recommendation with personalized attention." In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2576-2584. 2019.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3292500.3330665>`_

    Attributes:
        user_embed_dim:
            The number of feature in the user vector.
        num_filters:
            The number of filters in the ``PersonalizedAttention``.
        preference_query_dim:
            The number of features in the preference query vector.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        user_embed_dim: int,
        num_filters: int,
        preference_query_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        # initialize
        self.news_query_projection = UserPreferenceQueryProjection(
            user_embed_dim=user_embed_dim,
            preference_query_dim=preference_query_dim,
            dropout_probability=dropout_probability,
        )
        self.personalized_attention = PersonalizedAttention(
            preference_query_dim=preference_query_dim, num_filters=num_filters
        )

    def forward(
        self, hist_news_vector: torch.Tensor, projected_users: torch.Tensor
    ) -> torch.Tensor:
        # batch_size, query_preference_dim
        news_preference_query = self.news_query_projection(projected_users)

        # batch_size, num_filters
        user_vector = self.personalized_attention(
            news_preference_query, hist_news_vector.permute(0, 2, 1)
        )

        return user_vector
