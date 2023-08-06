# Adapted from https://github.com/info-ruc/ai20projects/blob/ca6f993cfa569250b3116921f4b481d01de36197/2018202180/src/scripts/models/NPA.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserProjection(nn.Module):
    """Embeds user ID to dense vector through a lookup table.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "NPA: neural news recommendation with personalized attention." In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2576-2584. 2019.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3292500.3330665>`_
    """

    def __init__(self, num_users: int, user_embed_dim: int, dropout_probability: float) -> None:
        super().__init__()

        if not isinstance(num_users, int):
            raise ValueError(
                f"Expected keyword argument `num_users` to be an `int` but got {num_users}"
            )

        if not isinstance(user_embed_dim, int):
            raise ValueError(
                f"Expected keyword argument `user_embed_dim` to be an `int` but got {user_embed_dim}"
            )

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.user_embed = nn.Parameter(torch.rand(num_users, user_embed_dim))
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        """
        Args:
            users:
                Vector of users of size `batch_size`

        Returns:
            Projected users vector of size '(batch_size * user_embedding_dim)`
        """
        projected_users = self.user_embed[users]
        projected_users = self.dropout(projected_users)

        return projected_users


class UserPreferenceQueryProjection(nn.Module):
    """Projects dense user representations to preference query vector.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "NPA: neural news recommendation with personalized attention." In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2576-2584. 2019.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3292500.3330665>`_
    """

    def __init__(
        self, user_embed_dim: int, preference_query_dim: int, dropout_probability: float
    ) -> None:
        super().__init__()

        if not isinstance(user_embed_dim, int):
            raise ValueError(
                f"Expected keyword argument `user_embed_dim` to be an `int` but got {user_embed_dim}"
            )

        if not isinstance(preference_query_dim, int):
            raise ValueError(
                f"Expected keyword argument `preference_query_dim` to be an `int` but got {preference_query_dim}"
            )

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.preference_query_projection = nn.Linear(user_embed_dim, preference_query_dim)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, projected_users: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projected_user:
                Vector of project users of size `(batch_size * embedding_dim)`

        Returns:
            Project query vector of size `(batch_size * preference_dim)`
        """
        query = self.preference_query_projection(projected_users)
        query = F.relu(query)
        query = self.dropout(query)

        return query
