# Adapted from https://github.com/taoqi98/FedNewsRec/blob/master/code/models.py

import torch
import torch.nn as nn

from newsreclib.models.components.layers.attention import AdditiveAttention


class UserEncoder(nn.Module):
    """Implements the user encoder of CenNewsRec.

    Reference: Qi, Tao, Fangzhao Wu, Chuhan Wu, Yongfeng Huang, and Xing Xie. "Privacy-Preserving News Recommendation Model Learning." In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1423-1432. 2020.

    For further details, please refer to the `paper <https://aclanthology.org/2020.findings-emnlp.128/>`_

    Attributes:
        num_filters:
            The number of input features in the ``MultiheadAttention``
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
        gru_hidden_dim:
            The number of features in the hidden state of the ``GRU``.
        num_recent_news:
            Number of recent news to be encoded in the short-term user representation.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        num_filters: int,
        num_heads: int,
        query_dim: int,
        gru_hidden_dim: int,
        num_recent_news: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(num_recent_news, int):
            raise ValueError(
                f"Expected keyword argument `num_recent_news` to be an `int` but got {num_recent_news}"
            )

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.num_recent_news = num_recent_news

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=num_filters, num_heads=num_heads
        )
        self.additive_attention = AdditiveAttention(input_dim=num_filters, query_dim=query_dim)
        self.gru = nn.GRU(input_size=num_filters, hidden_size=gru_hidden_dim, batch_first=True)
        self.final_additive_attention = AdditiveAttention(
            input_dim=gru_hidden_dim, query_dim=query_dim
        )
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, hist_news_vector: torch.Tensor) -> torch.Tensor:
        # long-term user representation
        # batch_size, num_clicked_news, num_filters
        longterm_user_vector, _ = self.multihead_attention(
            hist_news_vector, hist_news_vector, hist_news_vector
        )
        longterm_user_vector = self.dropout(longterm_user_vector)

        # batch_size, num_filters
        longterm_user_vector = self.additive_attention(longterm_user_vector)

        # short-term user representation
        # batch_size, num_recent_news, num_filters
        recent_hist_news = hist_news_vector[:, -self.num_recent_news :, :]

        # 1, batch_size, gru_hidden_dim
        _, hidden = self.gru(recent_hist_news)

        # batch_size, gru_hidden_dim
        shortterm_user_vector = hidden.squeeze(dim=0)

        # aggregated user representation
        # batch_size, 2, gru_hidden_dim
        user_vector = torch.stack([shortterm_user_vector, longterm_user_vector], dim=1)

        # batch_size, gru_hidden_dim
        user_vector = self.final_additive_attention(user_vector)

        return user_vector
