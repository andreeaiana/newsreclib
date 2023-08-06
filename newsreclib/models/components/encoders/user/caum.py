# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from newsreclib.models.components.layers.attention import DenseAttention


class UserEncoder(nn.Module):
    """Implements the user encoder of CAUM.

    Reference: Qi, Tao, Fangzhao Wu, Chuhan Wu, and Yongfeng Huang. "News recommendation with candidate-aware user modeling." In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1917-1921. 2022.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3477495.3531778>`_

    Attributes:
        news_embed_dim:
            The number of features in the news vector.
        num_filters:
            The number of output features in the first linear layer.
        dense_att_hidden_dim1:
            The number of output features in the first hidden state of the ``DenseAttention``.
        dense_att_hidden_dim2:
            The number of output features in the second hidden state of the ``DenseAttention``.
        user_vector_dim:
            The number of features in the user vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        news_embed_dim: int,
        num_filters: int,
        dense_att_hidden_dim1: int,
        dense_att_hidden_dim2: int,
        user_vector_dim: int,
        num_heads: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(news_embed_dim, int):
            raise ValueError(
                f"Expected keyword argument `news_embed_dim` to be an `int` but got {news_embed_dim}"
            )

        if not isinstance(num_filters, int):
            raise ValueError(
                f"Expected keyword argument `num_filters` to be an `int` but got {num_filters}"
            )

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.dropout2 = nn.Dropout(p=dropout_probability)
        self.dropout3 = nn.Dropout(p=dropout_probability)

        self.linear1 = nn.Linear(in_features=news_embed_dim * 4, out_features=num_filters)
        self.linear2 = nn.Linear(in_features=news_embed_dim * 2, out_features=user_vector_dim)
        self.linear3 = nn.Linear(
            in_features=num_filters + user_vector_dim, out_features=user_vector_dim
        )

        self.dense_att = DenseAttention(
            input_dim=user_vector_dim * 2,
            hidden_dim1=dense_att_hidden_dim1,
            hidden_dim2=dense_att_hidden_dim2,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=user_vector_dim, num_heads=num_heads
        )

    def forward(
        self, hist_news_vector: torch.Tensor, cand_news_vector: torch.Tensor
    ) -> torch.Tensor:
        cand_news_vector = self.dropout1(cand_news_vector)
        hist_news_vector = self.dropout2(hist_news_vector)

        repeated_cand_news_vector = cand_news_vector.unsqueeze(dim=1).repeat(
            1, hist_news_vector.shape[1], 1
        )

        # candi-cnn
        hist_news_left = torch.cat(
            [hist_news_vector[:, -1:, :], hist_news_vector[:, :-1, :]], dim=-2
        )
        hist_news_right = torch.cat(
            [hist_news_vector[:, 1:, :], hist_news_vector[:, :1, :]], dim=-2
        )
        hist_news_cnn = torch.cat(
            [hist_news_left, hist_news_vector, hist_news_right, repeated_cand_news_vector], dim=-1
        )

        hist_news_cnn = self.linear1(hist_news_cnn)

        # candi-selfatt
        hist_news = torch.cat([repeated_cand_news_vector, hist_news_vector], dim=-1)
        hist_news = self.linear2(hist_news)
        hist_news_self, _ = self.multihead_attention(hist_news, hist_news, hist_news)

        hist_news_all = torch.cat([hist_news_cnn, hist_news_self], dim=-1)
        hist_news_all = self.dropout3(hist_news_all)
        hist_news_all = self.linear3(hist_news_all)

        # candi-att
        attention_vector = torch.cat([hist_news_all, repeated_cand_news_vector], dim=-1)
        attention_score = self.dense_att(attention_vector)
        attention_score = attention_score.squeeze(dim=-1)
        attention_score = F.softmax(attention_score, dim=-1)

        user_vector = torch.bmm(attention_score.unsqueeze(dim=1), hist_news_all).squeeze(dim=1)

        scores = torch.bmm(
            cand_news_vector.unsqueeze(dim=1), user_vector.unsqueeze(dim=-1)
        ).flatten()

        return scores
