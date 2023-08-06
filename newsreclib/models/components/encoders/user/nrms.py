import torch
import torch.nn as nn

from newsreclib.models.components.layers.attention import AdditiveAttention


class UserEncoder(nn.Module):
    """Implements the user encoder of NRMS.

    Reference: Wu, Chuhan, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. "Neural news recommendation with multi-head self-attention." In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 6389-6394. 2019.

    For further details, please refer to the `paper <https://aclanthology.org/D19-1671/>`_

    Attributes:
        news_embed_dim:
            The number of features in the news vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
    """

    def __init__(self, news_embed_dim: int, num_heads: int, query_dim: int) -> None:
        super().__init__()

        # initialize
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=news_embed_dim, num_heads=num_heads
        )
        self.additive_attention = AdditiveAttention(input_dim=news_embed_dim, query_dim=query_dim)

    def forward(self, hist_news_vector: torch.Tensor) -> torch.Tensor:
        # batch_size, num_clicked_news_user, news_embed_dim
        user_vector, _ = self.multihead_attention(
            hist_news_vector, hist_news_vector, hist_news_vector
        )

        # batch_size, news_embeding_dim
        user_vector = self.additive_attention(user_vector)

        return user_vector
