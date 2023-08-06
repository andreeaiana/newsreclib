import torch
import torch.nn as nn

from newsreclib.models.components.layers.attention import AdditiveAttention


class UserEncoder(nn.Module):
    """Implements the user encoder of NAML.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "Neural news recommendation with attentive multi-view learning." arXiv preprint arXiv:1907.05576 (2019).

    For further details, please refer to the `paper <https://www.ijcai.org/proceedings/2019/0536.pdf>`_

    Attributes:
        news_embed_dim:
            The number of features in the user vector.
        query_dim:
            The number of features in the query vector.
    """

    def __init__(self, news_embed_dim: int, query_dim: int) -> None:
        super().__init__()

        # initialize
        self.additive_attention = AdditiveAttention(input_dim=news_embed_dim, query_dim=query_dim)

    def forward(self, hist_news_vector: torch.Tensor) -> torch.Tensor:
        # batch_size, num_clicked_news, news_embed_dim -> batch_size, news_embed_dim
        user_vector = self.additive_attention(hist_news_vector)

        return user_vector
