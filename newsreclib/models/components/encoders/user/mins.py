import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from newsreclib.models.components.layers.attention import AdditiveAttention


class UserEncoder(nn.Module):
    """Implements the user encoder of MINS.

    Reference: Wang, Rongyao, Shoujin Wang, Wenpeng Lu, and Xueping Peng. "News recommendation via multi-interest news sequence modelling." In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 7942-7946. IEEE, 2022.

    For further details, please refer to the `paper <https://ieeexplore.ieee.org/abstract/document/9747149/>`_

    Attributes:
        news_embed_dim:
            The number of features in the news vector.
        query_dim:
            The number of features in the query vector.
        num_filters:
            The number of filters used in the `GRU`.
        num_gru_channels:
            The number of channels used in the `GRU`.
    """

    def __init__(
        self,
        news_embed_dim: int,
        query_dim: int,
        num_filters: int,
        num_gru_channels: int,
    ) -> None:
        super().__init__()

        if not isinstance(num_gru_channels, int):
            raise ValueError(
                f"Expected keyword argument `num_gru_channels` to be an `int` but got {num_gru_channels}"
            )
        assert num_filters % num_gru_channels == 0

        # initialize
        self.num_gru_channels = num_gru_channels

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=news_embed_dim, num_heads=num_gru_channels
        )
        self.additive_attention = AdditiveAttention(input_dim=news_embed_dim, query_dim=query_dim)

        self.gru = nn.GRU(int(num_filters / num_gru_channels), int(num_filters / num_gru_channels))
        self.multi_channel_gru = nn.ModuleList([self.gru for _ in range(num_gru_channels)])

    def forward(self, hist_news_vector: torch.Tensor, hist_size: torch.Tensor) -> torch.Tensor:
        # batch_size, hist_size, news_embed_dim
        multihead_user_vector, _ = self.multihead_attention(
            hist_news_vector, hist_news_vector, hist_news_vector
        )

        # batch_size, hist_size, news_embed_dim / num_gru_channels
        user_vector_channels = torch.chunk(
            input=multihead_user_vector, chunks=self.num_gru_channels, dim=2
        )
        channels = []
        for n, gru in zip(range(self.num_gru_channels), self.multi_channel_gru):
            packed_hist_news_vector = pack_padded_sequence(
                input=user_vector_channels[n],
                lengths=hist_size.cpu().int(),
                batch_first=True,
                enforce_sorted=False,
            )

            # 1, batch_size, num_filters / num_gru_channels
            _, last_hidden = gru(packed_hist_news_vector)

            channels.append(last_hidden)

        # batch_size, 1, news_embed_dim
        multi_channel_vector = torch.cat(channels, dim=2).transpose(0, 1)

        # batch_size, news_embed_dim
        user_vector = self.additive_attention(multi_channel_vector)

        return user_vector
