import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class UserEncoder(nn.Module):
    """Implements the user encoder of LSTUR.

    Reference: An, Mingxiao, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu, and Xing Xie. "Neural news recommendation with long-and short-term user representations." In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 336-345. 2019.

    For further details, please refer to the `paper <https://aclanthology.org/P19-1033/>`_

    Attributes:
        num_users:
            The number of users.
        input_dim:
            The number of input features in the embeddng layer for the long-term user representation.
        user_masking_probability:
            The probability for randomly masking users in the long-term user representation.
        long_short_term_method:
            The method for combining long and short-term user representations. If ``ini`` is chosen, the  `GRU` will be initialized with the long-term user representation. If ``con`` is chosen, the long and short-term user representations will be concatenated.
    """

    def __init__(
        self,
        num_users: int,
        input_dim: int,
        user_masking_probability: float,
        long_short_term_method: str,
    ) -> None:
        super().__init__()

        if not isinstance(num_users, int):
            raise ValueError(
                f"Expected keyword argument `num_users` to be an `int` but got {num_users}"
            )

        if not isinstance(input_dim, int):
            raise ValueError(
                f"Expected keyword argument `input_dim` to be an `int` but got {input_dim}"
            )

        if not isinstance(user_masking_probability, float):
            raise ValueError(
                f"Expected keyword argument `user_masking_probability` to be a `float` but got {user_masking_probability}"
            )

        if not isinstance(long_short_term_method, str):
            raise ValueError(
                f"Expected keyword argument `long_short_term_method` to be a `str` but got {long_short_term_method}"
            )
        assert long_short_term_method in ["ini", "con"]
        self.long_short_term_method = long_short_term_method

        self.long_term_user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=input_dim
            if self.long_short_term_method == "ini"
            else int(input_dim * 0.5),
            padding_idx=0,
        )
        self.dropout = nn.Dropout2d(p=user_masking_probability)
        self.gru = nn.GRU(
            input_dim, input_dim if self.long_short_term_method == "ini" else int(input_dim * 0.5)
        )

    def forward(
        self, user: torch.Tensor, hist_news_vector: torch.Tensor, hist_size: torch.Tensor
    ) -> torch.Tensor:
        # long-term user representation
        user_vector = self.long_term_user_embedding(user).unsqueeze(dim=0)
        user_vector = self.dropout(user_vector)

        # short-term user representation
        packed_hist_news_vector = pack_padded_sequence(
            input=hist_news_vector,
            lengths=hist_size.cpu().int(),
            batch_first=True,
            enforce_sorted=False,
        )
        if self.long_short_term_method == "ini":
            _, last_hidden = self.gru(packed_hist_news_vector, user_vector)
            return last_hidden.squeeze(dim=0)
        else:
            _, last_hidden = self.gru(packed_hist_news_vector)
            return torch.cat((last_hidden.squeeze(dim=0), user_vector.squeeze(dim=0)), dim=1)
