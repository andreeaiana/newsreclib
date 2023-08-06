# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/DKN/KCNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserEncoder(nn.Module):
    """Implements the user encoder of DKN.

    Reference: Wang, Hongwei, Fuzheng Zhang, Xing Xie, and Minyi Guo. "DKN: Deep knowledge-aware network for news recommendation." In Proceedings of the 2018 world wide web conference, pp. 1835-1844. 2018.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186175>`_

    Attributes:
        input_dim:
            The number of input features to the user encoder.
        hidden_dim:
            The number of features in the hidden state of the user encoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int):
            raise ValueError(
                f"Expected keyword argument `input_dim` to be an `int` but got {input_dim}"
            )

        if not isinstance(hidden_dim, int):
            raise ValueError(
                f"Expected keyword argument `hidden_dim` to be an `int` but got {hidden_dim}"
            )

        # initialize
        self.dnn = nn.Sequential(
            nn.Linear(in_features=input_dim * 2, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

    def forward(
        self,
        hist_news_vector: torch.Tensor,
        cand_news_vector: torch.Tensor,
        mask_hist: torch.Tensor,
        mask_cand: torch.Tensor,
    ) -> torch.Tensor:
        # num_clicked_news, batch_size, num_cand_news, input_dim
        expanded_cand_news_vector = torch.repeat_interleave(
            input=cand_news_vector.unsqueeze(dim=0), repeats=hist_news_vector.shape[1], dim=0
        )

        # batch_size, num_cand_news, num_clicked_news, input_dim
        expanded_cand_news_vector = expanded_cand_news_vector.permute(1, 2, 0, 3)

        # batch_size, num_cand_news, num_clicked_new, input_dim
        repeated_hist_news_vector = hist_news_vector.unsqueeze(1).repeat(
            1, cand_news_vector.shape[1], 1, 1
        )

        # batch_size, num_cand_news, num_clicked_news, input_dim * 2
        concatenated_news_vector = torch.cat(
            [expanded_cand_news_vector, repeated_hist_news_vector], dim=-1
        )

        # batch_size, num_cand_news, num_clicked_news
        transformed_news_vector = self.dnn(concatenated_news_vector).squeeze(dim=-1)

        # num_clicked_news, batch_size, num_cand_news
        repeated_mask_cand = torch.repeat_interleave(
            input=mask_cand.unsqueeze(0), repeats=mask_hist.shape[1], dim=0
        )

        # batch_size, num_cand_news, num_clicked_news
        repeated_mask_cand = repeated_mask_cand.permute(1, 2, 0)

        # batch_size, num_cand_news, num_clicked_news
        repeated_mask_hist = mask_hist.unsqueeze(dim=1).repeat(1, mask_cand.shape[1], 1)

        # softmax only for relevant clicked news for each user
        masked_transformed_news_vector = torch.where(
            ~repeated_mask_hist,
            torch.tensor(
                torch.finfo(transformed_news_vector.dtype).min,
                device=transformed_news_vector.device,
            ),
            transformed_news_vector,
        )

        # batch_size, num_candidate_news, num_clicked_news
        hist_news_weights = F.softmax(masked_transformed_news_vector, dim=-1)

        # weights only for relevant candidate news for each user
        masked_hist_news_weights = torch.where(
            ~repeated_mask_cand,
            torch.tensor(0.0, device=hist_news_weights.device),
            hist_news_weights,
        )

        # batch_size, num_cand_news_user, input_dim
        user_vector = torch.bmm(masked_hist_news_weights, hist_news_vector)

        return user_vector
