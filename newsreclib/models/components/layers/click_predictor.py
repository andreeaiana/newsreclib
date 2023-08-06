import torch
import torch.nn as nn


class DotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, user_vec: torch.Tensor, cand_news_vector: torch.Tensor) -> torch.Tensor:
        predictions = torch.bmm(user_vec, cand_news_vector).squeeze(1)
        return predictions


class DNNPredictor(nn.Module):
    """Implementation of the click pedictor of DKN.

    Reference: Wang, Hongwei, Fuzheng Zhang, Xing Xie, and Minyi Guo. "DKN: Deep knowledge-aware network for news recommendation." In Proceedings of the 2018 world wide web conference, pp. 1835-1844. 2018.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186175>`_
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
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
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_vec: torch.Tensor, cand_news: torch.Tensor) -> torch.Tensor:
        concat_vectors = torch.cat([cand_news.permute(0, 2, 1), user_vec], dim=-1)
        predictions = self.dnn(concat_vectors).squeeze(dim=-1)

        return predictions
