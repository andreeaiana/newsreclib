from typing import Any, Optional

import torch
from torchmetrics.retrieval.base import RetrievalMetric

from newsreclib.metrics.functional import diversity


class Diversity(RetrievalMetric):
    """Implementation of the `Aspect-based Diversity`.

    Reference: Iana, Andreea, Goran Glavaš, and Heiko Paulheim.
    "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation."
    arXiv preprint arXiv:2307.16089 (2023).
    `https://arxiv.org/pdf/2307.16089.pdf`

    For further details, please refer to the `paper <https://arxiv.org/abs/2307.16089>`_
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )

        if (top_k is not None) and not (isinstance(top_k, int) and top_k > 0):
            raise ValueError("`top_k` has to be a positive integer or None")
        self.num_classes = num_classes
        self.top_k = top_k
        self.allow_non_binary_target = True

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return diversity(preds, target, self.num_classes, top_k=self.top_k)
