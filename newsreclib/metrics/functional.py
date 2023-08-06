from typing import Optional

import torch
import torch.nn.functional as F
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def diversity(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, top_k: Optional[int] = None
) -> torch.Tensor:
    """Computes `Aspect-based Diversity`.

    Reference: Iana, Andreea, Goran Glavaš, and Heiko Paulheim.
    "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation."
    arXiv preprint arXiv:2307.16089 (2023).
    `https://arxiv.org/pdf/2307.16089.pdf`

    Args:
        preds:
            Estimated probabilities of each candidate news to be clicked.
        target:
            Ground truth about the aspect :math:`A_p` of the news being relevant or not.
        num_classes:
            Number of classes of the aspect :math:`A_p`.
        top_k:
            Consider only the top k elements for each query (default: ``None``, which considers them all).

    Returns:
        A single-value tensor with the aspect-based diversity (:math:`D_{A_p}`) of the predictions ``preds`` wrt the labels ``target``.
    """
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    top_k = preds.shape[-1] if top_k is None else top_k

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    sorted_target = target[torch.argsort(preds, dim=-1, descending=True)][:top_k]
    target_count = torch.bincount(sorted_target)
    padded_target_count = F.pad(target_count, pad=(0, num_classes - target_count.shape[0]))
    target_prob = padded_target_count / padded_target_count.shape[0]
    target_dist = torch.distributions.Categorical(target_prob)

    diversity = torch.div(
        target_dist.entropy(), torch.log(torch.tensor(num_classes, device=preds.device))
    )

    return diversity


def personalization(
    preds: torch.Tensor,
    predicted_aspects: torch.Tensor,
    target_aspects: torch.Tensor,
    num_classes: int,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Computes `Aspect-based Personalization`.

    Reference: Iana, Andreea, Goran Glavaš, and Heiko Paulheim.
    "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation."
    arXiv preprint arXiv:2307.16089 (2023).
    `https://arxiv.org/pdf/2307.16089.pdf`

    Args:
        preds:
            Estimated probabilities of each candidate news to be clicked.
        predicted_aspects:
            Aspects of the news predicted to be clicked.
        target_aspects:
            Ground truth about the aspect :math:`A_p` of the news being relevant or not.
        num_classes:
            Number of classes of the aspect :math:`A_p`.
        top_k:
            Consider only the top k elements for each query (default: ``None``, which considers them all).

    Returns:
        A single-value tensor with the aspect-based personalization (:math:`PS_{A_p}`) of the predictions ``preds`` and ``predicted_aspects`` wrt the labels ``target_aspects``.
    """
    preds, predicted_aspects = _check_retrieval_functional_inputs(
        preds, predicted_aspects, allow_non_binary_target=True
    )

    top_k = preds.shape[-1] if top_k is None else top_k

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    sorted_predicted_aspects = predicted_aspects[torch.argsort(preds, dim=-1, descending=True)][
        :top_k
    ]
    predicted_aspects_count = torch.bincount(sorted_predicted_aspects)
    padded_predicted_aspects_count = F.pad(
        predicted_aspects_count, pad=(0, num_classes - predicted_aspects_count.shape[0])
    )

    target_aspects_count = torch.bincount(target_aspects)
    padded_target_aspects_count = F.pad(
        target_aspects_count, pad=(0, num_classes - target_aspects_count.shape[0])
    )

    personalization = generalized_jaccard(
        padded_predicted_aspects_count, padded_target_aspects_count
    )

    return personalization


def generalized_jaccard(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the Generalized Jaccard metric.

    Reference: Bonnici, Vincenzo. "Kullback-Leibler divergence between quantum distributions, and its upper-bound." arXiv preprint arXiv:2008.05932 (2020).

    Args:
        preds:
            Estimated probability distribution.
        target:
            Target probability distribution.

    Returns:
        A single-value tensor with the generalized Jaccard of the predictions ``preds`` wrt the labels ``target``.
    """
    assert pred.shape == target.shape

    jaccard = torch.min(pred, target).sum(dim=0) / torch.max(pred, target).sum(dim=0)

    return jaccard


def harmonic_mean(scores: torch.Tensor) -> torch.Tensor:
    """Computes the harmonic mean of `N` scores.

    Args:
        scores:
            A tensor of scores.

    Returns:
        A single-value tensor with the harmonic mean of the scores.
    """

    weights = torch.ones(scores.shape, device=scores.device)
    harmonic_mean = torch.div(torch.sum(weights), torch.sum(torch.div(weights, scores)))

    return harmonic_mean
