import torch


def pairwise_cosine_similarity(
    x: torch.Tensor, y: torch.Tensor, zero_diagonal: bool = False
) -> torch.Tensor:
    """
    Implemented from https://github.com/duynguyen-0203/miner/blob/master/src/utils.py

    Calculates the pairwise cosine similarity matrix.

    Args:
        x:
            (batch_size, M, d)
        y:
            (batch_size, N, d)
        zero_diagonal:
            Determines if the diagonal of the distance matrix should be set to zero.

    Returns:
        A single-value tensor with the pairwise cosine similarity between ``x`` and ``y``.
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)

    x_norm[torch.where(x_norm) == 0.0] = 1
    y_norm[torch.where(y_norm) == 0.0] = 1

    x = torch.div(x, 1e-8 + x_norm)
    y = torch.div(y, 1e-8 + y_norm)

    distance = torch.matmul(x, y.permute(0, 2, 1))

    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)

    return distance
