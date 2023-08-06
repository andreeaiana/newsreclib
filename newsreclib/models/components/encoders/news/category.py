from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):
    """Implements a category encoder.

    Attributes:
        pretrained_embeddings:
            Matrix of pretrained embeddings.
        from_pretrained:
            If ``True``, it initializes the category embedding layer with pretrained embeddings. If ``False``, it initializes the category embedding layer with random weights.
        freeze_pretrained_emb:
            If ``True``, it freezes the pretrained embeddings during training. If ``False``, it updates the pretrained embeddings during training.
        num_categories:
            Number of categories.
        embed_dim:
            Number of features in the category vector.
        use_dropout:
            Whether to use dropout after the embedding layer.
        dropout_probability:
            Dropout probability.
        linear_transform:
            Whether to linearly transform the category vector.
        output_dim:
            Number of output features in the category encoder (equivalent to the final dimensionality of the category vector).
    """

    def __init__(
        self,
        pretrained_embeddings: Optional[torch.Tensor],
        from_pretrained: bool,
        freeze_pretrained_emb: bool,
        num_categories: int,
        embed_dim: Optional[int],
        use_dropout: bool,
        dropout_probability: Optional[float],
        linear_transform: bool,
        output_dim: Optional[int],
    ) -> None:
        super().__init__()

        if from_pretrained:
            assert isinstance(pretrained_embeddings, torch.Tensor)

        # initialize
        if from_pretrained:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=pretrained_embeddings, freeze=freeze_pretrained_emb, padding_idx=0
            )
        else:
            assert isinstance(embed_dim, int) and embed_dim > 0
            self.embedding_layer = nn.Embedding(
                num_embeddings=num_categories, embedding_dim=embed_dim, padding_idx=0
            )

        self.use_dropout = use_dropout
        if self.use_dropout:
            if not isinstance(dropout_probability, float):
                raise ValueError(
                    f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
                )
            self.dropout = nn.Dropout(p=dropout_probability)

        self.linear_transform = linear_transform
        if self.linear_transform:
            assert isinstance(output_dim, int)
            self.linear = nn.Linear(in_features=embed_dim, out_features=output_dim)

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        category_vector = self.embedding_layer(category)

        if self.use_dropout:
            category_vector = self.dropout(category_vector)

        if self.linear_transform:
            category_vector = F.relu(self.linear(category_vector))

        return category_vector
