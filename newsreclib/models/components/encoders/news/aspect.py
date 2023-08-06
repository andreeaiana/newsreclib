import torch
import torch.nn as nn


class SentimentEncoder(nn.Module):
    """Implements the sentiment encoder from SentiDebias.

    Reference: Wu, Chuhan, Fangzhao Wu, Tao Qi, Wei-Qiang Zhang, Xing Xie, and Yongfeng Huang. "Removing AI’s sentiment manipulation of personalized news delivery." Humanities and Social Sciences Communications 9, no. 1 (2022): 1-9.

    For further details, please refer to the `paper <https://www.nature.com/articles/s41599-022-01473-1>`_

    Attributes:
        num_sent_classes:
            Number of sentiment classes.
        sent_embed_dim:
            Number of features in the sentiment embedding.
        sent_output_dim:
            Number of output features in the linear layer (equivalent to the final dimensionality of the sentiment vector).
    """

    def __init__(self, num_sent_classes: int, sent_embed_dim: int, sent_output_dim: int) -> None:
        super().__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=num_sent_classes + 1, embedding_dim=sent_embed_dim, padding_idx=0
        )
        self.linear = nn.Linear(in_features=sent_embed_dim, out_features=sent_output_dim)

    def forward(self, sentiment) -> torch.Tensor:
        return torch.tanh(self.linear(self.embedding_layer(sentiment)))
