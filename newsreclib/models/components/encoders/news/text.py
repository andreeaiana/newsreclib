from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from newsreclib.models.components.layers.attention import (
    AdditiveAttention,
    PersonalizedAttention,
)
from newsreclib.models.components.layers.projection import UserPreferenceQueryProjection


class PLM(nn.Module):
    """Implements a text encoder based on a pretrained language model.

    Attributes:
        plm_model:
            Name of the pretrained language model.
        frozen_layers:
            List of layers to freeze during training.
        embed_dim:
            Number of features in the text vector.
        use_mhsa:
            If ``True``, it aggregates the token embeddings with a multi-head self-attention network into a final text representation. If ``False``, it uses the `CLS` embedding as the final text representation.
        apply_reduce_dim:
            Whether to linearly reduce the dimensionality of the news vector.
        reduced_embed_dim:
            The number of features in the reduced news vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        plm_model,
        frozen_layers: Optional[List[int]],
        embed_dim: int,
        use_mhsa: bool,
        apply_reduce_dim: bool,
        reduced_embed_dim: Optional[int],
        num_heads: Optional[int],
        query_dim: Optional[int],
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(plm_model, str):
            raise ValueError(
                f"Expected keyword argument `plm_model` to be a `str` but got {plm_model}"
            )

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        self.use_mhsa = use_mhsa
        self.apply_reduce_dim = apply_reduce_dim

        # initialize
        self.plm_model = AutoModel.from_pretrained(plm_model)

        # freeze PLM layers
        for name, param in self.plm_model.base_model.named_parameters():
            for layer in frozen_layers:
                if "layer." + str(layer) + "." in name:
                    param.requires_grad = False

        if self.use_mhsa:
            # PLM-NR proposed in "Empowering News Recommendation with Pre-trained Language Models"
            assert isinstance(num_heads, int) and num_heads > 0
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads
            )
            self.additive_attention = AdditiveAttention(input_dim=embed_dim, query_dim=query_dim)
            self.dropout = nn.Dropout(p=dropout_probability)

        if self.apply_reduce_dim:
            assert isinstance(reduced_embed_dim, int) and reduced_embed_dim > 0
            self.reduce_dim = nn.Linear(in_features=embed_dim, out_features=reduced_embed_dim)
            self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        if self.use_mhsa:
            # batch_size, num_words_text
            text_vector = self.plm_model(**text)[0]
            text_vector = self.dropout(text_vector)

            # batch_size, num_words, embed_dim
            text_vector, _ = self.multihead_attention(text_vector, text_vector, text_vector)
            text_vector = self.dropout(text_vector)

            # batch_size, embed_dim
            text_vector = self.additive_attention(text_vector)

        else:
            text_vector = self.plm_model(**text).last_hidden_state[:, 0, :]

        if self.apply_reduce_dim:
            text_vector = self.reduce_dim(text_vector)
            text_vector = self.dropout(text_vector)

        return text_vector


class CNNAddAtt(nn.Module):
    """Implements a text encoder based on CNN and additive attention.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "Neural news recommendation with attentive multi-view learning." arXiv preprint arXiv:1907.05576 (2019).

    For further details, please refer to the `paper <https://www.ijcai.org/proceedings/2019/0536.pdf>`_

    Attributes:
        pretrained_embeddings:
            Matrix of pretrained embeddings.
        embed_dim:
            The number of features in the text vector.
        num_filters:
            The number of filters in the ``CNN``.
        window_size:
            The window size in the ``CNN``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        pretrained_embeddings: torch.Tensor,
        embed_dim: int,
        num_filters: int,
        window_size: int,
        query_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings), freeze=False, padding_idx=0
        )
        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(window_size, embed_dim),
            padding=(int((window_size - 1) / 2), 0),
        )
        self.additive_attention = AdditiveAttention(input_dim=num_filters, query_dim=query_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, embed_dim
        text_vector = self.embedding_layer(text)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters, num_words_text
        text_vector = self.cnn(text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters
        text_vector = self.additive_attention(text_vector.transpose(1, 2))

        return text_vector


class MHSAAddAtt(nn.Module):
    """Implements a text encoder based on multi-head self-attention and additive attention.

    Reference: Wu, Chuhan, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. "Neural news recommendation with multi-head self-attention." In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 6389-6394. 2019.

    For further details, please refer to the `paper <https://aclanthology.org/D19-1671/>`_

    Attributes:
        pretrained_embeddings:
            Matrix of pretrained embeddings.
        embed_dim:
            The number of features in the text vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        pretrained_embeddings: torch.Tensor,
        embed_dim: int,
        num_heads: int,
        query_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings), freeze=False, padding_idx=0
        )
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.additive_attention = AdditiveAttention(input_dim=embed_dim, query_dim=query_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, embed_dim
        text_vector = self.embedding_layer(text)
        text_vector = self.dropout(text_vector)

        # num_words_text, batch_size, embed_dim
        text_vector = text_vector.permute(1, 0, 2)
        text_vector, _ = self.multihead_attention(text_vector, text_vector, text_vector)
        text_vector = self.dropout(text_vector)

        # batch_size, embed_dim
        text_vector = text_vector.permute(1, 0, 2)
        text_vector = self.additive_attention(text_vector)

        return text_vector


class CNNMHSAAddAtt(nn.Module):
    """Implements a text encoder based on CNN, multi-head self-attention, and additive attention.

    Reference: Qi, Tao, Fangzhao Wu, Chuhan Wu, Yongfeng Huang, and Xing Xie. "Privacy-Preserving News Recommendation Model Learning." In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1423-1432. 2020.

    For further details, please refer to the `paper <https://aclanthology.org/2020.findings-emnlp.128/>`_

    Attributes:
        pretrained_embeddings:
            Matrix of pretrained embeddings.
        num_filters:
            The number of filters in the ``CNN``.
        window_size:
            The window size in the ``CNN``.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        pretrained_embeddings: torch.Tensor,
        embed_dim: int,
        num_filters: int,
        window_size: int,
        num_heads: int,
        query_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings), freeze=False, padding_idx=0
        )
        self.cnn = nn.Conv1d(
            in_channels=embed_dim, out_channels=num_filters, kernel_size=window_size, padding=1
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=num_filters, num_heads=num_heads
        )
        self.additive_attention = AdditiveAttention(input_dim=num_filters, query_dim=query_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, embed_dim
        text_vector = self.embedding_layer(text)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters, num_words_text
        text_vector = self.cnn(text_vector.permute(0, 2, 1))
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)

        # num_words_text, batch_size, embed_dim
        text_vector = text_vector.permute(2, 0, 1)
        text_vector, _ = self.multihead_attention(text_vector, text_vector, text_vector)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters
        text_vector = self.additive_attention(text_vector.transpose(1, 0))

        return text_vector


class CNNPersAtt(nn.Module):
    """Implements a text encoder based on CNN and Personalized Attention.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "NPA: neural news recommendation with personalized attention." In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2576-2584. 2019.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3292500.3330665>`_

    Attributes:
        pretrained_embeddings:
            Matrix of pretrained embeddings.
        text_embed_dim:
            The number of features in the text vector.
        user_embed_dim:
            The number of features in the user vector.
        num_filters:
            The number of filters in the ``CNN``.
        window_size:
            The window size in the ``CNN``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
    """

    def __init__(
        self,
        pretrained_embeddings: torch.Tensor,
        text_embed_dim: int,
        user_embed_dim: int,
        num_filters: int,
        window_size: int,
        query_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings), freeze=False, padding_idx=0
        )
        self.cnn = nn.Conv1d(
            in_channels=text_embed_dim,
            out_channels=num_filters,
            kernel_size=window_size,
            padding=1,
        )
        self.text_query_projection = UserPreferenceQueryProjection(
            user_embed_dim=user_embed_dim,
            preference_query_dim=query_dim,
            dropout_probability=dropout_probability,
        )
        self.personalized_attention = PersonalizedAttention(
            preference_query_dim=query_dim, num_filters=num_filters
        )
        self.dropout = nn.Dropout(dropout_probability)

    def forward(
        self, text: torch.Tensor, lengths: torch.Tensor, projected_users: torch.Tensor
    ) -> torch.Tensor:
        # batch_size, num_words_text, text_embed_dim
        text_vector = self.embedding_layer(text)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters, num_words_text
        text_vector = self.cnn(text_vector.permute(0, 2, 1))
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)

        # batch_size, query_dim
        text_preference_query = self.text_query_projection(projected_users)
        text_preference_query = torch.repeat_interleave(text_preference_query, lengths, dim=0)

        # batch_size, num_filters
        text_vector = self.personalized_attention(text_preference_query, text_vector)

        return text_vector
