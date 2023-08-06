from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from newsreclib.models.components.layers.attention import AdditiveAttention


class NewsEncoder(nn.Module):
    """Implements a news encoder.

    Attributes:
        dataset_attributes:
            List of news features available in the used dataset.
        attributes2encode:
            List of news features used as input to the news encoder.
        concatenate_inputs:
            Whether the inputs (e.g., title and abstract) were concatenated into a single sequence.
        text_encoder:
            The text encoder module.
        category_encoder:
            The category encoder module.
        entity_encoder:
            The entity encoder module.
        combine_vectors:
            Whether to aggregate the representations of multiple news features.
        combine_type:
            The type of aggregation to use for combining multiple news features representations. Choose between `add_att` (additive attention), `linear`, and `concat` (concatenate).
        input_dim:
            The number of input features in the aggregation layer.
        query_dim:
            The number of features in the query vector.
        output_dim:
            The number of features in the final news vector.
    """

    def __init__(
        self,
        dataset_attributes: List[str],
        attributes2encode: List[str],
        concatenate_inputs: bool,
        text_encoder: Optional[nn.Module],
        category_encoder: Optional[nn.Module],
        entity_encoder: Optional[nn.Module],
        combine_vectors: bool,
        combine_type: Optional[str],
        input_dim: Optional[int],
        query_dim: Optional[int],
        output_dim: Optional[int],
    ) -> None:
        super().__init__()

        # at least one attribute is needed to encode the news
        assert len(dataset_attributes) > 0

        # flags to determine which news attributes are encoded
        self.concatenate_inputs = concatenate_inputs
        self.encode_text = False
        self.encode_category = False
        self.encode_entity = False

        # text encoders
        text_encoder_cand = ["title", "abstract"]
        if ("title" in attributes2encode) or ("abstract" in attributes2encode):
            assert isinstance(text_encoder, nn.Module)
            if not self.concatenate_inputs:
                self.text_encoders = nn.ModuleDict(
                    {
                        name: text_encoder
                        for name in (
                            set(dataset_attributes)
                            & set(attributes2encode)
                            & set(text_encoder_cand)
                        )
                    }
                )
            else:
                self.text_encoders = nn.ModuleDict({"text": text_encoder})
            self.encode_text = True

        # category encoders
        categ_encoder_cand = ["category", "subcategory"]
        if ("category" in attributes2encode) or ("subcategory" in attributes2encode):
            assert isinstance(category_encoder, nn.Module)
            self.category_encoders = nn.ModuleDict(
                {
                    name: category_encoder
                    for name in (
                        set(dataset_attributes) & set(attributes2encode) & set(categ_encoder_cand)
                    )
                }
            )
            self.encode_category = True

        # entity encoders
        entity_encoder_cand = ["title_entities", "abstract_entities"]
        if ("title_entities" in attributes2encode) or ("abstract_entities" in attributes2encode):
            assert isinstance(entity_encoder, nn.Module)
            if not self.concatenate_inputs:
                self.entity_encoders = nn.ModuleDict(
                    {
                        name: entity_encoder
                        for name in (
                            set(dataset_attributes)
                            & set(attributes2encode)
                            & set(entity_encoder_cand)
                        )
                    }
                )
            else:
                self.entity_encoders = nn.ModuleDict({"entities": entity_encoder})
            self.encode_entity = True

        if combine_vectors:
            assert isinstance(combine_type, str)

            self.combine_type = combine_type
            if self.combine_type == "add_att":
                assert isinstance(input_dim, int) and input_dim > 0
                assert isinstance(query_dim, int) and query_dim > 0
                self.combine_layer = AdditiveAttention(input_dim=input_dim, query_dim=query_dim)
            elif self.combine_type == "linear":
                assert isinstance(input_dim, int) and input_dim > 0
                assert isinstance(output_dim, int) and output_dim > 0
                self.combine_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
            elif self.combine_type == "concat":
                self.combine_layer = lambda vectors: torch.cat(vectors, dim=1)
            else:
                raise ValueError(
                    f"Expected keyword argument `combine_type` to be in [`add_att`, `linear`, `concat`] but got {self.combine_type}."
                )

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_vectors = []
        category_vectors = []
        entity_vectors = []

        if self.encode_text:
            text_vectors = [encoder(news[name]) for name, encoder in self.text_encoders.items()]

        if self.encode_category:
            category_vectors = [
                encoder(news[name]) for name, encoder in self.category_encoders.items()
            ]

        if self.encode_entity:
            entity_vectors = [
                encoder(news[name]) for name, encoder in self.entity_encoders.items()
            ]

        if self.encode_category and not self.encode_entity:
            all_vectors = text_vectors + category_vectors
        elif self.encode_entity and not self.encode_category:
            all_vectors = text_vectors + entity_vectors
        else:
            all_vectors = text_vectors + category_vectors + entity_vectors

        if len(all_vectors) == 1:
            news_vector = all_vectors[0]
        else:
            if self.combine_type == "add_att":
                news_vector = self.combine_layer(torch.stack(all_vectors, dim=1))
            elif self.combine_type == "linear":
                text_vectors = text_vectors[0] if len(text_vectors) == 1 else text_vectors
                category_vectors = (
                    category_vectors[0] if len(category_vectors) == 1 else category_vectors
                )
                entity_vectors = entity_vectors[0] if len(entity_vectors) == 1 else entity_vectors

                if self.encode_entity and not self.encode_category:
                    all_vectors = torch.cat([text_vectors, entity_vectors], dim=-1)
                elif self.encode_category and not self.encode_entity:
                    all_vectors = torch.cat([text_vectors, category_vectors], dim=-1)
                else:
                    all_vectors = torch.cat(
                        [text_vectors, category_vectors, entity_vectors], dim=-1
                    )
                news_vector = self.combine_layer(all_vectors)
            else:
                news_vector = self.combine_layer(all_vectors)

        return news_vector


class KCNN(nn.Module):
    """Implements the knowledge-aware CNN from DKN.

    Reference: Wang, Hongwei, Fuzheng Zhang, Xing Xie, and Minyi Guo. "DKN: Deep knowledge-aware network for news recommendation." In Proceedings of the 2018 world wide web conference, pp. 1835-1844. 2018.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186175>`_

    Attributes:
        pretrained_text_embeddings:
            Matrix of pretrained text embeddings.
        pretrained_entity_embeddings:
            Matrix of pretrained entity embeddings.
        pretrained_context_embeddings:
            Matrix of pretrained context embeddings.
        use_context:
            Whether to use context embeddings.
        text_embed_dim:
            The number of features in the text vector.
        entity_embed_dim:
            The number of features in the entity vector.
        num_filters:
            The number of filters in the ``CNN``.
        window_sizes:
            List of window sizes for the ``CNN``.
    """

    def __init__(
        self,
        pretrained_text_embeddings: torch.Tensor,
        pretrained_entity_embeddings: torch.Tensor,
        pretrained_context_embeddings: Optional[torch.Tensor],
        use_context: bool,
        text_embed_dim: int,
        entity_embed_dim: int,
        num_filters: int,
        window_sizes: List[int],
    ) -> None:
        super().__init__()

        # initialize
        self.window_sizes = window_sizes

        self.text_embedding_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_text_embeddings), freeze=False, padding_idx=0
        )

        self.entity_embedding_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_entity_embeddings), freeze=False, padding_idx=0
        )

        self.use_context = use_context
        if self.use_context:
            assert isinstance(pretrained_context_embeddings, torch.Tensor)
            self.context_embedding_layer = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_context_embeddings), freeze=False, padding_idx=0
            )

        self.transform_matrix = nn.Parameter(
            torch.empty(entity_embed_dim, text_embed_dim).uniform_(-0.1, 0.1)
        )
        self.transform_bias = nn.Parameter(torch.empty(text_embed_dim).uniform_(-0.1, 0.1))

        self.conv_filters = nn.ModuleDict(
            {
                str(x): nn.Conv2d(3 if self.use_context else 2, num_filters, (x, text_embed_dim))
                for x in self.window_sizes
            }
        )

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        # batch_size, num_words_text, text_embed_dim
        text_vector = self.text_embedding_layer(news["title"])

        # batch_size, num_words_text, entity_embed_dim
        entity_vector = self.entity_embedding_layer(news["title_entities"])

        # batch_size, num_words_text, text_embed_dim
        transformed_entity_vector = torch.tanh(
            torch.add(torch.matmul(entity_vector, self.transform_matrix), self.transform_bias)
        )

        if self.use_context:
            # batch_size, num_words_text, entity_embed_dim
            context_vector = self.context_embedding_layer(news["title_entities"])

            # batch_size, num_words_text, entity_embed_dim
            transformed_context_vector = torch.tanh(
                torch.add(torch.matmul(context_vector, self.transform_matrix), self.transform_bias)
            )

            # batch_size, 3, num_words_text, text_embedding_layer
            multi_channel_vector = torch.stack(
                [text_vector, transformed_entity_vector, transformed_context_vector], dim=1
            )

        else:
            # batch_size, 2, num_words_text, text_embed_dim
            multi_channel_vector = torch.stack([text_vector, transformed_entity_vector], dim=1)

        pooled_vectors = []
        for size in self.window_sizes:
            # batch_size, num_filters, num_words_text + 1 - size
            convoluted = self.conv_filters[str(size)](multi_channel_vector).squeeze(dim=3)
            activated = F.relu(convoluted)

            # batch_size, num_filters
            pooled = activated.max(dim=-1)[0]

            pooled_vectors.append(pooled)

        # batch_size, len(window_sizes) * num_filters
        news_vector = torch.cat(pooled_vectors, dim=1)

        return news_vector
