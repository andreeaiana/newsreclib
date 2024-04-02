"""
Original Source Code: https://github.com/taoqi98/PP-Rec/tree/main
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class CPJA(nn.Module):
    """Implements the Content-Populairty Joing Attention Network of PP-REC

    PP-Rec: News Recommendation with Personalized User Interest 
    and Time-aware News Popularity

    Paper: https://aclanthology.org/2021.acl-long.424.pdf
    """

    def __init__(self, text_embed_dim: int, pop_embedding_dim: int, cpja_hidden_dim: int) -> None:
        super().__init__()

        self.Wu = nn.Linear(text_embed_dim + pop_embedding_dim, cpja_hidden_dim)  # Linear transformation
        self.q = nn.Parameter(torch.randn(cpja_hidden_dim, 1))  # Trainable parameter q

    def forward(self, news_mhsa_emb: torch.Tensor, pop_emb: torch.Tensor) -> None:
        # Concatenate news embedding and popularity embedding along the last dimension
        news_concat_pop = torch.cat(
            [news_mhsa_emb, pop_emb], dim=-1)

        # Linear transformation followed by tanh
        transformed = torch.tanh(self.Wu(news_concat_pop))

        # Compute attention scores by multiplying with q (and squeezing to remove last dim)
        scores = torch.matmul(transformed, self.q).squeeze(-1)

        # Apply softmax to normalize scores across num_news dimension
        alpha = F.softmax(scores, dim=-1)

        return alpha


class PopularityAwareUserEncoder(nn.Module):
    """ Implements the popularity aware user encoder of PP-Rec

    PP-Rec: News Recommendation with Personalized User Interest 
    and Time-aware News Popularity

    Paper: https://aclanthology.org/2021.acl-long.424.pdf
    """

    def __init__(self, text_embed_dim: int, text_num_heads: int, cpja_hidden_dim: int, pop_num_embeddings: int, pop_embedding_dim: int) -> None:
        super().__init__()

        # initialize Multi Head Attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=text_embed_dim, num_heads=text_num_heads)

        # initialize popularity embedding layer
        self.popularity_embedding_layer = nn.Embedding(
            pop_num_embeddings, pop_embedding_dim)

        # initialize content-popularity attention network (CPJA)
        self.cpja = CPJA(
            text_embed_dim=text_embed_dim, pop_embedding_dim=pop_embedding_dim, cpja_hidden_dim=cpja_hidden_dim
        )

    def forward(self, hist_news_vector: Dict[str, torch.Tensor], ctr: torch.Tensor) -> torch.Tensor:
        news_multih_att_emb, _ = self.multihead_attention(
            hist_news_vector, hist_news_vector, hist_news_vector
        )

        # Popularity Embedding without news recency and content to avoid nondifferentiable quantization operation
        popularity_embedding = self.popularity_embedding_layer(ctr)

        # Pass inputs into content-popularity joint attention network (CPJA)
        alpha = self.cpja(news_multih_att_emb, popularity_embedding)

        # Compute user_interest_emb as a weighted sum of news embeddings
        user_interest_emb = torch.sum(alpha.unsqueeze(-1) * news_multih_att_emb, dim=1)

        return user_interest_emb
