"""
Original Source Code: https://github.com/taoqi98/PP-Rec/tree/main
"""

import torch
import torch.nn
from newsreclib.models.components.encoders.news.popularity_predictor import TimeAwareNewsPopularityPredictor
from newsreclib.models.components.layers.attention import AdditiveAttention


class CPJA(nn.Module):
    """Implements the Content-Populairty Joing Attention Network of PP-REC

    PP-Rec: News Recommendation with Personalized User Interest 
    and Time-aware News Popularity

    Paper: https://aclanthology.org/2021.acl-long.424.pdf
    """

    def __init__(self, news_embed_dim: int, query_dim: int) -> None:
        self.additive_attention = AdditiveAttention(
            input_dim=news_embed_dim, query_dim=query_dim)

    def forward(self, news_mhsa_emb: torch.Tensor, pop_emb: torch.Tensor) -> None:
        # Concatenate news embedding and popularity embedding
        news_concat_pop = torch.cat(
            [news_mhsa_emb, pop_emb], dim=-1)

        alpha = self.additive_attention(news_concat_pop)

        return alpha


class PopularityAwareUserEncoder(nn.Module):
    """ Implements the popularity user encoder of PP-Rec

    PP-Rec: News Recommendation with Personalized User Interest 
    and Time-aware News Popularity

    Paper: https://aclanthology.org/2021.acl-long.424.pdf
    """

    def __init__(self, news_embed_dim: int, num_heads: int, query_dim: int, pop_num_embeddings: int, pop_embedding_dim: int) -> None:
        super().__init__()

        # initialize Multi Head Attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=news_embed_dim, num_heads=num_heads)

        # initialize popularity embedding layer
        self.popularity_embedding_layer = nn.Embedding(
            pop_num_embeddings, pop_embedding_dim)

        # initialize content-popularity attention network (CPJA)
        self.cpja = CPJA(news_embed_dim=news_embed_dim, query_dim=query_dim)

    def forward(self, hist_news_vector: torch.Tensor, ctr: torch.Tensor) -> torch.Tensor:
        news_multih_att_emb, _ = self.multihead_attention(
            hist_news_vector, hist_news_vector, hist_news_vector
        )

        # Popularity Embedding without news recency and content to avoid nondifferentiable quantization operation
        popularity_embedding = self.popularity_embedding_layer(ctr)

        # Pass inputs into content-popularity joint attention network (CPJA)
        alpha = self.cpja(news_multih_att_emb, popularity_embedding)

        # Compute user_interest_emb as a weighted sum of news embeddings
        user_interest_emb = torch.sum(alpha * news_multih_att_emb, dim=1)

        return user_interest_emb
