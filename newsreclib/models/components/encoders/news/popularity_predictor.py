import torch
import torch.nn as nn


class TimeAwareNewsPopularityPredictor(nn.Module):
    """ Implements the Time Aware News Popularity Predictor from PP-Rec

    PP-Rec: News Recommendation with Personalized User Interest 
    and Time-aware News Popularity

    Paper: https://aclanthology.org/2021.acl-long.424.pdf
    """

    def __init__(self, hidden_size: int, text_embed_dim: int, rec_num_embeddings: int, rec_embedding_dim: int) -> None:
        super(TimeAwareNewsPopularityPredictor, self).__init__()

        # dense layers
        self.dense_recency = nn.Linear(hidden_size, 1)
        self.dense_news = nn.Linear(text_embed_dim, 1)
        self.gate = nn.Linear(hidden_size + text_embed_dim, 1)
        self.recency_encoder = nn.Embedding(
            rec_num_embeddings, rec_embedding_dim)

        # activation function
        self.sigmoid = nn.Sigmoid()

        # trainable weights
        self.w_ctr = nn.Parameter(torch.rand(1))
        self.w_p_hat = nn.Parameter(torch.rand(1))

    def forward(self, news_emb: torch.Tensor, recency: torch.Tensor, ctr: torch.Tensor) -> torch.Tensor:        
        # Compute news and time embeddins
        recency_emb = self.recency_encoder(recency)

        # Compute recency-aware news popularity (p_hat_r)
        p_hat_r = self.dense_recency(recency_emb)

        # Compute content-based news popularity (p_hat_c)
        p_hat_c = self.dense_news(news_emb)

        # Combine the time and news representations
        concat_emb = torch.cat((news_emb, recency_emb), dim=-1)

        # Compute theta
        theta = self.sigmoid(self.gate(concat_emb))

        # Compute news popularity (p_hat) as a weighted sum
        p_hat = theta * p_hat_c + (1 - theta) * p_hat_r
        
        # Final Popularity Score
        score_popularity = self.w_ctr * ctr + self.w_p_hat * p_hat.squeeze(-1)
        
        return score_popularity
