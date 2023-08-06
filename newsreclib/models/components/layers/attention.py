import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, query_dim: int) -> None:
        super().__init__()

        if not isinstance(input_dim, int):
            raise ValueError(
                f"Expected keyword argument `input_dim` to be an `int` but got {input_dim}"
            )

        if not isinstance(query_dim, int):
            raise ValueError(
                f"Expected keyword argument `query_dim` to be an `int` but got {query_dim}"
            )

        # initialize
        self.linear = nn.Linear(in_features=input_dim, out_features=query_dim)
        self.query = nn.Parameter(torch.empty(query_dim).uniform_(-0.1, 0.1))

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_vector:
                User tensor of shape `(batch_size, hidden_dim, output_dim)`.

        Returns:
            User tensor of shape `(batch_size, news_emb_dim)`.
        """
        # batch_size, hidden_dim, output_dim
        attention = torch.tanh(self.linear(input_vector))

        # batch_size, hidden_dim
        attention_weights = F.softmax(torch.matmul(attention, self.query), dim=1)

        # batch_size, output_dim
        weighted_input = torch.bmm(attention_weights.unsqueeze(dim=1), input_vector).squeeze(dim=1)

        return weighted_input


class PolyAttention(nn.Module):
    """Implementation of Poly attention scheme (used in MINER) that extracts K attention vectors
    through K additive attentions.

    Adapted from https://github.com/duynguyen-0203/miner/blob/master/src/model/model.py.

    Reference: Li, Jian, Jieming Zhu, Qiwei Bi, Guohao Cai, Lifeng Shang, Zhenhua Dong, Xin Jiang, and Qun Liu. "MINER: multi-interest matching network for news recommendation." In Findings of the Association for Computational Linguistics: ACL 2022, pp. 343-352. 2022.

    For further details, please refer to the `paper <https://aclanthology.org/2022.findings-acl.29/>`_
    """

    def __init__(self, input_dim: int, num_context_codes: int, context_code_dim: int) -> None:
        """
        Args:
            input_dim:
                The number of expected features in the input.
            num_context_codes:
                The number of attention vectors.
            context_code_dim:
                The number of features in a context code.
        """

        super().__init__()

        if not isinstance(input_dim, int):
            raise ValueError(
                f"Expected keyword argument `input_dim` to be an `int` but got {input_dim}"
            )

        if not isinstance(num_context_codes, int):
            raise ValueError(
                f"Expected keyword argument `num_context_codes` to be an `int` but got {num_context_codes}"
            )

        if not isinstance(context_code_dim, int):
            raise ValueError(
                f"Expected keyword argument `context_code_dim` to be an `int` but got {context_code_dim}"
            )

        # initialize
        self.linear = nn.Linear(in_features=input_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(num_context_codes, context_code_dim),
                gain=nn.init.calculate_gain("tanh"),
            )
        )

    def forward(
        self, embeddings: torch.Tensor, attn_mask: torch.Tensor, bias: torch.Tensor = None
    ):
        """
        Args:
            embeddings:
                `(batch_size, hist_length, embed_dim)`
            attn_mask:
                `(batch_size, hist_length)`
            bias:
                `(batch_size, hist_length, num_candidates)`

        Returns:
            torch.Tensor: `(batch_size, num_context_codes, embed_dim)`
        """
        projection = torch.tanh(self.linear(embeddings))

        if bias is None:
            weights = torch.matmul(projection, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(projection, self.context_codes.T) + bias

        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = F.softmax(weights, dim=2)

        poly_news_vector = torch.matmul(weights, embeddings)

        return poly_news_vector


class TargetAwareAttention(nn.Module):
    """Implementation of the target-aware attention network used in MINER.

    Adapted from https://github.com/duynguyen-0203/miner/blob/master/src/model/model.py

    Reference: Li, Jian, Jieming Zhu, Qiwei Bi, Guohao Cai, Lifeng Shang, Zhenhua Dong, Xin Jiang, and Qun Liu. "MINER: multi-interest matching network for news recommendation." In Findings of the Association for Computational Linguistics: ACL 2022, pp. 343-352. 2022.

    For further details, please refer to the `paper <https://aclanthology.org/2022.findings-acl.29/>`_
    """

    def __init__(self, input_dim: int) -> None:
        """
        Args:
            input_dim:
                The number of features in the query and key vectors.
        """

        super().__init__()

        if not isinstance(input_dim, int):
            raise ValueError(
                f"Expected keyword argument `input_dim` to be an `int` but got {input_dim}"
            )

        # initialize
        self.linear = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:
                `(batch_size, num_context_codes, input_embed_dim)`
            key:
                `(batch_size, num_candidates, input_embed_dim)`
            value:
                `(batch_size, num_candidates, num_context_codes)`
        """
        projection = F.gelu(self.linear(query))
        weights = F.softmax(torch.matmul(key, projection.permute(0, 2, 1)), dim=2)
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs


class DenseAttention(nn.Module):
    """Dense attention used in CAUM.

    Reference: Qi, Tao, Fangzhao Wu, Chuhan Wu, and Yongfeng Huang. "News recommendation with candidate-aware user modeling." In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1917-1921. 2022.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3477495.3531778>`_
    """

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int) -> None:
        super().__init__()

        if not isinstance(input_dim, int):
            raise ValueError(
                f"Expected keyword argument `input_dim` to be an `int` but got {input_dim}"
            )

        if not isinstance(hidden_dim1, int):
            raise ValueError(
                f"Expected keyword argument `hidden_dim1` to be an `int` but got {hidden_dim1}"
            )

        if not isinstance(hidden_dim2, int):
            raise ValueError(
                f"Expected keyword argument `hidden_dim2` to be an `int` but got {hidden_dim2}"
            )

        # initialize
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim2, 1)

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        transformed_vector = self.linear(input_vector)
        transformed_vector = self.tanh1(transformed_vector)
        transformed_vector = self.linear2(transformed_vector)
        transformed_vector = self.tanh2(transformed_vector)
        transformed_vector = self.linear3(transformed_vector)

        return transformed_vector


class PersonalizedAttention(nn.Module):
    """Personalized attention used in NPA.

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "NPA: neural news recommendation with personalized attention." In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2576-2584. 2019.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3292500.3330665>`_
    """

    def __init__(self, preference_query_dim: int, num_filters: int) -> None:
        super().__init__()

        if not isinstance(preference_query_dim, int):
            raise ValueError(
                f"Expected keyword argument `preference_query_dim` to be an `int` but got {preference_query_dim}"
            )

        if not isinstance(num_filters, int):
            raise ValueError(
                f"Expected keyword argument `num_filters` to be an `int` but got {num_filters}"
            )

        # initialize
        self.preference_query_projection = nn.Linear(preference_query_dim, num_filters)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:
                `(batch_size * preference_dim)`
            keys:
                `(batch_size * num_filters * num_words_text)`

        Returns:
            `(batch_size * num_filters)`
        """
        # batch_size * 1 * num_filters
        query = torch.tanh(self.preference_query_projection(query).unsqueeze(dim=1))

        # batch_size * 1 * num_words_text
        attn_results = torch.bmm(query, keys)

        # batch_size * num_words_text * 1
        attn_weights = F.softmax(attn_results, dim=2).permute(0, 2, 1)

        # batch_size * num_filters
        attn_aggr = torch.bmm(keys, attn_weights).squeeze()

        return attn_aggr
