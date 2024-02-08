from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from newsreclib.data.components.batch import RecommendationBatch
from newsreclib.data.components.mind_dataframe import MINDDataFrame


class RecommendationDatasetTrain(MINDDataFrame):
    """
    Attributes:
        news:
            A dataframe containing news with various features (e.g., title, category)
        behaviors:
            A dataframe of user click behaviors.
        max_history_len:
            Maximum history length.
        neg_sampling_ratio:
            The number of negatives to positives to sample for training.
    """

    def __init__(
        self,
        news: pd.DataFrame,
        behaviors: pd.DataFrame,
        max_history_len: int,
        neg_sampling_ratio: float,
    ) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_len = max_history_len
        self.neg_sampling_ratio = neg_sampling_ratio

    def __getitem__(self, index: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[index]

        user = np.array([int(bhv["user"])])
        history = np.array(bhv["history"])[: self.max_history_len]
        candidates = np.array(bhv["candidates"])
        labels = np.array(bhv["labels"])

        candidates, labels = self._sample_candidates(candidates, labels)

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return user, history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def _sample_candidates(
        self, candidates: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Negative sampling of news candidates.

        Args:
            candidates (np.array): candidates news.
            labels (np.array): Labels of candidates news.

        Returns:
            - List: Sampled candidates.
            - np.array: Bool labels of sampled candidates (e.g. True if candidatesidate was clicked, False otherwise)
        """
        pos_ids = np.where(labels == 1)[0]
        neg_ids = np.array([]).astype(int)

        # sample with replacement if the candidates set is smaller than the negative sampling ratio
        replace_flag = (
            True
            if (self.neg_sampling_ratio * len(pos_ids) > len(labels) - len(pos_ids))
            else False
        )

        # negative sampling
        neg_ids = np.random.choice(
            np.random.permutation(np.where(labels == 0)[0]),
            self.neg_sampling_ratio * len(pos_ids),
            replace=replace_flag,
        )

        indices = np.concatenate((pos_ids, neg_ids))
        indices = np.random.permutation(indices)
        candidates = candidates[indices]
        labels = labels[indices]

        return candidates, labels


class RecommendationDatasetTest(MINDDataFrame):
    def __init__(self, news: pd.DataFrame, behaviors: pd.DataFrame, max_history_len: int) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_len = max_history_len

    def __getitem__(self, idx: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[idx]

        user = np.array([int(bhv["user"])])
        history = np.array(bhv["history"])[: self.max_history_len]
        candidates = np.array(bhv["candidates"])
        labels = np.array(bhv["labels"])

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return user, history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)


@dataclass
class DatasetCollate:
    def __init__(
        self,
        dataset_attributes: List[str],
        use_plm: bool,
        tokenizer: Optional[PreTrainedTokenizer],
        max_title_len: int,
        max_abstract_len: int,
        concatenate_inputs: bool,
    ) -> None:
        self.dataset_attributes = dataset_attributes
        self.use_plm = use_plm
        self.max_title_len = max_title_len

        if "abstract" in self.dataset_attributes:
            assert isinstance(max_abstract_len, int) and max_abstract_len > 0
            self.max_abstract_len = max_abstract_len

        if self.use_plm:
            self.tokenizer = tokenizer

        self.concatenate_inputs = concatenate_inputs

    def __call__(self, batch) -> RecommendationBatch:
        users, histories, candidates, labels = zip(*batch)

        batch_hist = self._make_batch_asignees(histories)
        batch_cand = self._make_batch_asignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories))
        x_cand = self._tokenize_df(pd.concat(candidates))
        labels = torch.from_numpy(np.concatenate(labels)).float()
        users = torch.from_numpy(np.concatenate(users)).long()

        return RecommendationBatch(
            batch_hist=batch_hist,
            batch_cand=batch_cand,
            x_hist=x_hist,
            x_cand=x_cand,
            labels=labels,
            users=users,
        )

    def _tokenize_embeddings(self, text: List[List[int]], max_len: Optional[int]) -> torch.Tensor:
        if max_len is None:
            max_len = max([len(item) for item in text])

        text_padded = [
            F.pad(torch.tensor(item), (0, max_len - len(item)), "constant", 0) for item in text
        ]

        return torch.vstack(text_padded).long()

    def _tokenize_plm(self, text: List[str]):
        return self.tokenizer(
            text, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        batch_out = {}

        if not self.concatenate_inputs:
            # prepare text
            if not self.use_plm:
                title = self._tokenize_embeddings(
                    df["tokenized_title"].values.tolist(), max_len=self.max_title_len
                )
                if "abstract" in self.dataset_attributes:
                    abstract = self._tokenize_embeddings(
                        df["tokenized_abstract"].values.tolist(), max_len=self.max_abstract_len
                    )
            else:
                title = self._tokenize_plm(df["title"].values.tolist())
                if "abstract" in self.dataset_attributes:
                    abstract = self._tokenize_plm(df["abstract"].values.tolist())

            batch_out["title"] = title

            if "abstract" in self.dataset_attributes:
                batch_out["abstract"] = abstract

            if "title_entities" in self.dataset_attributes:
                # prepare entities
                title_entities = self._tokenize_embeddings(
                    df["title_entities"].values.tolist(), max_len=self.max_title_len
                )
                batch_out["title_entities"] = title_entities

            if "abstract_entities" in self.dataset_attributes:
                abstract_entities = self._tokenize_embeddings(
                    df["abstract_entities"].values.tolist(), max_len=self.max_abstract_len
                )
                batch_out["abstract_entities"] = abstract_entities

        else:
            # prepare text
            if not self.use_plm:
                if "abstract" in self.dataset_attributes:
                    text = [
                        [*l1, *l2]
                        for (l1, l2) in list(
                            zip(
                                df["tokenized_title"].values.tolist(),
                                df["tokenized_abstract"].values.tolist(),
                            )
                        )
                    ]
                    text = self._tokenize_embeddings(
                        text, max_len=self.max_title_len + self.max_abstract_len
                    )
                else:
                    text = self._tokenize_embeddings(
                        df["tokenized_title"].values.tolist(), max_len=self.max_title_len
                    )
            else:
                if "abstract" in self.dataset_attributes:
                    text = self._tokenize_plm(df[["title", "abstract"]].values.tolist())
                else:
                    text = self._tokenize_plm(df["title"].values.tolist())
            batch_out["text"] = text

            if "title_entities" and "abstract_entities" in self.dataset_attributes:
                if "abstract_entities" in self.dataset_attributes:
                    # prepare entities
                    entities = [
                        [*l1, *l2]
                        for (l1, l2) in list(
                            zip(
                                df["title_entities"].values.tolist(),
                                df["abstract_entities"].values.tolist(),
                            )
                        )
                    ]
                    entities = self._tokenize_embeddings(entities, max_len=None)
                else:
                    entities = self._tokenize_embeddings(
                        df["title_entities"].values.tolist(), max_len=self.max_title_len
                    )
                batch_out["entities"] = entities

        # prepare other aspects
        category = torch.from_numpy(df["category_class"].values).long()
        subcategory = torch.from_numpy(df["subcategory_class"].values).long()

        batch_out["category"] = category
        batch_out["subcategory"] = subcategory

        if ("sentiment_class" in self.dataset_attributes) or (
            "sentiment_score" in self.dataset_attributes
        ):
            sentiment = torch.from_numpy(df["sentiment_class"].values).long()
            sentiment_score = torch.from_numpy(df["sentiment_score"].values).float()

            batch_out["sentiment"] = sentiment
            batch_out["sentiment_score"] = sentiment_score

        return batch_out

    def _make_batch_asignees(self, items: Sequence[Sequence[Any]]) -> torch.Tensor:
        sizes = torch.tensor([len(x) for x in items])
        batch = torch.repeat_interleave(torch.arange(len(items)), sizes)

        return batch
