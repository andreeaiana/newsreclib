from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from datetime import datetime, timezone

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
        include_ctr:
            Controling if we should include CTR or not into history/candidates
    """

    def __init__(
        self,
        news: pd.DataFrame,
        behaviors: pd.DataFrame,
        max_history_len: int,
        neg_sampling_ratio: float,
        include_ctr: Optional[bool] = False,
    ) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_len = max_history_len
        self.neg_sampling_ratio = neg_sampling_ratio
        self.include_ctr = include_ctr

    def __getitem__(self, index: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bhv = self.behaviors.iloc[index]

        user_id = np.array([int(bhv["uid"].split("U")[-1])])
        user_idx = np.array([int(bhv["user"])])

        labels = np.array(bhv["labels"])
        if isinstance(bhv["time"], str):
            time = datetime.strptime(bhv["time"], "%Y-%m-%d %H:%M:%S")
        else:
            time = bhv["time"]

        if self.include_ctr:
            # -- Get History CTR
            history_ctr = np.array(bhv["history_ctr"])[: self.max_history_len]
            history, history_ctr = zip(*history_ctr)
            # convert to np array
            history = np.array(history)
            history_ctr = np.array(history_ctr)

            # -- Get candidates CTR and Recency
            candidates_ctr = np.array(bhv["candidates_ctr"])
            candidates, candidates_ctr, candidates_rec = zip(*candidates_ctr)

            # convert to np array
            candidates = np.array(candidates)
            candidates, labels, indices = self._sample_candidates(candidates, labels)

            candidates_ctr = np.array(candidates_ctr)[indices]
            candidates_rec = np.array(candidates_rec)[indices]

        else:
            history = np.array(bhv["history"])[: self.max_history_len]
            candidates = np.array(bhv["candidates"])
            candidates, labels, _ = self._sample_candidates(candidates, labels)

        if history.size == 1 and history[0] == '':
            history = self._initialize_cold_start()
        else:
            history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        if self.include_ctr:
            return user_id, user_idx, history, candidates, labels, time, history_ctr, candidates_ctr, candidates_rec
        return user_id, user_idx, history, candidates, labels


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

        return candidates, labels, indices

    def _initialize_cold_start(self):
        """
        In cold start cases, history can be empty thus we need to 
        add a dataframe with empty values for the embedding.
        """
        # Initialize an empty DataFrame with specified columns
        history = pd.DataFrame(columns=['title', 'abstract', 'sentiment_class', 'sentiment_score'])

        # Create a new DataFrame for the row you wish to append
        new_row = pd.DataFrame([{
            'title': '', 
            'abstract': '', 
            'sentiment_class': 0,
            'sentiment_score': 0.0
        }])

        # Use pandas.concat to append the new row to the original DataFrame
        history = pd.concat([history, new_row], ignore_index=True)

        # Explicitly set the data types for the entire DataFrame
        history = history.astype({
            'title': 'object',
            'abstract': 'object',
            'sentiment_class': 'int64',
            'sentiment_score': 'float64'
        })

        return history


class RecommendationDatasetTest(MINDDataFrame):
    def __init__(self, news: pd.DataFrame, behaviors: pd.DataFrame, max_history_len: int, include_ctr: Optional[bool] = False) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_len = max_history_len
        self.include_ctr = include_ctr
        

    def __getitem__(self, idx: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bhv = self.behaviors.iloc[idx]

        user_id = np.array([int(bhv["uid"].split("U")[-1])])
        user_idx = np.array([int(bhv["user"])])

        labels = np.array(bhv["labels"])
        if isinstance(bhv["time"], str):
            time = datetime.strptime(bhv["time"], "%Y-%m-%d %H:%M:%S")
        else:
            time = bhv["time"]

        if self.include_ctr:
            # -- Get History CTR
            history_ctr = np.array(bhv["history_ctr"])[: self.max_history_len]
            history, history_ctr, _ = zip(*history_ctr)
            # convert to np array
            history = np.array(history)
            history_ctr = np.array(history_ctr)

            # -- Get candidates CTR and Recency
            candidates_ctr = np.array(bhv["candidates_ctr"])
            candidates, candidates_ctr, candidates_rec = zip(*candidates_ctr)
            # convert to np array
            candidates = np.array(candidates)
            candidates_ctr = np.array(candidates_ctr)
            candidates_rec = np.array(candidates_rec)
        else:
            history = np.array(bhv["history"])[: self.max_history_len]
            candidates = np.array(bhv["candidates"])

        if history.size == 1 and history[0] == '':
            history = self._initialize_cold_start()
        else:
            history = self.news.loc[history]
        candidates = self.news.loc[candidates]


        if self.include_ctr:
            return user_id, user_idx, history, candidates, labels, time, history_ctr, candidates_ctr, candidates_rec
        return user_id, user_idx, history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def _initialize_cold_start(self):
        """
        In cold start cases, history can be empty thus we need to 
        add a dataframe with empty values for the embedding.
        """
        # Initialize an empty DataFrame with specified columns
        history = pd.DataFrame(columns=['title', 'abstract', 'sentiment_class', 'sentiment_score'])

        # Create a new DataFrame for the row you wish to append
        new_row = pd.DataFrame([{
            'title': '', 
            'abstract': '', 
            'sentiment_class': 0,
            'sentiment_score': 0.0
        }])

        # Use pandas.concat to append the new row to the original DataFrame
        history = pd.concat([history, new_row], ignore_index=True)

        # Explicitly set the data types for the entire DataFrame
        history = history.astype({
            'title': 'object',
            'abstract': 'object',
            'sentiment_class': 'int64',
            'sentiment_score': 'float64'
        })

        return history


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
        include_ctr: Optional[bool] = False,
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
        self.include_ctr = include_ctr

    def __call__(self, batch) -> RecommendationBatch:
        if self.include_ctr:
            # Under this condition histories and candidates includes CTR information
            user_ids, user_idx, histories, candidates, labels, times, histories_ctr, candidates_ctr, candidates_rec = zip(*batch)

            histories_ctr = torch.from_numpy(np.concatenate(histories_ctr).astype(np.int64)).long()
            candidates_ctr = torch.from_numpy(np.concatenate(candidates_ctr).astype(np.int64)).long()
            candidates_rec = torch.from_numpy(np.concatenate(candidates_rec).astype(np.float32)).long()
            times = self._get_timestamp(times)
        else:
            user_ids, user_idx, histories, candidates, labels = zip(*batch)

        batch_hist = self._make_batch_asignees(histories)
        batch_cand = self._make_batch_asignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories))
        x_cand = self._tokenize_df(pd.concat(candidates))
        labels = torch.from_numpy(np.concatenate(labels)).float()
        user_ids = torch.from_numpy(np.concatenate(user_ids)).long()
        user_idx = torch.from_numpy(np.concatenate(user_idx)).long()

        if self.include_ctr:
            return RecommendationBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                labels=labels,
                user_ids=user_ids,
                user_idx=user_idx,
                times=times,
                x_hist_ctr=histories_ctr,
                x_cand_ctr=candidates_ctr,
                x_cand_rec=candidates_rec
            )
        else:
            return RecommendationBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                labels=labels,
                user_ids=user_ids,
                user_idx=user_idx,
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

        # news IDs (i.e., keep only numeric part of unique NID)
        nids = np.array([int(nid.split("N")[-1]) for nid in df.index.values])
        batch_out["news_ids"] = torch.from_numpy(nids).long()

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

        if ("category_class" in self.dataset_attributes) or (
            "subcategory_class" in self.dataset_attributes
        ):
            # prepare other aspects
            category = torch.from_numpy(df["category_class"].values).long()
            subcategory = torch.from_numpy(
                df["subcategory_class"].values).long()

            batch_out["category"] = category
            batch_out["subcategory"] = subcategory

        if ("sentiment_class" in self.dataset_attributes) or (
            "sentiment_score" in self.dataset_attributes
        ):
            sentiment = torch.from_numpy(df["sentiment_class"].values).long()
            sentiment_score = torch.from_numpy(
                df["sentiment_score"].values).float()

            batch_out["sentiment"] = sentiment
            batch_out["sentiment_score"] = sentiment_score

        return batch_out

    def _make_batch_asignees(self, items: Sequence[Sequence[Any]]) -> torch.Tensor:
        """Constructs a batch tensor for assignees based on the sizes of sub-sequences within `items`.

        This method processes a sequence of item sequences, typically representing groups
        of query results from a news articles dataframe, and creates a batch tensor. Each element
        in `items` represents a group of articles, and the method generates a tensor where each
        group index is repeated according to the number of articles in the group. This tensor can
        be used to assign batch indices to individual articles based on their group.

        Parameters
        ----------
        items : Sequence[Sequence[Any]]
            A sequence of item sequences. Each inner sequence represents a group of articles
            and is expected to be a sequence of any type (usually article IDs or embeddings).

        Returns
        -------
        torch.Tensor
            A 1D tensor of type torch.int64, where each element is the index of a group in `items`.
            The index is repeated as many times as there are elements in the corresponding group.

        Examples
        --------
        >>> items = [['a1', 'a2', 'a3'], ['b1', 'b2'], ['c1']]
        >>> batch = _make_batch_assignees(items)
        >>> print(batch)
        tensor([0, 0, 0, 1, 1, 2])

        Note
        ----
        This method is specifically useful for preparing batches for model training or inference
        where articles are grouped based on certain criteria (e.g., publication date, topic) and
        each group needs to be processed together.

        """
        sizes = torch.tensor([len(x) for x in items])
        batch = torch.repeat_interleave(torch.arange(len(items)), sizes)

        return batch

    def _get_timestamp(self, times: Sequence[str]) -> torch.Tensor:
        """
        Get a list of the format 
        (array('2019-11-14 07:01:48', dtype='<U19'), array('2019-11-14 08:38:04', dtype='<U19'))
        and convert it into torch.Tensor
        """
        # Convert to list of datetime strings
        datetime_strings = [str(date) for date in times]

        # Step 1: Convert datetime strings to datetime objects
        datetime_objects = pd.to_datetime(datetime_strings)

        # Step 2: Convert datetime objects to POSIX timestamps
        timestamps = datetime_objects.astype('int64') // 10**9

        # Step 3: Create a PyTorch tensor from these timestamps
        timestamps_tensor = torch.tensor(timestamps)

        return timestamps_tensor
