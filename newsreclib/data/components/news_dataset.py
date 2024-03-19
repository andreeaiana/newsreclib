import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from newsreclib.data.components.batch import NewsBatch
from newsreclib.data.components.mind_dataframe import MINDDataFrame


class NewsDataset(MINDDataFrame):
    """
    Attributes:
        news:
            A dataframe containing news with various features (e.g., title, category)
        behaviors:
            A dataframe of user click behaviors.
        aspect:
            The aspect to use for deriving labels (e.g., `category`, `sentiment`).
    """

    def __init__(
        self,
        news: pd.DataFrame,
        behaviors: pd.DataFrame,
        aspect: str,
    ) -> None:
        news_ids = np.array(
            list(
                set(
                    list(itertools.chain.from_iterable(behaviors.history))
                    + list(itertools.chain.from_iterable(behaviors.candidates))
                )
            )
        )

        self.news = news.loc[news_ids]
        self.labels = np.array(self.news[aspect + "_class"])

    def __getitem__(self, index: Any) -> Tuple[pd.DataFrame, int]:
        news = self.news.iloc[[index]]
        label = self.labels[index]

        return news, label

    def __len__(self) -> int:
        return len(self.news)


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

    def __call__(self, batch) -> NewsBatch:
        news, labels = zip(*batch)

        transformed_news = self._tokenize_df(pd.concat(news))
        labels = torch.tensor(labels).long()

        return NewsBatch(news=transformed_news, labels=labels)

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
                    df["title_entities"].values.tolist(), max_len=None
                )
                batch_out["title_entities"] = title_entities

            if "abstract" in self.dataset_attributes:
                abstract_entities = self._tokenize_embeddings(
                    df["abstract_entities"].values.tolist(), max_len=None
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

        return batch_out
