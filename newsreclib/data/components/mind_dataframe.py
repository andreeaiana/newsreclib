import json
import os
from ast import literal_eval
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

import newsreclib.data.components.data_utils as data_utils
import newsreclib.data.components.file_utils as file_utils
from newsreclib import utils

tqdm.pandas()

log = utils.get_pylogger(__name__)


class MINDDataFrame(Dataset):
    """Creates a dataframe for the MIND dataset.

    Additionally:
        - Downloads the dataset for the specified size.
        - Downloads pretrained embeddings.
        - Parses the news and behaviors data.
        - Annotates the news with additional aspects (e.g., `sentiment`).
        - Split the behaviors into `train` and `validation` sets by time.

    Attributes:
        dataset_size:
            A string indicating the type of the dataset. Choose between `large` and `small`.
        dataset_url:
            Dictionary of URLs for downloading the `train` and `dev` datasets for the specified `dataset_size`.
        data_dir:
            Path to the data directory.
        dataset_attributes:
            List of news features available in the used dataset (e.g., title, category, etc.).
        id2index_filenames:
            Dictionary mapping id2index dictionary to corresponding filenames.
        pretrained_embeddings_url:
            URL for downloading pretrained word embeddings (e.g., Glove).
        word_embeddings_dirname:
            Directory where to download and extract the pretrained word embeddings.
        word_embeddings_fpath:
            Filepath to the pretrained word embeddings.
        entity_embeddings_filename:
            Filepath to the pretrained entity embeddings.
        use_plm:
            If ``True``, it will process the data for a petrained language model (PLM) in the news encoder. If ``False``, it will tokenize the news title and abstract to be used initialized with pretrained word embeddings.
        use_pretrained_categ_embeddings:
            Whether to initialize category embeddings with pretrained word embeddings.
        categ_embed_dim:
            Dimensionality of category embeddings.
        word_embed_dim:
            Dimensionality of word embeddings.
        entity_embed_dim:
            Dimensionality of entity embeddings.
        entity_freq_threshold:
            Minimum frequency for an entity to be included in the processed dataset.
        entity_conf_threshold:
            Minimum confidence for an entity to be included in the processed dataset.
        sentiment_annotator:
            The sentiment annotator module used.
        valid_time_split:
            A string with the date before which click behaviors are included in the train set. After this date, behaviors are included in the validation set.
        train:
            If ``True``, the data will be processed and used for training. If ``False``, it will be processed and used for validation or testing.
        validation:
            If ``True`` and `train` is also``True``, the data will be processed and used for validation. If ``False`` and `train` is `True``, the data will be processed ad used for training. If ``False`` and `train` is `False``, the data will be processed and used for testing.
        download:
            Whether to download the dataset, if not already downloaded.
    """

    def __init__(
        self,
        dataset_size: str,
        dataset_url: DictConfig,
        data_dir: str,
        dataset_attributes: List[str],
        id2index_filenames: DictConfig,
        pretrained_embeddings_url: Optional[str],
        word_embeddings_dirname: Optional[str],
        word_embeddings_fpath: Optional[str],
        entity_embeddings_filename: str,
        use_plm: bool,
        use_pretrained_categ_embeddings: bool,
        word_embed_dim: Optional[int],
        categ_embed_dim: Optional[int],
        entity_embed_dim: int,
        entity_freq_threshold: int,
        entity_conf_threshold: float,
        sentiment_annotator: nn.Module,
        valid_time_split: str,
        train: bool,
        validation: bool,
        download: bool,
    ) -> None:
        super().__init__()

        self.dataset_size = dataset_size
        self.dataset_url = dataset_url
        self.data_dir = data_dir
        self.dataset_attributes = dataset_attributes
        self.id2index_filenames = id2index_filenames

        self.use_plm = use_plm
        self.use_pretrained_categ_embeddings = use_pretrained_categ_embeddings

        if not self.use_plm or self.use_pretrained_categ_embeddings:
            assert isinstance(word_embed_dim, int)
            self.word_embed_dim = word_embed_dim

        if self.use_pretrained_categ_embeddings:
            assert isinstance(categ_embed_dim, int)
            self.categ_embed_dim = categ_embed_dim

        self.entity_embed_dim = entity_embed_dim
        self.entity_freq_threshold = entity_freq_threshold
        self.entity_conf_threshold = entity_conf_threshold
        self.entity_embeddings_filename = entity_embeddings_filename

        self.sentiment_annotator = sentiment_annotator

        self.valid_time_split = valid_time_split

        self.validation = validation
        self.data_split = "train" if train else "dev"

        self.dst_dir = os.path.join(
            self.data_dir, "MIND" + self.dataset_size + "_" + self.data_split
        )

        if download:
            url = dataset_url[dataset_size][self.data_split]
            log.info(
                f"Downloading MIND{self.dataset_size} dataset for {self.data_split} from {url}."
            )
            data_utils.download_and_extract_dataset(
                data_dir=self.data_dir,
                url=url,
                filename=url.split("/")[-1],
                extract_compressed=True,
                dst_dir=self.dst_dir,
                clean_archive=False,
            )

            if not self.use_plm or self.use_pretrained_categ_embeddings:
                assert isinstance(pretrained_embeddings_url, str)
                assert isinstance(word_embeddings_dirname, str)
                assert isinstance(word_embeddings_fpath, str)
                data_utils.download_and_extract_pretrained_embeddings(
                    data_dir=self.data_dir,
                    url=pretrained_embeddings_url,
                    pretrained_embeddings_fpath=word_embeddings_fpath,
                    filename=pretrained_embeddings_url.split("/")[-1],
                    dst_dir=os.path.join(self.data_dir, word_embeddings_dirname),
                    clean_archive=True,
                )

        self.word_embeddings_fpath = word_embeddings_fpath

        self.news, self.behaviors = self.load_data()

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        user_bhv = self.behaviors.iloc[index]

        history = user_bhv["history"]
        cand = user_bhv["cand"]
        labels = user_bhv["labels"]

        history = self.news[history]
        cand = self.news.loc[cand]
        labels = np.array(labels)

        return history, cand, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the parsed news and user behaviors.

        Returns:
            Tuple of news and behaviors datasets.
        """
        news = self._load_news()
        log.info(f"News data size: {len(news)}")

        behaviors = self._load_behaviors()
        log.info(
            f"Behaviors data size for data split {self.data_split}, validation={self.validation}: {len(behaviors)}"
        )

        return news, behaviors

    def _load_news(self) -> pd.DataFrame:
        """Loads the parsed news. If not already parsed, loads and preprocesses the raw news data.

        Args:
            news:
                Dataframe of news articles.

        Returns:
            Parsed and annotated news data.
        """
        parsed_news_file = os.path.join(self.dst_dir, "parsed_news.tsv")

        if file_utils.check_integrity(parsed_news_file):
            # news already parsed
            log.info(f"News already parsed. Loading from {parsed_news_file}.")

            attributes2convert = ["title_entities", "abstract_entities"]
            if not self.use_plm:
                attributes2convert.extend(["tokenized_title", "tokenized_abstract"])
            news = pd.read_table(
                filepath_or_buffer=parsed_news_file,
                converters={attribute: literal_eval for attribute in attributes2convert},
            )
            news["abstract"].fillna("", inplace=True)
        else:
            log.info("News not parsed. Loading and parsing raw data.")
            columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]
            news = pd.read_table(
                filepath_or_buffer=os.path.join(self.dst_dir, "news.tsv"),
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )
            news = news.drop(columns=["url"])

            # replace missing values
            news["abstract"].fillna("", inplace=True)
            news["title_entities"].fillna("[]", inplace=True)
            news["abstract_entities"].fillna("[]", inplace=True)

            if not self.use_plm:
                word2index_fpath = os.path.join(
                    self.data_dir,
                    "MIND" + self.dataset_size + "_train",
                    self.id2index_filenames["word2index"],
                )
                transformed_word_embeddings_fpath = os.path.join(
                    self.dst_dir,
                    "transformed_word_embeddings",
                )

            if self.use_pretrained_categ_embeddings:
                transformed_categ_embeddings_fpath = os.path.join(
                    self.dst_dir,
                    "transformed_categ_embeddings",
                )

            entity2index_fpath = os.path.join(
                self.data_dir,
                "MIND" + self.dataset_size + "_train",
                self.id2index_filenames["entity2index"],
            )
            categ2index_fpath = os.path.join(
                self.data_dir,
                "MIND" + self.dataset_size + "_train",
                self.id2index_filenames["categ2index"],
            )
            subcateg2index_fpath = os.path.join(
                self.data_dir,
                "MIND" + self.dataset_size + "_train",
                self.id2index_filenames["subcateg2index"],
            )
            transformed_entity_embeddings_fpath = os.path.join(
                self.dst_dir,
                "transformed_entity_embeddings",
            )

            if (
                "sentiment_class" in self.dataset_attributes
                or "sentiment_score" in self.dataset_attributes
            ):
                sentiment2index_fpath = os.path.join(
                    self.data_dir,
                    "MIND" + self.dataset_size + "_train",
                    self.id2index_filenames["sentiment2index"],
                )

                # compute sentiment classes
                log.info("Computing sentiments.")
                news["sentiment_preds"] = news["title"].progress_apply(
                    lambda text: self.sentiment_annotator(text)
                )
                news["sentiment_class"], news["sentiment_score"] = zip(*news["sentiment_preds"])
                news.drop(columns=["sentiment_preds"], inplace=True)
                log.info("Sentiments computation completed.")

            if self.data_split == "train":
                if not self.use_plm:
                    # tokenize text
                    news["tokenized_title"] = news["title"].progress_apply(
                        data_utils.word_tokenize
                    )
                    news["tokenized_abstract"] = news["abstract"].progress_apply(
                        data_utils.word_tokenize
                    )

                    # construct word2index map
                    log.info("Constructing word2index map.")
                    word_cnt = Counter()
                    for idx in tqdm(news.index.tolist()):
                        word_cnt.update(news.loc[idx]["tokenized_title"])
                        word_cnt.update(news.loc[idx]["tokenized_abstract"])
                    word2index = {k: v + 1 for k, v in zip(word_cnt, range(len(word_cnt)))}
                    log.info(
                        f"Saving word2index map of size {len(word2index)} in {word2index_fpath}"
                    )
                    file_utils.to_tsv(
                        df=pd.DataFrame(word2index.items(), columns=["word", "index"]),
                        fpath=word2index_fpath,
                    )

                # construct entity2index map
                log.info("Constructing entity2index map.")

                # keep only entities with a confidence over the threshold
                entity2freq = {}
                entity2freq = self._count_entity_freq(news["title_entities"], entity2freq)
                entity2freq = self._count_entity_freq(news["abstract_entities"], entity2freq)

                # keep only entities with a frequency over the threshold
                entity2index = {}
                for entity, freq in entity2freq.items():
                    if freq > self.entity_freq_threshold:
                        entity2index[entity] = len(entity2index) + 1

                log.info(
                    f"Saving entity2index map of size {len(entity2index)} in {entity2index_fpath}"
                )
                file_utils.to_tsv(
                    df=pd.DataFrame(entity2index.items(), columns=["entity", "index"]),
                    fpath=entity2index_fpath,
                )

                # construct category2index
                log.info("Constructing categ2index map.")
                news_category = news["category"].drop_duplicates().reset_index(drop=True)
                categ2index = {v: k + 1 for k, v in news_category.to_dict().items()}
                log.info(
                    f"Saving categ2index map of size {len(categ2index)} in {categ2index_fpath}"
                )
                file_utils.to_tsv(
                    df=pd.DataFrame(categ2index.items(), columns=["category", "index"]),
                    fpath=categ2index_fpath,
                )

                # subcateg2index map
                log.info("Constructing subcateg2index map.")
                news_subcategory = news["subcategory"].drop_duplicates().reset_index(drop=True)
                subcateg2index = {v: k + 1 for k, v in news_subcategory.to_dict().items()}
                log.info(
                    f"Saving subcateg2index map of size {len(subcateg2index)} in {subcateg2index_fpath}"
                )
                file_utils.to_tsv(
                    df=pd.DataFrame(subcateg2index.items(), columns=["subcategory", "index"]),
                    fpath=subcateg2index_fpath,
                )

                # compute sentiment classes
                if (
                    "sentiment_class" in self.dataset_attributes
                    or "sentiment_score" in self.dataset_attributes
                ):
                    # sentiment2index map
                    log.info("Constructing sentiment2index map.")
                    news_sentiment = (
                        news["sentiment_class"].drop_duplicates().reset_index(drop=True)
                    )
                    sentiment2index = {v: k + 1 for k, v in news_sentiment.to_dict().items()}
                    log.info(
                        f"Saving sentiment2index map of size {len(sentiment2index)} in {sentiment2index_fpath}"
                    )
                    file_utils.to_tsv(
                        df=pd.DataFrame(sentiment2index.items(), columns=["sentiment", "index"]),
                        fpath=sentiment2index_fpath,
                    )

            else:
                log.info("Loading indices maps.")

                if not self.use_plm:
                    # load word2index map
                    word2index = file_utils.load_idx_map_as_dict(word2index_fpath)

                # load entity2index map
                entity2index = file_utils.load_idx_map_as_dict(entity2index_fpath)

                # load categ2index map
                categ2index = file_utils.load_idx_map_as_dict(categ2index_fpath)

                # load subcateg2index map
                subcateg2index = file_utils.load_idx_map_as_dict(subcateg2index_fpath)

                if (
                    "sentiment_class" in self.dataset_attributes
                    or "sentiment_score" in self.dataset_attributes
                ):
                    # load sentiment2index map
                    sentiment2index = file_utils.load_idx_map_as_dict(sentiment2index_fpath)

            log.info(f"Number of category classes: {len(categ2index)}.")
            log.info(f"Number of subcategory classes: {len(subcateg2index)}.")
            if (
                "sentiment_class" in self.dataset_attributes
                or "sentiment_score" in self.dataset_attributes
            ):
                log.info(f"Number of sentiment classes: {len(sentiment2index)}.")

            if not self.use_plm:
                # construct word embeddings matrix
                log.info("Constructing word embedding matrix.")
                data_utils.generate_pretrained_embeddings(
                    word2index=word2index,
                    embeddings_fpath=self.word_embeddings_fpath,
                    embed_dim=self.word_embed_dim,
                    transformed_embeddings_fpath=transformed_word_embeddings_fpath,
                )

            if self.use_pretrained_categ_embeddings:
                # construct category embeddings matrix
                log.info("Constructing category embedding matrix.")
                data_utils.generate_pretrained_embeddings(
                    word2index=categ2index,
                    embeddings_fpath=self.word_embeddings_fpath,
                    embed_dim=self.categ_embed_dim,
                    transformed_embeddings_fpath=transformed_categ_embeddings_fpath,
                )

            # construct entity embeddings matrix
            log.info("Constructing entity embedding matrix.")
            self.generate_entity_embeddings(
                entity2index=entity2index,
                transformed_embeddings_fpath=transformed_entity_embeddings_fpath,
            )

            # parse news
            log.info("Parsing news")
            if not self.use_plm:
                news["tokenized_title"] = news["title"].progress_apply(data_utils.word_tokenize)
                news["tokenized_abstract"] = news["abstract"].progress_apply(
                    data_utils.word_tokenize
                )
                news["tokenized_title"] = news["tokenized_title"].progress_apply(
                    lambda title: [word2index.get(x, 0) for x in title]
                )
                news["tokenized_abstract"] = news["tokenized_abstract"].progress_apply(
                    lambda abstract: [word2index.get(x, 0) for x in abstract]
                )

            news["category_class"] = news["category"].progress_apply(
                lambda category: categ2index.get(category, 0)
            )
            news["subcategory_class"] = news["subcategory"].progress_apply(
                lambda subcategory: subcateg2index.get(subcategory, 0)
            )

            news["title_entities"] = news["title_entities"].progress_apply(
                lambda row: self._filter_entities(row, entity2index)
            )
            news["abstract_entities"] = news["abstract_entities"].progress_apply(
                lambda row: self._filter_entities(row, entity2index)
            )

            if (
                "sentiment_class" in self.dataset_attributes
                or "sentiment_score" in self.dataset_attributes
            ):
                news["sentiment_class"] = news["sentiment_class"].progress_apply(
                    lambda sentiment: sentiment2index.get(sentiment, 0)
                )

            # cache parsed news
            log.info(f"Caching parsed news of size {len(news)} to {parsed_news_file}.")
            file_utils.to_tsv(news, parsed_news_file)

        news = news.set_index("nid", drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """Loads the parsed user behaviors. If not already parsed, loads and parses the raw
        behavior data.

        Returns:
            Parsed and split user behavior data.
        """
        file_prefix = ""
        if self.data_split == "train":
            file_prefix = "train_" if not self.validation else "val_"
        parsed_bhv_file = os.path.join(self.dst_dir, file_prefix + "parsed_behaviors.tsv")

        if file_utils.check_integrity(parsed_bhv_file):
            # behaviors already parsed
            log.info(f"User behaviors already parsed. Loading from {parsed_bhv_file}.")
            behaviors = pd.read_table(
                filepath_or_buffer=parsed_bhv_file,
                converters={
                    "history": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "candidates": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "labels": lambda x: list(map(int, x.strip("[]").split(", "))),
                },
            )
        else:
            log.info("User behaviors not parsed. Loading and parsing raw data.")

            # load behaviors
            column_names = ["impid", "uid", "time", "history", "impressions"]
            behaviors = pd.read_table(
                filepath_or_buffer=os.path.join(self.dst_dir, "behaviors.tsv"),
                header=None,
                names=column_names,
                usecols=range(len(column_names)),
            )

            # parse behaviors
            log.info("Parsing behaviors.")
            behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
            behaviors["history"] = behaviors["history"].fillna("").str.split()
            behaviors["impressions"] = behaviors["impressions"].str.split()
            behaviors["candidates"] = behaviors["impressions"].apply(
                lambda x: [impression.split("-")[0] for impression in x]
            )
            behaviors["labels"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[1]) for impression in x]
            )
            behaviors = behaviors.drop(columns=["impressions"])

            cnt_bhv = len(behaviors)
            behaviors = behaviors[behaviors["history"].apply(len) > 0]
            dropped_bhv = cnt_bhv - len(behaviors)
            log.info(
                f"Removed {dropped_bhv} ({dropped_bhv / cnt_bhv}%) behaviors without user history"
            )

            behaviors = behaviors.reset_index(drop=True)

            if self.data_split == "train":
                log.info("Splitting behavior data into train and validation sets.")
                if not self.validation:
                    # training set
                    behaviors = behaviors.loc[behaviors["time"] < self.valid_time_split]
                    behaviors = behaviors.reset_index(drop=True)

                    # construct uid2index map
                    log.info("Constructing uid2index map")
                    uid2index = {}
                    for idx in tqdm(behaviors.index.tolist()):
                        uid = behaviors.loc[idx]["uid"]
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index) + 1

                    fpath = os.path.join(self.dst_dir, self.id2index_filenames["uid2index"])
                    log.info(f"Saving uid2index map of size {len(uid2index)} in {fpath}.")
                    file_utils.to_tsv(
                        df=pd.DataFrame(uid2index.items(), columns=["uid", "index"]), fpath=fpath
                    )

                else:
                    # validation set
                    behaviors = behaviors.loc[behaviors["time"] >= self.valid_time_split]
                    behaviors = behaviors.reset_index(drop=True)

                    # load uid2index map
                    log.info("Loading uid2index map.")
                    fpath = os.path.join(
                        self.data_dir,
                        "MIND" + self.dataset_size + "_train",
                        self.id2index_filenames["uid2index"],
                    )
                    uid2index = file_utils.load_idx_map_as_dict(fpath)

            else:
                # test set
                # load uid2index map
                log.info("Loading uid2index map.")
                fpath = os.path.join(
                    self.data_dir,
                    "MIND" + self.dataset_size + "_train",
                    self.id2index_filenames["uid2index"],
                )
                uid2index = file_utils.load_idx_map_as_dict(fpath)

            log.info(f"Number of users: {len(uid2index)}.")

            # map uid to index
            log.info("Mapping uid to index.")
            behaviors["user"] = behaviors["uid"].apply(lambda x: uid2index.get(x, 0))

            # cache parsed behaviors
            log.info(f"Caching parsed behaviors of size {len(behaviors)} to {parsed_bhv_file}.")
            behaviors = behaviors[["uid", "user", "history", "candidates", "labels"]]
            file_utils.to_tsv(behaviors, parsed_bhv_file)

        return behaviors

    def _count_entity_freq(self, data: pd.Series, entity2freq: Dict[str, int]) -> Dict[str, int]:
        for row in tqdm(data):
            for entity in json.loads(row):
                times = len(entity["OccurrenceOffsets"]) * entity["Confidence"]
                if times > 0:
                    if entity["WikidataId"] not in entity2freq:
                        entity2freq[entity["WikidataId"]] = times
                    else:
                        entity2freq[entity["WikidataId"]] += times

        return entity2freq

    def _filter_entities(self, data: pd.Series, entity2index: Dict[str, int]) -> List[int]:
        filtered_entities = []
        for entity in json.loads(data):
            if (
                entity["Confidence"] > self.entity_conf_threshold
                and entity["WikidataId"] in entity2index
            ):
                filtered_entities.append(entity2index[entity["WikidataId"]])

        return filtered_entities

    def generate_entity_embeddings(
        self, entity2index: pd.DataFrame, transformed_embeddings_fpath: str
    ):
        entity2index_df = pd.DataFrame(entity2index.items(), columns=["entity", "index"])
        entity_embedding = pd.read_table(
            os.path.join(self.dst_dir, self.entity_embeddings_filename), header=None
        )
        entity_embedding["vector"] = entity_embedding.iloc[:, 1:101].values.tolist()
        entity_embedding = entity_embedding[[0, "vector"]].rename(columns={0: "entity"})

        merged_df = pd.merge(entity_embedding, entity2index_df, on="entity").sort_values("index")
        entity_embedding_transformed = np.random.normal(
            size=(len(entity2index_df) + 1, self.entity_embed_dim)
        )
        for row in merged_df.itertuples(index=False):
            entity_embedding_transformed[row.index] = row.vector

        # cache transformed embeddings
        np.save(
            transformed_embeddings_fpath,
            entity_embedding_transformed,
            allow_pickle=True,
        )
