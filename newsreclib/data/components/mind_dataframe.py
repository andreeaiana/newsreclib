import json
import os
from ast import literal_eval
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

import newsreclib.data.components.data_utils as data_utils
import newsreclib.data.components.file_utils as file_utils
import newsreclib.utils as utils 
from newsreclib import utils
from datetime import datetime, timezone, timedelta

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
        include_ctr:
            Controling if we should include CTR or not into history/candidates
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
        include_ctr: Optional[bool],
    ) -> None:
        super().__init__()

        self.dataset_size = dataset_size
        self.dataset_url = dataset_url
        self.data_dir = data_dir
        self.dataset_attributes = dataset_attributes
        self.id2index_filenames = id2index_filenames

        self.include_ctr = include_ctr

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
            for data_type in ["train", "dev"]:
                dst_dir_ = os.path.join(
                    self.data_dir, "MIND" + self.dataset_size + "_" + data_type
                )
                url = dataset_url[dataset_size][data_type]
                log.info(
                    f"Downloading MIND{self.dataset_size} dataset for {data_type} from {url}."
                )
                data_utils.download_and_extract_dataset(
                    data_dir=self.data_dir,
                    url=url,
                    filename=url.split("/")[-1],
                    extract_compressed=True,
                    dst_dir=dst_dir_,
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
                    dst_dir=os.path.join(
                        self.data_dir, word_embeddings_dirname),
                    clean_archive=True,
                )

        self.word_embeddings_fpath = word_embeddings_fpath

        self.news, self.behaviors = self.load_data()
        self.news_metrics_bucket = None
        self.articles_est_pb_time = None

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

        if self.include_ctr:
            behaviors = self._load_behaviors_ctr(behaviors)

        return news, behaviors

    def _load_behaviors_ctr(self, behaviors: pd.DataFrame) -> pd.DataFrame:
        """ Load the parsed behaviors with CTR information.
        If it does not already have the information of CTR load it.
        """
        # Check if behaviors file with CTR already exists
        file_prefix = ""
        if self.data_split == "train":
            file_prefix = "train_" if not self.validation else "val_"
        parsed_bhv_ctr_file = os.path.join(
            self.dst_dir, file_prefix + "parsed_behaviors_ctr.tsv")

        if file_utils.check_integrity(parsed_bhv_ctr_file):
            log.info("CTR has already been added.")
            # behaviors already parsed
            log.info(
                f"User behaviors already parsed. Loading from {parsed_bhv_ctr_file}.")
            behaviors = pd.read_table(filepath_or_buffer=parsed_bhv_ctr_file)
            behaviors = pd.read_table(
                filepath_or_buffer=parsed_bhv_ctr_file,
                converters={
                    "history": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "candidates": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "labels": lambda x: list(map(int, x.strip("[]").split(", "))),
                    "history_ctr": lambda x: literal_eval(x),
                    "candidates_ctr": lambda x: literal_eval(x),
                }
            )
        else:
            # Load news metrics bucket file
            news_metrics_bucket_ptb, news_metrics_bucket_acc = self._load_news_metrics_bucket()

            log.info("Adding CTR information into behaviors file...")
            # Get pickle file for estimated publish time
            article2published = self.get_est_publish_time()
            acc = True
            if acc:
                # Parse behaviors to add ctr information
                behaviors = behaviors.progress_apply(
                    lambda row: self._ctr_per_row(news_metrics_bucket_acc, row, article2published), axis=1)
            else:
                # Parse behaviors to add ctr information
                behaviors = behaviors.progress_apply(
                    lambda row: self._ctr_per_row(news_metrics_bucket_ptb, row, article2published), axis=1)

            # Save behaviors file with ctr information
            file_utils.to_tsv(behaviors, parsed_bhv_ctr_file)
            log.info("CTR information added to behaviors file!")

        return behaviors

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
                attributes2convert.extend(
                    ["tokenized_title", "tokenized_abstract"])
            news = pd.read_table(
                filepath_or_buffer=parsed_news_file,
                converters={
                    attribute: literal_eval for attribute in attributes2convert},
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

            # add estimated publish time column
            article2published = self.get_est_publish_time(news['nid'].unique())
            news["est_publish_time"] = news["nid"].map(article2published)

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

            if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
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
                if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
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

                # compute sentiment classes
                if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
                    # sentiment2index map
                    log.info("Constructing sentiment2index map.")
                    news_sentiment = (
                        news["sentiment_class"].drop_duplicates(
                        ).reset_index(drop=True)
                    )
                    sentiment2index = {v: k + 1 for k,
                                       v in news_sentiment.to_dict().items()}
                    log.info(
                        f"Saving sentiment2index map of size {len(sentiment2index)} in {sentiment2index_fpath}"
                    )
                    file_utils.to_tsv(
                        df=pd.DataFrame(sentiment2index.items(), columns=[
                                        "sentiment", "index"]),
                        fpath=sentiment2index_fpath,
                    )

                if not self.use_plm:
                    # load word2index map
                    word2index = file_utils.load_idx_map_as_dict(word2index_fpath)

                # load entity2index map
                entity2index = file_utils.load_idx_map_as_dict(entity2index_fpath)

                # load categ2index map
                categ2index = file_utils.load_idx_map_as_dict(categ2index_fpath)

                # load subcateg2index map
                subcateg2index = file_utils.load_idx_map_as_dict(subcateg2index_fpath)

                if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
                    # load subcateg2index map
                    sentiment2index = file_utils.load_idx_map_as_dict(sentiment2index_fpath)

            log.info(f"Number of category classes: {len(categ2index)}.")
            log.info(f"Number of subcategory classes: {len(subcateg2index)}.")
            if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
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

            if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
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
                }
            )
        else:
            log.info("User behaviors not parsed. Loading and parsing raw data.")

            # load behaviors
            column_names = ["impid", "uid", "time", "history", "impressions"]
            behaviors = pd.read_table(
                filepath_or_buffer=os.path.join(self.dst_dir, "behaviors.tsv"),
                header=None,
                names=column_names,
                usecols=range(len(column_names))
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
            behaviors = behaviors[["user", "history", "candidates", "labels", "time"]]
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

    def get_est_publish_time(self, unique_ids: Optional[object] = None) -> Dict:
        """
        Get the estimated publish time of the news article.
        It only applies for the MIND dataset, because it doesn't
        have the publish time information per news article.

        In order to get the publish time we'll use the pickle file
        made available by the paper: 
        "Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation"

        Ref (1): https://github.com/summmeer/session-based-news-recommendation
        Ref (2): https://github.com/summmeer/session-based-news-recommendation/issues/6#issuecomment-1233830425

        Arguments:
            - unique_ids: the set of ids on news dataframe
        """
        # -- Check if we already have this file
        est_pb_time = os.path.join(self.data_dir, "articles_est_pb_time.pkl")

        if file_utils.check_integrity(est_pb_time):
            return pd.read_pickle(est_pb_time)
        else:
            article2published = self.get_est_pbt_click_time()
            # Save DataFrame to a pickle file
            pbt_path = os.path.join(self.data_dir, "articles_est_pb_time.pkl")
            with open(pbt_path, 'wb') as file:
                # Save dict as a pickle file
                pickle.dump(article2published, file)

            return article2published

    def get_est_pbt_click_time(self):
        """
        In the case for news articles that are not on articles_timeDict_103630.pkl file
        we need to get the publication from another logic. The idea is to estimate the 
        publication time based on the first click happened from the behaviors file. 
        """
        behaviors_path_train = os.path.join(
            self.data_dir, "MIND" + self.dataset_size + "_train/behaviors.tsv"
        )
        behaviors_path_dev = os.path.join(
            self.data_dir, "MIND" + self.dataset_size + "_dev/behaviors.tsv"
        )
        article2published, _, _ = utils.get_article2clicks(behaviors_path_train, behaviors_path_dev)

        return article2published


    def get_bucket(self, time):
        """Get the equivalent time bucket for the specific time 
        object received as an input.
        """
        # Convert the input string to a datetime object
        time = datetime.strptime(time, "%m/%d/%Y %I:%M:%S %p")

        # Define the format to include day, month, year, and hour
        date_hour_format = '%m/%d/%Y %H:00'  # Adjusted to match your input format

        # Start of the hour for the given time
        start_hour = time.replace(minute=0, second=0, microsecond=0)

        # End of the hour (start of the next hour)
        end_hour = start_hour + pd.Timedelta(hours=1)

        # Format the time bucket string
        bucket_str = f"{start_hour.strftime(date_hour_format)} to {end_hour.strftime('%H:00')}"

        # Return both the bucket string and the end_hour as a datetime object
        return bucket_str, end_hour

    def extract_bucket_info(self, row):
        """Extracts information for each impression."""
        article_set = set()  # make sure we don't process any news article twice for same impression
        user_id = row["user"]
        time = row["time"]
        for news_id, clicked in row['impressions_split']:
            if news_id not in article_set:
                article_set.add(news_id)
                time_bucket, time_bucket_end_hour = self.get_bucket(time)
                yield {
                    'time_bucket': time_bucket,
                    'time_bucket_end_hour': time_bucket_end_hour,
                    'news_id': news_id,
                    'clicked': int(clicked),
                    'user_id': user_id
                }

    # Convert the 'impressions' column into a more manageable format
    def parse_impressions(self, impressions):
        """Parse the impressions string into a list of tuples (news_id, clicked)."""
        return [imp.split('-') for imp in impressions]

    def _get_nmb_ptb(self, bucket_info, path_nmb_pkl):
        # Compute num_clicks_ptb and exposures by grouping by 'time_bucket' and 'news_id'
        news_metrics_bucket = bucket_info.groupby(['time_bucket', 'news_id', 'time_bucket_end_hour']).agg(
            num_clicks_ptb=('clicked', 'sum'),
            exposures_ptb=('clicked', 'count')
        ).reset_index()

        # Compute total number of impressions per time bucket
        total_impressions_ptb = bucket_info.groupby(
            'time_bucket').size().reset_index(name='total_impressions_ptb')

        # Merge article metrics with total impressions to get all information together
        news_metrics_bucket = pd.merge(
            news_metrics_bucket, total_impressions_ptb, on='time_bucket')

        # exposure per impression (epi)
        news_metrics_bucket["epi_ptb"] = news_metrics_bucket["exposures_ptb"] / \
            news_metrics_bucket["total_impressions_ptb"]

        # avoidance rate (ar)
        news_metrics_bucket["ar_ptb"] = 1 - news_metrics_bucket["num_clicks_ptb"] / \
            news_metrics_bucket["exposures_ptb"]

        # clicks attained
        total_clicks_ptb = news_metrics_bucket.groupby(
            'time_bucket')['num_clicks_ptb'].sum().reset_index(name='total_clicks_ptb')

        # Merge this information back with the original DataFrame
        news_metrics_bucket = pd.merge(
            news_metrics_bucket, total_clicks_ptb, on='time_bucket')

        # Calculate 'clicks_attained' as a percentage
        news_metrics_bucket['clicks_attained_ptb'] = news_metrics_bucket['num_clicks_ptb'] / \
            news_metrics_bucket['total_clicks_ptb']

        # Replace any potential NaN values with 0 (in case there are time buckets with 0 clicks leading to division by 0)
        news_metrics_bucket['clicks_attained_ptb'] = news_metrics_bucket['clicks_attained_ptb'].fillna(
            0)

        # Save news metrics bucket into csv and pickle     
        news_metrics_bucket.to_pickle(path_nmb_pkl)
        log.info("(ptb) News metric bucket file created!")

        return news_metrics_bucket

    def _get_nmb_acc(self, bucket_info, path_nmb_acc):
        # --- Compute ptb logic first
        # Compute num_clicks and exposures by grouping by 'time_bucket' and 'news_id'
        news_metrics_bucket = bucket_info.groupby(['time_bucket', 'news_id', 'time_bucket_end_hour']).agg(
            num_clicks_ptb=('clicked', 'sum'),
            exposures_ptb=('clicked', 'count')
        ).reset_index()

        # Compute total number of impressions per time bucket
        total_impressions_ptb = bucket_info.groupby(
            'time_bucket').size().reset_index(name='total_impressions_ptb')

        # Merge to get the total impressions per time bucket alongside the news_metrics
        news_metrics_bucket = pd.merge(
            news_metrics_bucket, total_impressions_ptb, on='time_bucket')

        # --- Compute acc logic now
        # Sort the DataFrame by 'news_id' and 'time_bucket' to ensure correct order for cumulative sum
        news_metrics_bucket.sort_values(
            by=['news_id', 'time_bucket'], inplace=True)

        # Calculate cumulative sums for 'num_clicks_ptb' and 'exposures_ptb' to get 'num_clicks_acc' and 'exposures_acc'
        news_metrics_bucket['num_clicks_acc'] = news_metrics_bucket.groupby('news_id')[
            'num_clicks_ptb'].cumsum()
        news_metrics_bucket['exposures_acc'] = news_metrics_bucket.groupby('news_id')[
            'exposures_ptb'].cumsum()

        # Compute cumulative total impressions as well
        news_metrics_bucket['total_impressions_acc'] = news_metrics_bucket.groupby(
            'news_id')['total_impressions_ptb'].cumsum()

        # Remove temporary per-time-bucket (ptb) metrics
        news_metrics_bucket.drop(
            ['num_clicks_ptb', 'exposures_ptb', 'total_impressions_ptb'], axis=1, inplace=True)

        # Exposure Per Impression (EPI)
        news_metrics_bucket["epi_acc"] = news_metrics_bucket["exposures_acc"] / \
            news_metrics_bucket["total_impressions_acc"]

        # Avoidance Rate (AR)
        news_metrics_bucket["ar_acc"] = 1 - news_metrics_bucket["num_clicks_acc"] / \
            news_metrics_bucket["exposures_acc"]

        # Clicks attained
        total_clicks_acc = news_metrics_bucket.groupby(
            'time_bucket')['num_clicks_acc'].sum().reset_index(name='total_clicks_acc')

        # Merge this information back with the original DataFrame
        news_metrics_bucket = pd.merge(
            news_metrics_bucket, total_clicks_acc, on='time_bucket')

        # Calculate 'clicks_attained' as a percentage
        news_metrics_bucket['clicks_attained_acc'] = news_metrics_bucket['num_clicks_acc'] / \
            news_metrics_bucket['total_clicks_acc']

        # Replace any potential NaN values with 0 (in case there are time buckets with 0 clicks leading to division by 0)
        news_metrics_bucket['clicks_attained_acc'] = news_metrics_bucket['clicks_attained_acc'].fillna(
            0)

        # Save news metrics bucket into csv and pickle
        news_metrics_bucket.to_pickle(path_nmb_acc)
        log.info("(acc) News metric bucket file created!")

        return news_metrics_bucket

    def _load_news_metrics_bucket(self):
        """
        News Metric Bucket - nmb

        Get a dataframe that reports news article news_metrics_bucket per time bucket.

        This dataframe needs to be calculated considering both train and dev dataframes.
        We're simulating a near real time information.

            time_bucket                 time                news_id          num_click       num_exposure    num_impressions percent_clicks_attained avoidance_rate
        0   09/11/2019 00:00 to 01:00   2019-11-09 00:00:19 N15134  0           1               31              0.0                     1.0
        1   09/11/2019 00:00 to 01:00   2019-11-09 00:00:19 N16560  0           1               31              0.0                     1.0
        ...
        """
        path_folder = os.path.join(self.data_dir, "MINDmetrics_bucket")
        # Ensure the folder exists
        os.makedirs(path_folder, exist_ok=True)

        path_bucket = os.path.join(path_folder, "mind_news_bucket.tsv")
        path_nmb_pkl_acc = os.path.join(
            path_folder, "mind_news_metrics_bucket_acc.pkl")
        path_nmb_pkl_ptb = os.path.join(
            path_folder, "mind_news_metrics_bucket_ptb.pkl") # TODO: add this file as an option to the acc 

        path_behaviors_train = os.path.join(
            self.data_dir, "MIND" + self.dataset_size + "_train" + "/behaviors.tsv"
        )
        path_behaviors_dev = os.path.join(
            self.data_dir, "MIND" + self.dataset_size + "_dev" + "/behaviors.tsv"
        )

        if not file_utils.check_integrity(path_nmb_pkl_acc):
            log.info("Creating news metric bucket file...")
            # Load train behaviors
            behaviors_train = pd.read_table(
                filepath_or_buffer=path_behaviors_train,
                header=None,
                index_col=["impression_id"],
                names=["impression_id", "user", "time",
                       "clicked_news", "impressions"]
            )

            # Load dev behaviors
            behaviors_dev = pd.read_table(
                filepath_or_buffer=path_behaviors_dev,
                header=None,
                index_col=["impression_id"],
                names=["impression_id", "user", "time",
                       "clicked_news", "impressions"]
            )

            # Join the two behaviors
            behaviors = pd.concat([behaviors_train, behaviors_dev])
            behaviors.clicked_news = behaviors.clicked_news.fillna(" ")
            behaviors.impressions = behaviors.impressions.str.split()

            # Let's split impressions column to make it easier
            behaviors['impressions_split'] = behaviors['impressions'].apply(
                self.parse_impressions)

            # Apply the function and create a new DataFrame
            bucket_info = pd.DataFrame(
                [info for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)) for info in self.extract_bucket_info(row)]
            )

            # Save bucket info
            file_utils.to_tsv(bucket_info, path_bucket)
            log.info("News bucket info file created!")

            # --- Metrics computed using per time bucket (ptb) logic
            news_metrics_bucket_ptb = self._get_nmb_ptb(
                bucket_info, path_nmb_pkl_ptb)

            # --- Metrics computed using accumulative logic (acc)
            news_metrics_bucket_acc = self._get_nmb_acc(
                bucket_info, path_nmb_pkl_acc)
        else:
            log.info("News metric bucket file already created!")
            news_metrics_bucket_acc = pd.read_pickle(path_nmb_pkl_acc)
            news_metrics_bucket_ptb = pd.read_pickle(path_nmb_pkl_ptb)

        return news_metrics_bucket_ptb, news_metrics_bucket_acc

    def _get_ctr(self, news_metrics_bucket: pd.DataFrame, news_id: str, time: object, is_history: bool = True, article2published: Optional[object] = None) -> int:
        """
        Receives time and news article id and returns the CTR of this
        news article for that specific time. Optimized with indexing.

        NOTE: 1)edgecase:
            Handles edge case for news article recency when recency equals zero. A recency value of zero
            indicates that the article was published no more than a second or a minute prior to being
            clicked by the user. For consistency and to avoid dealing with imprecisely recorded publish
            times, recency is assumed to be a minimum of 10 minutes in such cases.
        """
        recency = None
        if not is_history:
            if isinstance(time, str):
                time_bhv = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            else:
                time_bhv = time.replace(tzinfo=timezone.utc)
            time_pub = article2published[news_id].replace(tzinfo=timezone.utc)

            recency = (time_bhv - time_pub).total_seconds()
            # 1) edgecase
            if recency <= 0:
                recency = timedelta(minutes=10).total_seconds()
            recency = recency / 3600

        try:
            filtered = news_metrics_bucket.loc[news_metrics_bucket["news_id"] == news_id]

            # Now, filtered is a subset of the DataFrame where news_id matches.
            # Next, find the first entry where 'time_bucket_end_hour' <= time.
            most_recent = filtered[
                filtered['time_bucket_end_hour'] <= time
            ].iloc[0]

            if is_history:
                # history column does not have recency only number of clicks acc
                return news_id, most_recent['num_clicks_acc']

            # only candidates column have recency
            return news_id, most_recent['num_clicks_acc'], recency

        except Exception as err:
            # KeyError if the news_id is not in the index, IndexError if no rows match the time condition.
            # Return 0 since we have no information about news articles clicks
            # print(f"**: {err}")
            if not is_history:
                return news_id, 0, recency
            else:
                return news_id, 0

    def _ctr_per_row(self, news_metrics_bucket: pd.DataFrame, row: any, article2published:any) -> any:
        """
        1) Convert the history columns from the format below
        [N57021, N4747, N9803, N45994, N24346, N46185]
        to the following format
        [(N57021, 23), (N4747, 10), (N9803, 5), (N45994, 3), (N24346, 1), (N46185, 11)]
        Where the tuple consists of (news_id, ctr). 

        2) Convert the candidates columns from the format below
        [N57021, N4747, N9803, N45994, N24346, N46185]
        to the following format
        [(N57021, 23, time_obj), (N4747, 10, time_obj), (N9803, 5, time_obj), (N45994, 3, time_obj), (N24346, 1, time_obj), (N46185, 11, time_obj)]
        Where the tuple consists of (news_id, ctr, time_obj).
        Where time_obj is the how long this news article has been activate - "recency". 

        Parameters:
            - is_history: check if the method is for history or for candidate.
        """

        # Get ctr per news article in 'history' column
        row['history_ctr'] = [self._get_ctr(
            news_metrics_bucket, nid, row['time']) for nid in row['history']]

        # Get ctr per news article in 'candidates' column
        row['candidates_ctr'] = [self._get_ctr(
            news_metrics_bucket, nid, row['time'], is_history=False, article2published=article2published) for nid in row['candidates']
        ]

        return row
