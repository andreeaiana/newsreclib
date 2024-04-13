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
            log.info('Loading news articles publish time...')
            article2published = self.get_est_publish_time()
            log.info('News articles publish time loaded.')
            acc = True # TODO: add ptb configuration
            if acc:
                # Parse behaviors to add ctr information
                behaviors = self._get_ctr(article2published=article2published, behaviors=behaviors, news_metrics_bucket=news_metrics_bucket_acc)

            else:
                # Parse behaviors to add ctr information
                behaviors = self._get_ctr(article2published=article2published, behaviors=behaviors, news_metrics_bucket=news_metrics_bucket_ptb)

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
            news["abstract"] = news["abstract"].fillna("")
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
            news["abstract"] = news["abstract"].fillna("")
            news["title_entities"] = news["title_entities"].fillna("[]")
            news["abstract_entities"] = news["abstract_entities"].fillna("[]")

            # add estimated publish time column
            log.info('Loading news articles publish time...')
            article2published = self.get_est_publish_time(news['nid'].unique())
            log.info('News articles publish time loaded.')
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
                news = news.drop(columns=["sentiment_preds"])
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
        Retrieves the estimated publish time for news articles specifically for the MIND dataset, 
        which lacks direct publish time data. This function integrates data from two pickle files 
        to provide comprehensive publish time estimates.

        The first pickle file is generated using the utility method `get_article2clicks` from the 
        `utils` module, which outputs 'articles_est_pb_time.pkl'. This file estimates publish times 
        based on article clicks.

        The second file, 'articles_timeDict_103630.pkl', is provided by the research detailed in 
        the paper "Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based 
        News Recommendation". However, this file occasionally lists publish times that post-date 
        the actual article clicks, which is logically inconsistent.

        By merging data from both sources, this function aims to correct these inconsistencies 
        and provide a more accurate estimation of article publish times.

        Parameters:
            unique_ids (set): A set of news article IDs for which publish times are to be estimated.

        Returns:
            dict: A dictionary mapping news article IDs to their estimated publish times.

        References:
            - [Session-based News Recommendation GitHub Repository](https://github.com/summmeer/session-based-news-recommendation)
            - [Related GitHub Issue Discussion](https://github.com/summmeer/session-based-news-recommendation/issues/6#issuecomment-1233830425)

        Example Usage:
            >>> unique_ids = {'N1001', 'N1002', 'N1003'}
            >>> publish_times = get_estimated_publish_time(unique_ids)
            >>> print(publish_times)
        """
        
        # -- Check if we already have this file
        upt_path = os.path.join(self.data_dir, "updated_articles_publish_time.pkl")
        if file_utils.check_integrity(upt_path):
            return pd.read_pickle(upt_path)
        else:
            est_pb_time = os.path.join(self.data_dir, "articles_est_pb_time.pkl")
            if file_utils.check_integrity(est_pb_time):
                articles_est_pb = pd.read_pickle(est_pb_time)
            else:
                article2published = self.get_est_pbt_click_time()
                # Save DataFrame to a pickle file
                pbt_path = os.path.join(self.data_dir, "articles_est_pb_time.pkl")
                with open(pbt_path, 'wb') as file:
                    # Save dict as a pickle file
                    pickle.dump(article2published, file)

                articles_est_pb = article2published
        
            # -- load the other pickle file
            time_dict_path = os.path.join(self.data_dir, "articles_timeDict_103630.pkl")
            articles_time_dict = pd.read_pickle(time_dict_path)

            # Update articles_est_pb with missing data from articles_time_dict
            for key, value in articles_time_dict.items():
                if key not in articles_est_pb:
                    articles_est_pb[key] = value
                
            # -- Save the updated dictionary to a new pickle file
            with open(upt_path, 'wb') as handle:
                log.info('news articles publish time saved.')
                pickle.dump(articles_est_pb, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return articles_est_pb

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

        # Merge this information back with the original DataFrame
        news_metrics_bucket = pd.merge(
            news_metrics_bucket, total_clicks_ptb, on='time_bucket')

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
        news_metrics_bucket = news_metrics_bucket.sort_values(by=['news_id', 'time_bucket'])

        # Calculate cumulative sums for 'num_clicks_ptb' and 'exposures_ptb' to get 'num_clicks_acc' and 'exposures_acc'
        news_metrics_bucket['num_clicks_acc'] = news_metrics_bucket.groupby('news_id')[
            'num_clicks_ptb'].cumsum()
        news_metrics_bucket['exposures_acc'] = news_metrics_bucket.groupby('news_id')[
            'exposures_ptb'].cumsum()

        # Compute cumulative total impressions as well
        news_metrics_bucket['total_impressions_acc'] = news_metrics_bucket.groupby(
            'news_id')['total_impressions_ptb'].cumsum()

        # Remove temporary per-time-bucket (ptb) metrics
        news_metrics_bucket = news_metrics_bucket.drop(['num_clicks_ptb', 'exposures_ptb', 'total_impressions_ptb'], axis=1)

        # Merge this information back with the original DataFrame
        news_metrics_bucket = pd.merge(
            news_metrics_bucket, total_clicks_acc, on='time_bucket')

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
        """
        path_folder = os.path.join(self.data_dir, "MIND" + "_" + self.dataset_size + "_metrics_bucket")
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

    def aux_lst_f(self, series):
        """
        Define a custom aggregation function to create tuples of (history_news_id, num_clicks_acc)
        """
        return list(series)
    
    def _get_history_ctr(self, behaviors: pd.DataFrame, news_metrics_bucket: pd.DataFrame) -> any:
        # Explode the 'history' column to individual rows for easier processing
        bhv_hist_explode = behaviors.explode('history')
        bhv_hist_explode = bhv_hist_explode.rename(columns={'history': 'history_news_id'})
        bhv_hist_explode['time'] = pd.to_datetime(bhv_hist_explode['time'])

        # Filter the metrics bucket to include only news_ids present in bhv_hist_explode for efficiency
        unique_ids_metrics_bucket = news_metrics_bucket['news_id'].unique().tolist()
        bhv_hist_explode_filter = bhv_hist_explode[bhv_hist_explode['history_news_id'].isin(unique_ids_metrics_bucket)]

        # Merge filtered behaviors with metrics on news_id
        merged_df = pd.merge(bhv_hist_explode_filter, news_metrics_bucket, left_on='history_news_id', right_on='news_id', how='left')

        # Select entries where the behavioral event time is within the time bucket range
        valid_entries = merged_df[
            (merged_df['time'] >= merged_df['time_bucket_start_hour']) & 
            (merged_df['time'] <= merged_df['time_bucket_end_hour'])
        ]
        valid_entries = valid_entries.sort_values(by='time', ascending=False)
        final_df = valid_entries.drop_duplicates(subset=['history_news_id', 'time'], keep='first')

        # Reintegrate CTR data back to behaviors, filling gaps with zero where no data was found
        bhv_hist_ctr_acc = pd.merge(
            bhv_hist_explode_filter, 
            final_df[['history_news_id', 'time', 'num_clicks_acc']], 
            on=['history_news_id', 'time'], 
            how='left'
        )
        bhv_hist_ctr_acc['num_clicks_acc'] = bhv_hist_ctr_acc['num_clicks_acc'].fillna(0)
        bhv_hist_ctr_acc = bhv_hist_ctr_acc.drop_duplicates(subset=['history_news_id', 'time', 'num_clicks_acc'])

        # Reaggregate to match the original data granularity and form a history column with CTR data
        final_df = pd.merge(
            bhv_hist_explode,
            bhv_hist_ctr_acc[['history_news_id', 'impid', 'user', 'time', 'num_clicks_acc']],
            on=['history_news_id', 'impid', 'user', 'time'],
            how='left'
        )
        final_df['num_clicks_acc'] = final_df['num_clicks_acc'].fillna(0)
        result_df = final_df.groupby(['impid', 'user', 'time']).agg({
            'history_news_id': list,
            'num_clicks_acc': self.aux_lst_f
        }).reset_index()
        result_df['history_ctr'] = result_df.apply(lambda x: list(zip(x['history_news_id'], x['num_clicks_acc'])), axis=1)
        result_df = result_df.rename(columns={'history_news_id': 'history'})

         # Validate that merged data matches original behaviors data
        behaviors['time'] = pd.to_datetime(behaviors['time'])
        result_df['time'] = pd.to_datetime(result_df['time'])
        behaviors_ = pd.merge(behaviors, result_df, on=['impid', 'user', 'time'], how='inner')
        diff_mask = behaviors_['history_x'] != behaviors_['history_y']
        different_indexes = behaviors_.index[diff_mask].tolist()

        # Ensure no discrepancies exist
        assert len(different_indexes) == 0

        # Drop the 'history_y' column
        behaviors_ = behaviors_.drop(columns=['history_y'])

        # Rename 'history_x' to 'history'
        behaviors_ = behaviors_.rename(columns={'history_x': 'history'})
        behaviors_ = behaviors_.rename(columns={'num_clicks_acc': 'hist_num_clicks_acc'})

        return behaviors_


    def _get_candidate_ctr(self, behaviors: pd.DataFrame, news_metrics_bucket: pd.DataFrame, article2published: any) -> any:
        # Map publishing times

        # Explode the 'candidates' column to individual rows for easier processing
        bhv_cand_explode = behaviors.explode('candidates')
        bhv_cand_explode = bhv_cand_explode.rename(columns={'candidates': 'cand_news_id'})
        bhv_cand_explode['pb_time'] = bhv_cand_explode['cand_news_id'].map(article2published)
        bhv_cand_explode['time'] = pd.to_datetime(bhv_cand_explode['time'])

        # Filter the metrics bucket to include only news_ids present in bhv_cand_explode for efficiency
        unique_ids_metrics_bucket = news_metrics_bucket['news_id'].unique().tolist()
        bhv_cand_explode_filter = bhv_cand_explode[bhv_cand_explode['cand_news_id'].isin(unique_ids_metrics_bucket)]

        # Merge filtered behaviors with metrics on news_id
        merged_df = pd.merge(bhv_cand_explode_filter, news_metrics_bucket, left_on='cand_news_id', right_on='news_id', how='left')

        # Select entries where the behavioral event time is within the time bucket range
        valid_entries = merged_df[
            (merged_df['time'] >= merged_df['time_bucket_start_hour']) & 
            (merged_df['time'] <= merged_df['time_bucket_end_hour'])
        ]
        valid_entries = valid_entries.sort_values(by='time', ascending=False)
        final_df = valid_entries.drop_duplicates(subset=['cand_news_id', 'time'], keep='first')

        # Reintegrate CTR data back to behaviors, filling gaps with zero where no data was found
        bhv_cand_ctr_acc = pd.merge(
            bhv_cand_explode_filter, 
            final_df[['cand_news_id', 'time', 'num_clicks_acc']], 
            on=['cand_news_id', 'time'], 
            how='left'
        )
        bhv_cand_ctr_acc['num_clicks_acc'] = bhv_cand_ctr_acc['num_clicks_acc'].fillna(0)
        bhv_cand_ctr_acc = bhv_cand_ctr_acc.drop_duplicates(subset=['cand_news_id', 'time', 'num_clicks_acc'])

        # Reaggregate to match the original data granularity and form a cand column with CTR data
        final_df = pd.merge(
            bhv_cand_explode,
            bhv_cand_ctr_acc[['cand_news_id', 'impid', 'user', 'time', 'num_clicks_acc']],
            on=['cand_news_id', 'impid', 'user', 'time'],
            how='left'
        )
        final_df['num_clicks_acc'] = final_df['num_clicks_acc'].fillna(0)

        result_df = final_df.groupby(['impid', 'user', 'time']).agg({
            'cand_news_id': list,
            'num_clicks_acc': self.aux_lst_f,
            'pb_time': self.aux_lst_f,
        }).reset_index()
        result_df['candidates_ctr'] = result_df.apply(lambda x: list(zip(x['cand_news_id'], x['num_clicks_acc'], x['pb_time'])), axis=1)
        result_df = result_df.rename(columns={'cand_news_id': 'candidates'})
        result_df = result_df.rename(columns={'pb_time': 'cand_pb_time'})
        result_df = result_df.rename(columns={'num_clicks_acc': 'cand_num_clicks_acc'})

        # Parse to datetime format
        behaviors['time'] = pd.to_datetime(behaviors['time'])
        result_df['time'] = pd.to_datetime(result_df['time'])

        # Validate that merged data matches original behaviors data
        behaviors_ = pd.merge(behaviors, result_df, on=['impid', 'user', 'time'], how='inner')
        diff_mask = behaviors_['candidates_x'] != behaviors_['candidates_y']
        different_indexes = behaviors_.index[diff_mask].tolist()

        # Ensure no discrepancies exist
        assert len(different_indexes) == 0

        # Drop the 'candidates_y' column
        behaviors_ = behaviors_.drop(columns=['candidates_y'])

        # Rename 'cand_x' to 'cand'
        behaviors_ = behaviors_.rename(columns={'candidates_x': 'candidates'})
        behaviors_ = behaviors_.rename(columns={'num_clicks_acc': 'cand_num_clicks_acc'})

        return behaviors_

    def _get_ctr(self, behaviors: pd.DataFrame, news_metrics_bucket: pd.DataFrame, article2published: any) -> any:
        """
        Calculate the CTR for each news article over its respective time buckets from the news_metrics_bucket DataFrame.
        It matches news articles by ID and checks that the behavioral event time falls within the designated time buckets.
        
        Parameters:
            behaviors (pd.DataFrame): DataFrame containing user behavior data.
            news_metrics_bucket (pd.DataFrame): DataFrame with metrics for each news article over specific time buckets.
            row (any): Unused in this snippet, but typically used for row-specific operations.
            article2published (any): Unused in this snippet, could be used for mapping articles to published info.

        Returns:
            pd.DataFrame: Behaviors DataFrame enriched with the CTR information and checks for data consistency.

        Example usage:
            - Input DataFrame row: {'news_id': 'N3128', 'time_bucket': '11/13/2019 14:00 to 15:00', 'num_clicks_acc': 152}
            - Output: CTR values merged back into the original behaviors DataFrame.
        """
        # Assign impid to the index
        behaviors = behaviors.reset_index(names='impid')

        # Split the 'time_bucket' into two separate columns for start and end times
        time_bounds = news_metrics_bucket['time_bucket'].str.split(' to ', expand=True)
        news_metrics_bucket['time_bucket_start_hour'] = time_bounds[0]

        # Convert the new start and end time columns to datetime
        news_metrics_bucket['time_bucket_start_hour'] = pd.to_datetime(news_metrics_bucket['time_bucket_start_hour'], format='%m/%d/%Y %H:%M')
        news_metrics_bucket['time_bucket_end_hour'] = pd.to_datetime(news_metrics_bucket['time_bucket_end_hour'], format='%m/%d/%Y %H:%M')

        # Get CTR for history column
        df_history_ctr = self._get_history_ctr(behaviors=behaviors, news_metrics_bucket=news_metrics_bucket)

        # Get CTR for candidates column
        df_candidate_ctr = self._get_candidate_ctr(behaviors=behaviors, news_metrics_bucket=news_metrics_bucket, article2published=article2published)

        # Join informations
        behaviors = pd.merge(
            behaviors, 
            df_history_ctr[['impid', 'history_ctr', 'hist_num_clicks_acc']], 
            on='impid', 
            how='left'
        )
        behaviors = pd.merge(
            behaviors, 
            df_candidate_ctr[['impid', 'candidates_ctr', 'cand_num_clicks_acc']], 
            on='impid', 
            how='left'
        )

        return behaviors

       

