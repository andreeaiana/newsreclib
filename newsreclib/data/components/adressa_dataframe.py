import json
import os
import tarfile
from ast import literal_eval
from collections import Counter, defaultdict
from datetime import datetime
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

import newsreclib.data.components.data_utils as data_utils
import newsreclib.data.components.file_utils as file_utils
from newsreclib import utils
from newsreclib.data.components.adressa_user_info import UserInfo

tqdm.pandas()

log = utils.get_pylogger(__name__)


class AdressaDataFrame(Dataset):
    """Creates a dataframe for the MIND dataset.

    Additionally:
        - Downloads the dataset for the specified size.
        - Downloads pretrained embeddings.
        - Parses the news and behaviors data.
        - Annotates the news with additional aspects (e.g., `sentiment`).
        - Split the behaviors into `train` and `validation` sets by time.

    Attributes:
        seed:
            Seed for random states.
        dataset_size:
            A string indicating the type of the dataset. Choose between `one_week` and `three_month`.
        dataset_url:
            Dictionary of URLs for downloading the dataset for the specified `dataset_size`.
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
        use_plm:
            If ``True``, it will process the data for a petrained language model (PLM) in the news encoder. If ``False``, it will tokenize the news title and abstract to be used initialized with pretrained word embeddings.
        use_pretrained_categ_embeddings:
            Whether to initialize category embeddings with pretrained word embeddings.
        categ_embed_dim:
            Dimensionality of category embeddings.
        word_embed_dim:
            Dimensionality of word embeddings.
        sentiment_annotator:
            The sentiment annotator module used.
        train_date_split:
            A string with the date before which click behaviors are included in the history of a user.
        test_date_split:
            A string with the date after which click behaviors are included in the test set.
        neg_num:
            Number of negatives for constructing the impression log of a user.
        user_dev_size:
            The proportion of the training set to be used for validation.
        train:
            If ``True``, the data will be processed and used for training. If ``False``, it will be processed and used for validation or testing.
        validation:
            If ``True`` and `train` is also``True``, the data will be processed and used for validation. If ``False`` and `train` is `True``, the data will be processed ad used for training. If ``False`` and `train` is `False``, the data will be processed and used for testing.
        download:
            Whether to download the dataset, if not already downloaded.
    """

    def __init__(
        self,
        seed: int,
        dataset_size: str,
        dataset_url: DictConfig,
        data_dir: str,
        dataset_attributes: List[str],
        id2index_filenames: DictConfig,
        pretrained_embeddings_url: Optional[str],
        word_embeddings_dirname: Optional[str],
        word_embeddings_fpath: Optional[str],
        use_plm: bool,
        use_pretrained_categ_embeddings: bool,
        word_embed_dim: Optional[int],
        categ_embed_dim: Optional[int],
        sentiment_annotator: nn.Module,
        train_date_split: str,
        test_date_split: str,
        neg_num: int,
        user_dev_size: float,
        train: bool,
        validation: bool,
        download: bool,
    ) -> None:
        super().__init__()

        self.seed = seed
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

        self.sentiment_annotator = sentiment_annotator

        self.train_date_split = train_date_split
        self.test_date_split = test_date_split
        self.neg_num = neg_num
        self.user_dev_size = user_dev_size

        self.validation = validation

        if train:
            if not self.validation:
                self.data_split = "train"
            else:
                self.data_split = "dev"
        else:
            self.data_split = "test"

        self.dst_dir = os.path.join(self.data_dir, "Adressa_" + self.dataset_size)
        self.dst_dir_stage = os.path.join(
            self.data_dir, "Adressa_" + self.dataset_size, self.data_split
        )
        if not os.path.isdir(self.dst_dir_stage):
            os.makedirs(self.dst_dir_stage)

        self.adressa_gzip_filename = "Adressa_" + self.dataset_size + ".tar.gz"

        if download:
            url = dataset_url[dataset_size]
            log.info(f"Downloading Adressa {self.dataset_size} dataset from {url}.")
            try:
                data_utils.download_and_extract_dataset(
                    data_dir=self.data_dir,
                    url=url,
                    filename=self.adressa_gzip_filename,
                    extract_compressed=False,
                    dst_dir=None,
                    clean_archive=False,
                )
            except requests.exceptions.SSLError as e:
                log.warn(
                    f"Downloading Adressa {self.dataset_size} dataset from {url} failed due to error: {e}. Download the dataset manually and try again."
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
        parsed_news_file = os.path.join(self.dst_dir_stage, "parsed_news.tsv")

        if file_utils.check_integrity(parsed_news_file):
            # news already parsed
            log.info(f"News already parsed. Loading from {parsed_news_file}.")

            attributes2convert = ["title_entities"]
            if not self.use_plm:
                attributes2convert.extend(["tokenized_title"])
            news = pd.read_table(
                filepath_or_buffer=parsed_news_file,
                converters={attribute: literal_eval for attribute in attributes2convert},
            )
        else:
            log.info("News not parsed. Loading and parsing raw data.")

            raw_news_filepath = os.path.join(self.dst_dir_stage, "news.tsv")

            if not file_utils.check_integrity(raw_news_filepath):
                log.info("Compressed files not processed. Reading news data.")
                news_title, news_category, news_subcategory, nid2index = self._process_news_files(
                    os.path.join(self.data_dir, self.adressa_gzip_filename)
                )
                self._write_news_files(news_title, news_category, news_subcategory, nid2index)

                news_title_df = pd.DataFrame(news_title.items(), columns=["id", "title"])
                file_utils.to_tsv(news_title_df, os.path.join(self.dst_dir, "news_title.tsv"))

                nid2index_df = pd.DataFrame(nid2index.items(), columns=["id", "index"])
                file_utils.to_tsv(
                    nid2index_df, os.path.join(self.dst_dir, self.id2index_filenames["nid2index"])
                )

            log.info("Processing data.")
            columns_names = ["nid", "category", "subcategory", "title"]
            news = pd.read_table(
                filepath_or_buffer=raw_news_filepath,
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )

            if not self.use_plm:
                word2index_fpath = os.path.join(
                    self.dst_dir,
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

            categ2index_fpath = os.path.join(
                self.dst_dir,
                self.id2index_filenames["categ2index"],
            )
            subcateg2index_fpath = os.path.join(
                self.dst_dir,
                self.id2index_filenames["subcateg2index"],
            )

            if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
                sentiment2index_fpath = os.path.join(
                    self.dst_dir,
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

            if not self.use_plm:
                # tokenize text
                news["tokenized_title"] = news["title"].progress_apply(data_utils.word_tokenize)

                # construct word2index map
                log.info("Constructing word2index map.")
                word_cnt = Counter()
                for idx in tqdm(news.index.tolist()):
                    word_cnt.update(news.loc[idx]["tokenized_title"])
                word2index = {k: v + 1 for k, v in zip(word_cnt, range(len(word_cnt)))}
                log.info(f"Saving word2index map of size {len(word2index)} in {word2index_fpath}")
                file_utils.to_tsv(
                    df=pd.DataFrame(word2index.items(), columns=["word", "index"]),
                    fpath=word2index_fpath,
                )

            # construct category2index
            log.info("Constructing categ2index map.")
            news_category = news["category"].drop_duplicates().reset_index(drop=True)
            categ2index = {v: k + 1 for k, v in news_category.to_dict().items()}
            log.info(f"Saving categ2index map of size {len(categ2index)} in {categ2index_fpath}")
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
                news_sentiment = news["sentiment_class"].drop_duplicates().reset_index(drop=True)
                sentiment2index = {v: k + 1 for k, v in news_sentiment.to_dict().items()}
                log.info(
                    f"Saving sentiment2index map of size {len(sentiment2index)} in {sentiment2index_fpath}"
                )
                file_utils.to_tsv(
                    df=pd.DataFrame(sentiment2index.items(), columns=["sentiment", "index"]),
                    fpath=sentiment2index_fpath,
                )

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

            # parse news
            log.info("Parsing news")
            if not self.use_plm:
                news["tokenized_title"] = news["title"].progress_apply(data_utils.word_tokenize)
                news["tokenized_title"] = news["tokenized_title"].progress_apply(
                    lambda title: [word2index.get(x, 0) for x in title]
                )

            news["category_class"] = news["category"].progress_apply(
                lambda category: categ2index.get(category, 0)
            )
            news["subcategory_class"] = news["subcategory"].progress_apply(
                lambda subcategory: subcateg2index.get(subcategory, 0)
            )

            if "sentiment_class" or "sentiment_score" in self.dataset_attributes:
                news["sentiment_class"] = news["sentiment_class"].progress_apply(
                    lambda sentiment: sentiment2index.get(sentiment, 0)
                )

            # cache parsed news
            for stage in ["train", "dev", "test"]:
                stage_dir = os.path.join(self.data_dir, "Adressa_" + self.dataset_size, stage)
                parsed_news_filepath = os.path.join(stage_dir, "parsed_news.tsv")
                log.info(f"Caching parsed news of size {len(news)} to {parsed_news_file}.")
                file_utils.to_tsv(news, parsed_news_filepath)

        news = news.set_index("nid", drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """Loads the parsed user behaviors. If not already parsed, loads and parses the raw
        behavior data.

        Returns:
            Parsed and split user behavior data.
        """
        parsed_behaviors_file = os.path.join(
            self.dst_dir_stage, "parsed_behaviors_" + str(self.seed) + ".tsv"
        )

        if file_utils.check_integrity(parsed_behaviors_file):
            # behaviors data already parsed
            log.info(f"User behaviors data already parsed. Loading from {parsed_behaviors_file}.")
            behaviors = pd.read_table(
                filepath_or_buffer=parsed_behaviors_file,
                converters={
                    "history": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "candidates": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "labels": lambda x: list(map(int, x.strip("[]").split(", "))),
                },
            )
            behaviors["history"] = behaviors["history"].apply(lambda x: [int(e) for e in x])
            behaviors["candidates"] = behaviors["candidates"].apply(lambda x: [int(e) for e in x])

        else:
            log.info("User behaviors data not parsed. Loading and preprocessing raw data.")

            raw_behaviors_filepath = os.path.join(
                self.dst_dir, "behaviors_" + str(self.seed) + ".tsv"
            )

            if not file_utils.check_integrity(raw_behaviors_filepath):
                news_title = file_utils.load_idx_map_as_dict(
                    os.path.join(self.dst_dir, "news_title.tsv")
                )
                nid2index = file_utils.load_idx_map_as_dict(
                    os.path.join(self.dst_dir, self.id2index_filenames["nid2index"])
                )

                log.info("Compressed files not processed. Reading behavior data.")
                uid2index, user_info = self._process_users(
                    os.path.join(self.data_dir, self.adressa_gzip_filename), nid2index
                )

                log.info("Sorting user behavior data chronologically.")
                for uid in tqdm(user_info):
                    user_info[uid].sort_click()

                log.info("Constructing behaviors.")
                self.train_lines = []
                self.test_lines = []
                for uindex in tqdm(user_info):
                    uinfo = user_info[uindex]
                    train_news = uinfo.train_news
                    test_news = uinfo.test_news
                    hist_news = uinfo.hist_news
                    self._construct_behaviors(uindex, hist_news, train_news, test_news, news_title)

                shuffle(self.train_lines)
                shuffle(self.test_lines)

                test_split_lines, dev_split_lines = train_test_split(
                    self.test_lines, test_size=self.user_dev_size, random_state=self.seed
                )

                self._write_behavior_files(self.train_lines, "train")
                self._write_behavior_files(dev_split_lines, "dev")
                self._write_behavior_files(test_split_lines, "test")

            log.info("Compressed files read. Processing data.")
            columns_names = ["uid", "history", "impressions"]
            behaviors = pd.read_table(
                filepath_or_buffer=os.path.join(
                    self.dst_dir_stage, "behaviors_" + str(self.seed) + ".tsv"
                ),
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
                low_memory=False,
            )

            behaviors["history"] = behaviors["history"].fillna("").str.split()
            behaviors["history"] = behaviors["history"].apply(lambda x: [int(e) for e in x])
            behaviors["impressions"] = behaviors["impressions"].str.split()
            behaviors["candidates"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[0]) for impression in x]
            )
            behaviors["labels"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[1]) for impression in x]
            )
            behaviors = behaviors.drop(columns=["impressions"])

            # drop interactions of users without history
            count_interactions = len(behaviors)
            behaviors = behaviors[behaviors["history"].apply(len) > 0]
            dropped_interactions = count_interactions - len(behaviors)
            log.info(f"Removed {dropped_interactions} interactions without user history.")

            behaviors = behaviors.reset_index(drop=True)

            if self.data_split == "train":
                if not self.validation:
                    # compute uid2index map
                    log.info("Constructing uid2index map.")
                    uid2index = {}
                    for idx in tqdm(behaviors.index.tolist()):
                        uid = behaviors.loc[idx]["uid"]
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index) + 1

                    fpath = os.path.join(self.dst_dir, self.id2index_filenames["uid2index"])
                    log.info(f"Saving uid2index map of size {len(uid2index)} in {fpath}")
                    file_utils.to_tsv(
                        df=pd.DataFrame(uid2index.items(), columns=["uid", "index"]), fpath=fpath
                    )
                else:
                    # load uid2index map
                    log.info("Loading uid2index map.")
                    fpath = os.path.join(self.dst_dir, self.id2index_filenames["uid2index"])
                    uid2index = file_utils.load_idx_map_as_dict(fpath)

            else:
                # load uid2index map
                log.info("Loading uid2index map.")
                fpath = os.path.join(self.dst_dir, self.id2index_filenames["uid2index"])
                uid2index = file_utils.load_idx_map_as_dict(fpath)

            # map uid to index
            log.info("Mapping uid to index.")
            behaviors["user"] = behaviors["uid"].apply(lambda x: uid2index.get(x, 0))

            behaviors = behaviors[["uid", "user", "history", "candidates", "labels"]]

            # cache processed data
            log.info(
                f"Caching parsed behaviors of size {len(behaviors)} to {parsed_behaviors_file}."
            )
            file_utils.to_tsv(behaviors, parsed_behaviors_file)

        return behaviors

    def _process_news_files(
        self, filepath
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Processes the news data.

        Adapted from
        https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py
        """
        news_title = {}
        news_category = {}
        news_subcategory = {}

        tar = tarfile.open(filepath, "r:gz", encoding="utf-8")
        files = tar.getmembers()

        for file in tqdm(files):
            f = tar.extractfile(file)
            if f is not None:
                for line in f.readlines():
                    line = line.decode("utf-8")
                    event_dict = json.loads(line.strip("\n"))

                    if "id" in event_dict and "title" in event_dict and "category1" in event_dict:
                        if event_dict["id"] not in news_title:
                            news_title[event_dict["id"]] = event_dict["title"]
                        else:
                            assert news_title[event_dict["id"]] == event_dict["title"]

                        if event_dict["id"] not in news_category:
                            news_category[event_dict["id"]] = event_dict["category1"].split("|")[0]
                        else:
                            assert (
                                news_category[event_dict["id"]]
                                == event_dict["category1"].split("|")[0]
                            )

                        if event_dict["id"] not in news_subcategory:
                            news_subcategory[event_dict["id"]] = event_dict["category1"].split(
                                "|"
                            )[-1]
                        else:
                            assert (
                                news_subcategory[event_dict["id"]]
                                == event_dict["category1"].split("|")[-1]
                            )

        nid2index = {
            k: "N" + str(v) for k, v in zip(news_title.keys(), range(1, len(news_title) + 1))
        }

        return news_title, news_category, news_subcategory, nid2index

    def _write_news_files(
        self,
        news_title: Dict[str, str],
        news_category: Dict[str, str],
        news_subcategory: Dict[str, str],
        nid2index: Dict[str, int],
    ) -> None:
        """Writes news to the file.

        Adapted from
        https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py
        """

        news_lines = []
        for nid in tqdm(news_title):
            nindex = nid2index[nid]
            title = news_title[nid]
            category = news_category[nid]
            subcategory = news_subcategory[nid]
            news_line = "\t".join([str(nindex), category, subcategory, title]) + "\n"
            news_lines.append(news_line)

        for stage in ["train", "dev", "test"]:
            filepath = os.path.join(self.dst_dir, stage)
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
            with open(os.path.join(filepath, "news.tsv"), "w", encoding="utf-8") as f:
                f.writelines(news_lines)

    def _process_users(
        self, filepath: str, nid2index: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[int, Any]]:
        """Processes user behaviors.

        Adapted from
        https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py
        """
        uid2index = {}
        user_info = defaultdict(lambda: UserInfo(self.train_date_split, self.test_date_split))

        tar = tarfile.open(filepath, "r:gz", encoding="utf-8")
        files = tar.getmembers()

        for file in tqdm(files):
            f = tar.extractfile(file)
            if f is not None:
                for line in f.readlines():
                    line = line.decode("utf-8")
                    event_dict = json.loads(line.strip("\n"))

                    if (
                        "id" in event_dict
                        and "title" in event_dict
                        and event_dict["id"] in nid2index
                    ):
                        nindex = nid2index[event_dict["id"]]
                        uid = "U" + str(event_dict["userId"])

                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index)

                        user_index = uid2index[uid]
                        click_time = int(event_dict["time"])
                        if self.dataset_size == "one_week":
                            date = int(file.name[-1])
                        else:
                            date = datetime.utcfromtimestamp(click_time).isocalendar()[1]
                        user_info[user_index].update(nindex, click_time, date)

        return uid2index, user_info

    def _construct_behaviors(self, uindex, hist_news, train_news, test_news, news_title) -> None:
        probs = np.ones(len(news_title) + 1, dtype="float32")
        probs[hist_news] = 0
        probs[train_news] = 0
        probs[test_news] = 0
        probs[0] = 0
        probs /= probs.sum()

        train_hist_news = [str(i) for i in hist_news.tolist()]
        train_hist_line = " ".join(train_hist_news)

        for nindex in train_news:
            neg_cand = np.random.choice(
                len(news_title) + 1, size=self.neg_num, replace=False, p=probs
            ).tolist()
            cand_news = " ".join(
                [f"{str(nindex)}-1"] + [f"{str(nindex)}-0" for nindex in neg_cand]
            )

            train_behavior_line = f"{uindex}\t{train_hist_line}\t{cand_news}\n"
            self.train_lines.append(train_behavior_line)

        test_hist_news = [str(i) for i in hist_news.tolist() + train_news.tolist()]
        test_hist_line = " ".join(test_hist_news)

        for nindex in test_news:
            neg_cand = np.random.choice(
                len(news_title) + 1, size=self.neg_num, replace=False, p=probs
            ).tolist()
            cand_news = " ".join(
                [f"{str(nindex)}-1"] + [f"{str(nindex)}-0" for nindex in neg_cand]
            )

            test_behavior_line = f"{uindex}\t{test_hist_line}\t{cand_news}\n"
            self.test_lines.append(test_behavior_line)

    def _write_behavior_files(self, behavior_lines, stage: str) -> None:
        filepath = os.path.join(self.dst_dir, stage)
        with open(
            os.path.join(filepath, "behaviors_" + str(self.seed) + ".tsv"), "w", encoding="utf-8"
        ) as f:
            f.writelines(behavior_lines)
