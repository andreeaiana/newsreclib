from typing import Any, Dict, List, Optional

import torch.nn as nn
from lightning import LightningDataModule
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from newsreclib.data.components.adressa_dataframe import AdressaDataFrame
from newsreclib.data.components.rec_dataset import (
    DatasetCollate,
    RecommendationDatasetTest,
    RecommendationDatasetTrain,
)


class AdressaRecDataModule(LightningDataModule):
    """Example of LightningDataModule for the MIND dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

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
        max_title_len:
            Maximum title length.
        max_abstract_len:
            Maximum abstract length.
        concatenate_inputs:
            Whether to concatenate inputs (e.g., title and abstract) before feeding them into the news encoder.
        tokenizer_name:
            Name of the tokenizer, if using a pretrained language model in the news encoder.
        tokenizer_use_fast:
            Whether to use a fast tokenizer.
        tokenizer_max_len:
            Maximum length of the tokenizer.
        max_history_len:
            Maximum history length.
        neg_sampling_ratio:
            Number of negatives per positive sample for training.
        batch_size:
            How many samples per batch to load.
        num_workers:
            How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        pin_memory:
             If ``True``, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.
        drop_last:
             Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
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
        categ_embed_dim: Optional[int],
        word_embed_dim: Optional[int],
        sentiment_annotator: nn.Module,
        train_date_split: str,
        test_date_split: str,
        neg_num: int,
        user_dev_size: float,
        max_title_len: int,
        max_abstract_len: int,
        concatenate_inputs: bool,
        tokenizer_name: Optional[str],
        tokenizer_use_fast: Optional[bool],
        tokenizer_max_len: Optional[int],
        max_history_len: int,
        neg_sampling_ratio: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if self.hparams.use_plm:
            assert isinstance(self.hparams.tokenizer_name, str)
            assert isinstance(self.hparams.tokenizer_use_fast, bool)
            assert (
                isinstance(self.hparams.tokenizer_max_len, int)
                and self.hparams.tokenizer_max_len > 0
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name,
                use_fast=self.hparams.tokenizer_use_fast,
                model_max_length=self.hparams.tokenizer_max_len,
            )

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # download train set
        AdressaDataFrame(
            seed=self.hparams.seed,
            dataset_size=self.hparams.dataset_size,
            dataset_url=self.hparams.dataset_url,
            data_dir=self.hparams.data_dir,
            dataset_attributes=self.hparams.dataset_attributes,
            id2index_filenames=self.hparams.id2index_filenames,
            pretrained_embeddings_url=self.hparams.pretrained_embeddings_url,
            word_embeddings_dirname=self.hparams.word_embeddings_dirname,
            word_embeddings_fpath=self.hparams.word_embeddings_fpath,
            use_plm=self.hparams.use_plm,
            use_pretrained_categ_embeddings=self.hparams.use_pretrained_categ_embeddings,
            categ_embed_dim=self.hparams.categ_embed_dim,
            word_embed_dim=self.hparams.word_embed_dim,
            sentiment_annotator=self.hparams.sentiment_annotator,
            train_date_split=self.hparams.train_date_split,
            test_date_split=self.hparams.test_date_split,
            neg_num=self.hparams.neg_num,
            user_dev_size=self.hparams.user_dev_size,
            train=True,
            validation=False,
            download=True,
        )

        # download validation set
        AdressaDataFrame(
            seed=self.hparams.seed,
            dataset_size=self.hparams.dataset_size,
            dataset_url=self.hparams.dataset_url,
            data_dir=self.hparams.data_dir,
            dataset_attributes=self.hparams.dataset_attributes,
            id2index_filenames=self.hparams.id2index_filenames,
            pretrained_embeddings_url=self.hparams.pretrained_embeddings_url,
            word_embeddings_dirname=self.hparams.word_embeddings_dirname,
            word_embeddings_fpath=self.hparams.word_embeddings_fpath,
            use_plm=self.hparams.use_plm,
            use_pretrained_categ_embeddings=self.hparams.use_pretrained_categ_embeddings,
            categ_embed_dim=self.hparams.categ_embed_dim,
            word_embed_dim=self.hparams.word_embed_dim,
            sentiment_annotator=self.hparams.sentiment_annotator,
            train_date_split=self.hparams.train_date_split,
            test_date_split=self.hparams.test_date_split,
            neg_num=self.hparams.neg_num,
            user_dev_size=self.hparams.user_dev_size,
            train=False,
            validation=False,
            download=True,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = AdressaDataFrame(
                seed=self.hparams.seed,
                dataset_size=self.hparams.dataset_size,
                dataset_url=self.hparams.dataset_url,
                data_dir=self.hparams.data_dir,
                dataset_attributes=self.hparams.dataset_attributes,
                id2index_filenames=self.hparams.id2index_filenames,
                pretrained_embeddings_url=self.hparams.pretrained_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                use_plm=self.hparams.use_plm,
                use_pretrained_categ_embeddings=self.hparams.use_pretrained_categ_embeddings,
                categ_embed_dim=self.hparams.categ_embed_dim,
                word_embed_dim=self.hparams.word_embed_dim,
                sentiment_annotator=self.hparams.sentiment_annotator,
                train_date_split=self.hparams.train_date_split,
                test_date_split=self.hparams.test_date_split,
                neg_num=self.hparams.neg_num,
                user_dev_size=self.hparams.user_dev_size,
                train=True,
                validation=False,
                download=False,
            )
            validset = AdressaDataFrame(
                seed=self.hparams.seed,
                dataset_size=self.hparams.dataset_size,
                dataset_url=self.hparams.dataset_url,
                data_dir=self.hparams.data_dir,
                dataset_attributes=self.hparams.dataset_attributes,
                id2index_filenames=self.hparams.id2index_filenames,
                pretrained_embeddings_url=self.hparams.pretrained_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                use_plm=self.hparams.use_plm,
                use_pretrained_categ_embeddings=self.hparams.use_pretrained_categ_embeddings,
                categ_embed_dim=self.hparams.categ_embed_dim,
                word_embed_dim=self.hparams.word_embed_dim,
                sentiment_annotator=self.hparams.sentiment_annotator,
                train_date_split=self.hparams.train_date_split,
                test_date_split=self.hparams.test_date_split,
                neg_num=self.hparams.neg_num,
                user_dev_size=self.hparams.user_dev_size,
                train=True,
                validation=True,
                download=False,
            )
            testset = AdressaDataFrame(
                seed=self.hparams.seed,
                dataset_size=self.hparams.dataset_size,
                dataset_url=self.hparams.dataset_url,
                data_dir=self.hparams.data_dir,
                dataset_attributes=self.hparams.dataset_attributes,
                id2index_filenames=self.hparams.id2index_filenames,
                pretrained_embeddings_url=self.hparams.pretrained_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                use_plm=self.hparams.use_plm,
                use_pretrained_categ_embeddings=self.hparams.use_pretrained_categ_embeddings,
                categ_embed_dim=self.hparams.categ_embed_dim,
                word_embed_dim=self.hparams.word_embed_dim,
                sentiment_annotator=self.hparams.sentiment_annotator,
                train_date_split=self.hparams.train_date_split,
                test_date_split=self.hparams.test_date_split,
                neg_num=self.hparams.neg_num,
                user_dev_size=self.hparams.user_dev_size,
                train=False,
                validation=False,
                download=False,
            )

            self.data_train = RecommendationDatasetTrain(
                news=trainset.news,
                behaviors=trainset.behaviors,
                max_history_len=self.hparams.max_history_len,
                neg_sampling_ratio=self.hparams.neg_sampling_ratio,
            )
            self.data_val = RecommendationDatasetTest(
                news=validset.news,
                behaviors=validset.behaviors,
                max_history_len=self.hparams.max_history_len,
            )
            self.data_test = RecommendationDatasetTest(
                news=testset.news,
                behaviors=testset.behaviors,
                max_history_len=self.hparams.max_history_len,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=DatasetCollate(
                dataset_attributes=self.hparams.dataset_attributes,
                use_plm=self.hparams.use_plm,
                tokenizer=self.tokenizer if self.hparams.use_plm else None,
                max_title_len=self.hparams.max_title_len if not self.hparams.use_plm else None,
                max_abstract_len=self.hparams.max_abstract_len,
                concatenate_inputs=self.hparams.concatenate_inputs,
            ),
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=DatasetCollate(
                dataset_attributes=self.hparams.dataset_attributes,
                use_plm=self.hparams.use_plm,
                tokenizer=self.tokenizer if self.hparams.use_plm else None,
                max_title_len=self.hparams.max_title_len,
                max_abstract_len=self.hparams.max_abstract_len,
                concatenate_inputs=self.hparams.concatenate_inputs,
            ),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=DatasetCollate(
                dataset_attributes=self.hparams.dataset_attributes,
                use_plm=self.hparams.use_plm,
                tokenizer=self.tokenizer if self.hparams.use_plm else None,
                max_title_len=self.hparams.max_title_len if not self.hparams.use_plm else None,
                max_abstract_len=self.hparams.max_abstract_len,
                concatenate_inputs=self.hparams.concatenate_inputs,
            ),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
