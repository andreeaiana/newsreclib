from typing import Any, Dict, List, Optional

import torch.nn as nn
from lightning import LightningDataModule
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from newsreclib.data.components.rec_dataset import (
    DatasetCollate,
    RecommendationDatasetTest,
    RecommendationDatasetTrain,
)
from newsreclib.data.components.xmind_dataframe import xMINDDataFrame


class xMINDRecDataModule(LightningDataModule):
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
        dataset_size:
            A string indicating the type of the dataset. Choose between `large` and `small`.
        mind_dataset_url:
            Dictionary of URLs for downloading the MIND `train` and `dev` datasets for the specified `dataset_size`.
        data_dir:
            Path to the data directory.
        tgt_lang:
            Target language for the xMIND dataset.
        bilingual_train:
            Whether the user history and the candidates list is bilingual (or monolingual in target language) or monolingual in source language in training.
        pct_tgt_lang_train:
            The percentage of news in the target language in the bilingual user history and candidates list in training.
        bilingual_test:
            Whether the user history and the candidates list is bilingual (or monolingual in target language) or monolingual in source language in test.
        pct_tgt_lang_test:
            The percentage of news in the target language in the bilingual user history and candidates list in test.
        dataset_attributes:
            List of news features available in the used dataset (e.g., title, category, etc.).
        id2index_filenames:
            Dictionary mapping id2index dictionary to corresponding filenames.
        entity_embeddings_filename:
            Filepath to the pretrained entity embeddings.
        entity_embed_dim:
            Dimensionality of entity embeddings.
        entity_freq_threshold:
            Minimum frequency for an entity to be included in the processed dataset.
        entity_conf_threshold:
            Minimum confidence for an entity to be included in the processed dataset.
        valid_time_split:
            A string with the date before which click behaviors are included in the train set. After this date, behaviors are included in the validation set.
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
        dataset_size: str,
        mind_dataset_url: DictConfig,
        data_dir: str,
        tgt_lang: str,
        bilingual_train: bool,
        pct_tgt_lang_train: Optional[float],
        bilingual_test: bool,
        pct_tgt_lang_test: Optional[float],
        dataset_attributes: List[str],
        id2index_filenames: DictConfig,
        entity_embeddings_filename: str,
        entity_embed_dim: int,
        entity_freq_threshold: int,
        entity_conf_threshold: float,
        valid_time_split: str,
        max_title_len: int,
        max_abstract_len: int,
        concatenate_inputs: bool,
        tokenizer_name: str,
        tokenizer_use_fast: bool,
        tokenizer_max_len: int,
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
        xMINDDataFrame(
            dataset_size=self.hparams.dataset_size,
            mind_dataset_url=self.hparams.mind_dataset_url,
            data_dir=self.hparams.data_dir,
            tgt_lang=self.hparams.tgt_lang,
            bilingual=self.hparams.bilingual_train,
            pct_tgt_lang=self.hparams.pct_tgt_lang_train,
            dataset_attributes=self.hparams.dataset_attributes,
            id2index_filenames=self.hparams.id2index_filenames,
            entity_embeddings_filename=self.hparams.entity_embeddings_filename,
            entity_embed_dim=self.hparams.entity_embed_dim,
            entity_freq_threshold=self.hparams.entity_freq_threshold,
            entity_conf_threshold=self.hparams.entity_conf_threshold,
            valid_time_split=self.hparams.valid_time_split,
            train=True,
            validation=False,
            download_mind=True,
        )

        # download validation set
        xMINDDataFrame(
            dataset_size=self.hparams.dataset_size,
            mind_dataset_url=self.hparams.mind_dataset_url,
            data_dir=self.hparams.data_dir,
            tgt_lang=self.hparams.tgt_lang,
            bilingual=self.hparams.bilingual_test,
            pct_tgt_lang=self.hparams.pct_tgt_lang_test,
            dataset_attributes=self.hparams.dataset_attributes,
            id2index_filenames=self.hparams.id2index_filenames,
            entity_embeddings_filename=self.hparams.entity_embeddings_filename,
            entity_embed_dim=self.hparams.entity_embed_dim,
            entity_freq_threshold=self.hparams.entity_freq_threshold,
            entity_conf_threshold=self.hparams.entity_conf_threshold,
            valid_time_split=self.hparams.valid_time_split,
            train=False,
            validation=False,
            download_mind=True,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = xMINDDataFrame(
                dataset_size=self.hparams.dataset_size,
                mind_dataset_url=self.hparams.mind_dataset_url,
                data_dir=self.hparams.data_dir,
                tgt_lang=self.hparams.tgt_lang,
                bilingual=self.hparams.bilingual_train,
                pct_tgt_lang=self.hparams.pct_tgt_lang_train,
                dataset_attributes=self.hparams.dataset_attributes,
                id2index_filenames=self.hparams.id2index_filenames,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                entity_embed_dim=self.hparams.entity_embed_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_conf_threshold=self.hparams.entity_conf_threshold,
                valid_time_split=self.hparams.valid_time_split,
                train=True,
                validation=False,
                download_mind=False,
            )

            validset = xMINDDataFrame(
                dataset_size=self.hparams.dataset_size,
                mind_dataset_url=self.hparams.mind_dataset_url,
                data_dir=self.hparams.data_dir,
                tgt_lang=self.hparams.tgt_lang,
                bilingual=self.hparams.bilingual_train,
                pct_tgt_lang=self.hparams.pct_tgt_lang_train,
                dataset_attributes=self.hparams.dataset_attributes,
                id2index_filenames=self.hparams.id2index_filenames,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                entity_embed_dim=self.hparams.entity_embed_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_conf_threshold=self.hparams.entity_conf_threshold,
                valid_time_split=self.hparams.valid_time_split,
                train=True,
                validation=True,
                download_mind=False,
            )

            testset = xMINDDataFrame(
                dataset_size=self.hparams.dataset_size,
                mind_dataset_url=self.hparams.mind_dataset_url,
                data_dir=self.hparams.data_dir,
                tgt_lang=self.hparams.tgt_lang,
                bilingual=self.hparams.bilingual_test,
                pct_tgt_lang=self.hparams.pct_tgt_lang_test,
                dataset_attributes=self.hparams.dataset_attributes,
                id2index_filenames=self.hparams.id2index_filenames,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                entity_embed_dim=self.hparams.entity_embed_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_conf_threshold=self.hparams.entity_conf_threshold,
                valid_time_split=self.hparams.valid_time_split,
                train=False,
                validation=False,
                download_mind=False,
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
                use_plm=True,
                tokenizer=self.tokenizer,
                max_title_len=None,
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
                use_plm=True,
                tokenizer=self.tokenizer,
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
                use_plm=True,
                tokenizer=self.tokenizer,
                max_title_len=self.hparams.max_title_len,
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
