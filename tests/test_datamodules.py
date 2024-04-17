from pathlib import Path

import pytest

from newsreclib.data.adressa_rec_datamodule import AdressaRecDataModule
from newsreclib.data.components.sentiment_annotator import BERTSentimentAnnotator
from newsreclib.data.mind_rec_datamodule import MINDRecDataModule


@pytest.mark.parametrize("batch_size", [8, 64])
def test_mind_rec_small_datamodule(batch_size):
    dataset_size = "small"  # URLs for downloading the dataset
    dataset_url = {
        "large": {
            "train": "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
            "dev": "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
        },
        "small": {
            "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
            "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
        },
    }
    pretrained_embeddings_url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    data_dir = "data/"
    word_embeddings_dirname = "glove"
    word_embeddings_fpath = "data/glove/glove.840B.300d.txt"
    entity_embeddings_filename = "entity_embedding.vec"
    dataset_attributes = [
        "title",
        "abstract",
        "category",
        "subcategory",
        "title_entities",
        "abstract_entities",
        "category_class",
        "subcategory_class",
        "sentiment_class",
        "sentiment_score",
    ]
    id2index_filenames = {
        "word2index": "word2index.tsv",
        "entity2index": "entity2index.tsv",
        "categ2index": "categ2index.tsv",
        "subcateg2index": "subcateg2index.tsv",
        "sentiment2index": "sentiment2index.tsv",
        "uid2index": "uid2index.tsv",
    }
    use_plm = False
    use_pretrained_categ_embeddings = True
    word_embed_dim = 300
    categ_embed_dim = 300
    entity_embed_dim = 100
    entity_freq_threshold = 2
    entity_conf_threshold = 0.5
    sentiment_annotator = BERTSentimentAnnotator(
        model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer_max_len=96
    )
    valid_time_split = "2019-11-14 00:00:00"
    neg_sampling_ratio = 4
    max_title_len = 30
    max_abstract_len = 50
    max_history_len = 50
    concatenate_inputs = False
    tokenizer_name = "roberta-base"
    tokenizer_use_fast = True
    tokenizer_max_len = 96
    batch_size = 64
    num_workers = 0
    pin_memory = True
    drop_last = False

    dm = MINDRecDataModule(
        dataset_size=dataset_size,
        dataset_url=dataset_url,
        data_dir=data_dir,
        dataset_attributes=dataset_attributes,
        id2index_filenames=id2index_filenames,
        pretrained_embeddings_url=pretrained_embeddings_url,
        word_embeddings_dirname=word_embeddings_dirname,
        word_embeddings_fpath=word_embeddings_fpath,
        entity_embeddings_filename=entity_embeddings_filename,
        use_plm=use_plm,
        use_pretrained_categ_embeddings=use_pretrained_categ_embeddings,
        categ_embed_dim=categ_embed_dim,
        word_embed_dim=word_embed_dim,
        entity_embed_dim=entity_embed_dim,
        entity_freq_threshold=entity_freq_threshold,
        entity_conf_threshold=entity_conf_threshold,
        sentiment_annotator=sentiment_annotator,
        valid_time_split=valid_time_split,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
        concatenate_inputs=concatenate_inputs,
        tokenizer_name=tokenizer_name,
        tokenizer_use_fast=tokenizer_use_fast,
        tokenizer_max_len=tokenizer_max_len,
        max_history_len=max_history_len,
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MIND" + dataset_size + "_train").exists()
    assert Path(data_dir, "MIND" + dataset_size + "_dev").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 224665

    batch = next(iter(dm.train_dataloader()))

    assert len(batch["user_idx"]) == batch_size


@pytest.mark.parametrize("batch_size", [8, 64])
def test_adressa_rec_small_datamodule(batch_size):
    seed = 42
    dataset_size = "one_week"  # URLs for downloading the dataset
    dataset_url = {
        "three_month": "https://reclab.idi.ntnu.no/dataset/three_month.tar.gz",
        "one_week": "https://reclab.idi.ntnu.no/dataset/one_week.tar.gz",
    }
    pretrained_embeddings_url = (
        "https://bpemb.h-its.org/no/no.wiki.bpe.vs200000.d300.w2v.txt.tar.gz"
    )
    data_dir = "data/"
    word_embeddings_dirname = "glove"
    word_embeddings_fpath = "data/glove/no.wiki.bpe.vs200000.d300.w2v.txt"
    dataset_attributes = [
        "title",
        "category",
        "subcategory",
        "category_class",
        "subcategory_class",
        "sentiment_class",
        "sentiment_score",
    ]
    id2index_filenames = {
        "word2index": "word2index.tsv",
        "categ2index": "categ2index.tsv",
        "subcateg2index": "subcateg2index.tsv",
        "sentiment2index": "sentiment2index.tsv",
        "uid2index": "uid2index.tsv",
        "nid2index": "nid2index.tsv",
    }
    use_plm = False
    use_pretrained_categ_embeddings = True
    word_embed_dim = 300
    categ_embed_dim = 300
    sentiment_annotator = BERTSentimentAnnotator(
        model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer_max_len=96
    )
    train_date_split = 6
    test_date_split = 7
    neg_num = 20
    user_dev_size = 0.2
    neg_sampling_ratio = 4
    max_title_len = 30
    max_abstract_len = 50
    max_history_len = 50
    concatenate_inputs = False
    tokenizer_name = "NbAiLab/nb-bert-base"
    tokenizer_use_fast = True
    tokenizer_max_len = 96
    batch_size = 64
    num_workers = 0
    pin_memory = True
    drop_last = False

    dm = AdressaRecDataModule(
        seed=seed,
        dataset_size=dataset_size,
        dataset_url=dataset_url,
        data_dir=data_dir,
        dataset_attributes=dataset_attributes,
        id2index_filenames=id2index_filenames,
        pretrained_embeddings_url=pretrained_embeddings_url,
        word_embeddings_dirname=word_embeddings_dirname,
        word_embeddings_fpath=word_embeddings_fpath,
        use_plm=use_plm,
        use_pretrained_categ_embeddings=use_pretrained_categ_embeddings,
        categ_embed_dim=categ_embed_dim,
        word_embed_dim=word_embed_dim,
        sentiment_annotator=sentiment_annotator,
        train_date_split=train_date_split,
        test_date_split=test_date_split,
        neg_num=neg_num,
        user_dev_size=user_dev_size,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
        concatenate_inputs=concatenate_inputs,
        tokenizer_name=tokenizer_name,
        tokenizer_use_fast=tokenizer_use_fast,
        tokenizer_max_len=tokenizer_max_len,
        max_history_len=max_history_len,
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "Adressa_" + dataset_size).exists()
    assert Path(data_dir, "Adressa_" + dataset_size, "train").exists()
    assert Path(data_dir, "Adressa_" + dataset_size, "dev").exists()
    assert Path(data_dir, "Adressa_" + dataset_size, "test").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 365132

    batch = next(iter(dm.train_dataloader()))

    assert len(batch["user_idx"]) == batch_size
