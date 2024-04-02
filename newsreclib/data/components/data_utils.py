import os
import re
import tarfile
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from newsreclib import utils
from newsreclib.data.components.download_utils import (
    download_path,
    extract_file,
    maybe_download,
)
from newsreclib.data.components.file_utils import check_integrity

log = utils.get_pylogger(__name__)


def word_tokenize(sentence: str) -> List[str]:
    """Splits a sentence into word list using regex.

    Args:
        sentence:
            Input sentence

    Returns:
        List of words.
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sentence, str):
        return pat.findall(sentence.lower())
    else:
        return []


def generate_pretrained_embeddings(
    word2index: Dict[str, int],
    embeddings_fpath: str,
    embed_dim: int,
    transformed_embeddings_fpath: str,
) -> None:
    """Loads pretrained embeddings for the words (or entities) in word_dict.

    Args:
        word2index:
            Dictionary mapping words to indices.
        embeddings_fpath:
            Filepath of the pretrained embeddings to load.
        ebedding_dim:
            Dimensionality of the pretrained embeddings.
        transformed_embeddings_fpath:
            Destination directory for the transformed embeddings.
    """

    embedding_matrix = np.random.normal(size=(len(word2index) + 1, embed_dim))
    exist_word = set()

    with open(embeddings_fpath) as f:
        for line in tqdm(f):
            linesplit = line.split(" ")
            word = line[0]
            if len(word) != 0:
                if word in word2index:
                    embedding_matrix[word2index[word]] = np.asarray(
                        list(map(float, linesplit[1:]))
                    )
                    exist_word.add(word)

    log.info(f"Rate of word missed in pretrained embedding: {(len(exist_word)/len(word2index))}.")

    if not check_integrity(transformed_embeddings_fpath):
        log.info(f"Saving word embeddings in {transformed_embeddings_fpath}")
        np.save(transformed_embeddings_fpath, embedding_matrix, allow_pickle=True)


def download_and_extract_dataset(
    data_dir: str,
    url: str,
    filename: str,
    extract_compressed: bool,
    dst_dir: Optional[str],
    clean_archive: Optional[bool],
) -> None:
    """Downloads a dataset from the specified url and extracts the compessed data file.

    Args:
        data_dir:
            Path where to download data.
        url:
            URL of the file to download.
        filename:
            Name of the file to download.
        extract_compressed:
            Whether to extract the compressed downloaded file.
        dst_dir:
            Destination directory for the extracted file.
        clean_archive:
            Whether to delete the compressed file after extraction.
    """
    with download_path(data_dir) as path:
        path = maybe_download(url=url, filename=filename, work_directory=path)
        log.info("Compressed dataset downloaded")

        if extract_compressed:
            assert isinstance(dst_dir, str) and isinstance(clean_archive, bool)
            # extract the compressed dataset
            log.info(f"Extracting dataset from {path} into {dst_dir}.")
            extract_file(archive_file=path, dst_dir=dst_dir, clean_archive=clean_archive)
            log.info("Dataset extraction completed.")


def download_and_extract_pretrained_embeddings(
    data_dir: str,
    url: str,
    pretrained_embeddings_fpath: str,
    filename: str,
    dst_dir: str,
    clean_archive: bool,
) -> None:
    """Downloads pretrained embeddings from the specified url and extracts the compressed data
    file.

    Args:
        data_dir:
            Path where to download data.
        url:
            URL of the file to download.
        filename:
            Name of the file to download.
        dst_dir:
            Destination directory for the extracted file.
        clean_archive:
            Whether to delete the compressed file after extraction.
    """
    log.info(f"Downloading pretrained embeddings from {url}.")

    # download the pretrained embeddings
    with download_path(data_dir) as path:
        path = maybe_download(url=url, filename=filename, work_directory=path)
        log.info("Compressed pretrained embeddings downloaded.")

    # extract the compressed embeddings file
    if not check_integrity(pretrained_embeddings_fpath):
        log.info(f"Extracting pretrained embeddings from {path} into {dst_dir}.")
        if not path.endswith("gz"):
            extract_file(archive_file=path, dst_dir=dst_dir, clean_archive=clean_archive)
        else:
            tar = tarfile.open(os.path.join(data_dir, filename), "r:gz")
            for member in tar.getmembers():
                if member.isreg():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, dst_dir)
            tar.close()

        log.info("Pretrained embedding extraction completed.")


def get_article2clicks_mind(df_behaviors):
    """
    Get the time news articles were published. For the MIND dataset, 
    since this information is not available, it'll be obtained based 
    on their first appearance on the impression column. 

    Args:
    df_behaviors:
            Behaviors dataframe
    """
    print(df_behaviors.columns)
    article2published = {}
    article2clicks = defaultdict(list)
    k = 0
    for _, behavior in df_behaviors.iterrows():
        k+=1
        if isinstance(behavior["time"], str):
            time = datetime.strptime(behavior["time"], "%m/%d/%Y %I:%M:%S %p")
        else:
            time = behavior["time"]
        if k < 3:
            print(time)
            print(type(time))
        for article_id_and_clicked in behavior["impressions"]:
            article_id = article_id_and_clicked[:-2]  # article id ex: N113723
            article_clicked = article_id_and_clicked[
                -1
            ]  # 0 (not clicked) and 1 (clicked)

            # Track the first time an article appears and add that time occurance as publish time
            if (
                article_id not in article2published
                or time < article2published[article_id]
            ):
                article2published[article_id] = time

            # Append the hour when the article was clicked
            if article_clicked == "1":
                article2clicks[article_id].append(time)

    # --- Sort article2clicks dictionary
    for article_id, clicks in article2clicks.items():
        clicks.sort()
        for i, click in enumerate(clicks):
            clicks[i] = (click - article2published[article_id]
                         ).total_seconds() // 3600

    return article2published, article2clicks