from typing import Dict, List, Optional, Tuple

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import LightningModule
from MulticoreTSNE import MulticoreTSNE as TSNE
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import SupConLoss
from torchmetrics import MeanMetric, MinMetric

from newsreclib.data.components.batch import NewsBatch
from newsreclib.data.components.file_utils import load_idx_map_as_dict
from newsreclib.models.abstract_recommender import AbstractRecommneder
from newsreclib.models.components.encoders.news.news import NewsEncoder
from newsreclib.models.components.encoders.news.text import PLM, MHSAAddAtt


class AModule(AbstractRecommneder):
    """Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation.

    Reference: Iana, Andreea, Goran Glavaš, and Heiko Paulheim. "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation." arXiv preprint arXiv:2307.16089 (2023).

    For further details, please refer to the `paper <https://arxiv.org/abs/2307.16089>`_

    Attributes:
        dataset_attributes:
            List of news features available in the used dataset.
        attributes2encode:
            List of news features used as input to the news encoder.
        outputs:
            A dictionary of user-defined attributes needed for metric calculation at the end of each `*_step` of the pipeline.
        temperature:
            The temperature parameter for the supervised contrastive loss function.
        labels_path:
            The filepath to the dictionary mapping labels to indices.
        plm_model:
            Name of the pretrained language model.
        frozen_layers:
            List of layers to freeze during training.
        text_embed_dim:
            Number of features in the text vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
        use_entities:
            Whether to use entities as input features to the news encoder.
        pretrained_entity_embeddings_path:
            The filepath for the pretrained entity embeddings.
        entity_embed_dim:
            Number of features in the entity vector.
        optimizer:
            Optimizer used for model training.
        scheduler:
            Learning rate scheduler.
    """

    def __init__(
        self,
        dataset_attributes: List[str],
        attributes2encode: List[str],
        outputs: Dict[str, List[str]],
        temperature: float,
        labels_path: str,
        plm_model: Optional[str],
        frozen_layers: Optional[List[int]],
        text_embed_dim: int,
        num_heads: int,
        query_dim: int,
        dropout_probability: float,
        use_entities: bool,
        pretrained_entity_embeddings_path: str,
        entity_embed_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__(
            outputs=outputs,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # initialize text encoder
        text_encoder = PLM(
            plm_model=self.hparams.plm_model,
            frozen_layers=self.hparams.frozen_layers,
            embed_dim=self.hparams.text_embed_dim,
            use_mhsa=False,
            apply_reduce_dim=False,
            reduced_embed_dim=None,
            num_heads=self.hparams.num_heads,
            query_dim=self.hparams.query_dim,
            dropout_probability=self.hparams.dropout_probability,
        )

        # initialize entity encoder
        if self.hparams.use_entities:
            # load pretrained entity embeddings
            assert isinstance(self.hparams.pretrained_entity_embeddings_path, str)
            pretrained_entity_embeddings = self._init_embedding(
                filepath=self.hparams.pretrained_entity_embeddings_path
            )

            entity_encoder = MHSAAddAtt(
                pretrained_embeddings=pretrained_entity_embeddings,
                embed_dim=self.hparams.entity_embed_dim,
                num_heads=self.hparams.num_heads,
                query_dim=self.hparams.query_dim,
                dropout_probability=self.hparams.dropout_probability,
            )
        else:
            entity_encoder = None

        # initialize news encoder
        news_encoder_input_dim = (
            self.hparams.text_embed_dim + self.hparams.entity_embed_dim
            if self.hparams.use_entities
            else self.hparams.text_embed_dim
        )
        self.news_encoder = NewsEncoder(
            dataset_attributes=self.hparams.dataset_attributes,
            attributes2encode=self.hparams.attributes2encode,
            concatenate_inputs=True,
            text_encoder=text_encoder,
            category_encoder=None,
            entity_encoder=entity_encoder,
            combine_vectors=True,
            combine_type="linear",
            input_dim=news_encoder_input_dim,
            query_dim=None,
            output_dim=self.hparams.text_embed_dim,
        )

        self.tsne = TSNE(n_jobs=8)

        label2index = load_idx_map_as_dict(labels_path)
        labels = list(label2index.keys())
        self.index2label = {v: k for k, v in label2index.items()}
        self.color_map = dict(
            zip(
                labels,
                sns.color_palette(cc.glasbey_light, n_colors=len(labels)),
            )
        )

        # loss function
        distance_func = DotProductSimilarity(normalize_embeddings=False)
        self.criterion = SupConLoss(temperature=self.hparams.temperature, distance=distance_func)

        # collect outputs of `*_step`
        self.val_step_outputs = {key: [] for key in self.step_outputs["val"]}
        self.test_step_outputs = {key: [] for key in self.step_outputs["test"]}

    def forward(self, batch: NewsBatch) -> torch.Tensor:
        # encode news
        embeddings = self.news_encoder(batch["news"])

        return embeddings

    def on_train_start(self) -> None:
        pass

    def model_step(
        self, batch: NewsBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        embeddings = self.forward(batch)
        labels = batch["labels"]

        loss = self.criterion(embeddings, labels)

        return loss, embeddings, labels

    def training_step(self, batch: NewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: NewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # collect step outputs for metric computation
        self.val_step_outputs = self._collect_step_outputs(
            outputs_dict=self.val_step_outputs, local_vars=locals()
        )

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.current_epoch % 10 == 0:
            embeddings = (
                self._gather_step_outputs(self.val_step_outputs, "embeddings").detach().cpu()
            )
            labels = (
                self._gather_step_outputs(self.val_step_outputs, "labels").detach().cpu().numpy()
            )
            transformed_labels = [self.index2label[label] for label in labels]

            tsne_embeddings = self.tsne.fit_transform(embeddings)

            # plot TSNE embeddings
            fig = plt.figure(figsize=(10, 7))
            ax = sns.scatterplot(
                x=tsne_embeddings[:, 0],
                y=tsne_embeddings[:, 1],
                hue=[label for label in transformed_labels],
                palette=self.color_map,
                legend="full",
            )
            ax.set_title("Val Embeddings tSNE")
            lgd = ax.legend(bbox_to_anchor=(1, 1), loc=2)
            img_path = self.logger.save_dir + "/val_embeddings_" + str(self.current_epoch) + ".jpg"
            plt.savefig(img_path, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=400)

            self.logger.log_image(key="tSNE Embeddings", images=[img_path])

        # clear memory for the next epoch
        self.val_step_outputs = self._clear_epoch_outputs(self.val_step_outputs)

    def test_step(self, batch: NewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # collect step outputs for metric computation
        self.test_step_outputs = self._collect_step_outputs(
            outputs_dict=self.test_step_outputs, local_vars=locals()
        )

    def on_test_epoch_end(self) -> None:
        embeddings = self._gather_step_outputs(self.test_step_outputs, "embeddings").detach().cpu()
        labels = self._gather_step_outputs(self.test_step_outputs, "labels").detach().cpu().numpy()
        transformed_labels = [self.index2label[label] for label in labels]

        tsne_embeddings = self.tsne.fit_transform(embeddings)

        # plot TSNE embeddings
        fig = plt.figure(figsize=(10, 7))
        ax = sns.scatterplot(
            x=tsne_embeddings[:, 0],
            y=tsne_embeddings[:, 1],
            hue=[label for label in transformed_labels],
            palette=self.color_map,
            legend="full",
        )
        ax.set_title("Test Embeddings tSNE")
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc=2)
        img_path = self.logger.save_dir + "/test_embeddings_" + str(self.current_epoch) + ".jpg"
        plt.savefig(img_path, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=400)

        self.logger.log_image(key="tSNE Embeddings", images=[img_path])

        # clear memory for the next epoch
        self._clear_epoch_outputs(self.test_step_outputs)
