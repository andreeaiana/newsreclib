from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.utils import to_dense_batch
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from newsreclib.data.components.batch import RecommendationBatch
from newsreclib.metrics.diversity import Diversity
from newsreclib.metrics.personalization import Personalization
from newsreclib.models.abstract_recommender import AbstractRecommneder
from newsreclib.models.components.layers.click_predictor import DotProduct
from newsreclib.models.fair_rec.manner_a_module import AModule
from newsreclib.models.fair_rec.manner_cr_module import CRModule


class MANNERModule(AbstractRecommneder):
    """Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation.

    Reference: Iana, Andreea, Goran Glavaš, and Heiko Paulheim. "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation." arXiv preprint arXiv:2307.16089 (2023).

    For further details, please refer to the `paper <https://arxiv.org/abs/2307.16089>`_

    Attributes:
        outputs:
            A dictionary of user-defined attributes needed for metric calculation at the end of each `*_step` of the pipeline.
        cr_module_module_ckpt:
            The chekpoint of the CR-Module.
        a_module_categ_ckpt:
            The checkpoint of the category-based A-Module.
        a_module_sent_ckpt:
             The checkpoint of the sentiment-based A-Module.
        categ_weight:
            The weight of the category-based A-Module.
        sent_weight:
            The weight of the sentiment-based A-Module.
        num_categ_classes:
            The number of topical categories.
        num_sent_classes:
            The number of sentiment classes.
        optimizer:
            Optimizer used for model training.
        scheduler:
            Learning rate scheduler.
    """

    def __init__(
        self,
        outputs: Dict[str, List[str]],
        cr_module_module_ckpt: str,
        a_module_categ_ckpt: Optional[str],
        a_module_sent_ckpt: Optional[str],
        categ_weight: Optional[float],
        sent_weight: Optional[float],
        num_categ_classes: int,
        num_sent_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__(
            outputs=outputs,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.num_categ_classes = self.hparams.num_categ_classes + 1
        self.num_sent_classes = self.hparams.num_sent_classes + 1

        # load ensemble components
        self.cr_module = CRModule.load_from_checkpoint(
            checkpoint_path=self.hparams.cr_module_module_ckpt
        )

        if self.hparams.categ_weight != 0:
            assert isinstance(self.hparams.a_module_categ_ckpt, str)
            self.a_module_categ = AModule.load_from_checkpoint(
                checkpoint_path=self.hparams.a_module_categ_ckpt
            )
        if self.hparams.sent_weight != 0:
            assert isinstance(self.hparams.a_module_sent_ckpt, str)
            self.a_module_sent = AModule.load_from_checkpoint(
                checkpoint_path=self.hparams.a_module_sent_ckpt
            )

        self.click_predictor = DotProduct()

        # collect outputs of `*_step`
        self.test_step_outputs = {key: [] for key in self.step_outputs["test"]}

        # metric objects for calculating and averaging performance across batches
        rec_metrics = MetricCollection(
            {
                "auc": AUROC(task="binary", num_classes=2),
                "mrr": RetrievalMRR(),
                "ndcg@5": RetrievalNormalizedDCG(k=5),
                "ndcg@10": RetrievalNormalizedDCG(k=10),
            }
        )
        self.train_rec_metrics = rec_metrics.clone(prefix="train/")
        self.val_rec_metrics = rec_metrics.clone(prefix="val/")
        self.test_rec_metrics = rec_metrics.clone(prefix="test/")

        categ_div_metrics = MetricCollection(
            {
                "categ_div@5": Diversity(num_classes=self.num_categ_classes, top_k=5),
                "categ_div@10": Diversity(num_classes=self.num_categ_classes, top_k=10),
            }
        )
        sent_div_metrics = MetricCollection(
            {
                "sent_div@5": Diversity(num_classes=self.num_sent_classes, top_k=5),
                "sent_div@10": Diversity(num_classes=self.num_sent_classes, top_k=10),
            }
        )
        categ_pers_metrics = MetricCollection(
            {
                "categ_pers@5": Personalization(num_classes=self.num_categ_classes, top_k=5),
                "categ_pers@10": Personalization(num_classes=self.num_categ_classes, top_k=10),
            }
        )
        sent_pers_metrics = MetricCollection(
            {
                "sent_pers@5": Personalization(num_classes=self.num_sent_classes, top_k=5),
                "sent_pers@10": Personalization(num_classes=self.num_sent_classes, top_k=10),
            }
        )
        self.test_categ_div_metrics = categ_div_metrics.clone(prefix="test/")
        self.test_sent_div_metrics = sent_div_metrics.clone(prefix="test/")
        self.test_categ_pers_metrics = categ_pers_metrics.clone(prefix="test/")
        self.test_sent_pers_metrics = sent_pers_metrics.clone(prefix="test/")

    def _submodel_forward(
        self, batch: RecommendationBatch, model: Union[CRModule, AModule]
    ) -> torch.Tensor:
        # encode clicked news
        hist_news_vector = model.news_encoder(batch["x_hist"])
        hist_news_vector_agg, mask_hist = to_dense_batch(hist_news_vector, batch["batch_hist"])

        # encode candidate news
        cand_news_vector = model.news_encoder(batch["x_cand"])
        cand_news_vector_agg, mask_cand = to_dense_batch(cand_news_vector, batch["batch_cand"])

        # aggregated history
        hist_size = torch.tensor(
            [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
            device=self.device,
        )
        user_vector = torch.div(hist_news_vector_agg.sum(dim=1), hist_size.unsqueeze(dim=-1))

        scores = self.click_predictor(
            user_vector.unsqueeze(dim=1), cand_news_vector_agg.permute(0, 2, 1)
        )

        # z-score normalization
        cand_size = torch.tensor(
            [torch.where(mask_cand[i])[0].shape[0] for i in range(mask_cand.shape[0])],
            device=self.device,
        )
        std_devs = torch.stack(
            [torch.std(scores[i][mask_cand[i]]) for i in range(mask_cand.shape[0])]
        ).unsqueeze(-1)
        scores = torch.div(
            scores
            - torch.div(torch.sum(scores, dim=1), cand_size).unsqueeze(-1).expand_as(scores),
            std_devs,
        )

        return scores

    def forward(self, batch: RecommendationBatch) -> torch.Tensor:
        # recommendation (ideal) scores
        scores = self._submodel_forward(batch, model=self.cr_module)

        # category-based scores
        if self.hparams.categ_weight != 0:
            categ_scores = self._submodel_forward(batch, model=self.a_module_categ)
            scores += self.hparams.categ_weight * categ_scores

        # sentiment-based scores
        if self.hparams.sent_weight != 0:
            sent_scores = self._submodel_forward(batch, model=self.a_module_sent)
            scores += self.hparams.sent_weight * sent_scores

        return scores

    def on_train_start(self) -> None:
        pass

    def model_step(
        self, batch: RecommendationBatch
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        scores = self.forward(batch)

        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        candidate_categories, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"])
        candidate_sentiments, _ = to_dense_batch(batch["x_cand"]["sentiment"], batch["batch_cand"])

        clicked_categories, mask_hist = to_dense_batch(
            batch["x_hist"]["category"], batch["batch_hist"]
        )
        clicked_sentiments, _ = to_dense_batch(batch["x_hist"]["sentiment"], batch["batch_hist"])

        # model outputs for metric computation
        preds = self._collect_model_outputs(scores, mask_cand)
        targets = self._collect_model_outputs(y_true, mask_cand)

        hist_categories = self._collect_model_outputs(clicked_categories, mask_hist)
        hist_sentiments = self._collect_model_outputs(clicked_sentiments, mask_hist)

        target_categories = self._collect_model_outputs(candidate_categories, mask_cand)
        target_sentiments = self._collect_model_outputs(candidate_sentiments, mask_cand)

        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )
        hist_news_size = torch.tensor(
            [torch.where(mask_hist[n])[0].shape[0] for n in range(mask_hist.shape[0])]
        )

        return (
            preds,
            targets,
            cand_news_size,
            hist_news_size,
            target_categories,
            target_sentiments,
            hist_categories,
            hist_sentiments,
        )

    def training_step(self, batch: RecommendationBatch, batch_idx: int):
        pass

    def validation_step(self, batch: RecommendationBatch, batch_idx: int):
        pass

    def test_step(self, batch: RecommendationBatch, batch_idx: int):
        (
            preds,
            targets,
            cand_news_size,
            hist_news_size,
            target_categories,
            target_sentiments,
            hist_categories,
            hist_sentiments,
        ) = self.model_step(batch)

        # collect step outputs for metric computation
        self.test_step_outputs = self._collect_step_outputs(
            outputs_dict=self.test_step_outputs, local_vars=locals()
        )

    def on_test_epoch_end(self) -> None:
        preds = self._gather_step_outputs(self.test_step_outputs, "preds")
        targets = self._gather_step_outputs(self.test_step_outputs, "targets")

        target_categories = self._gather_step_outputs(self.test_step_outputs, "target_categories")
        target_sentiments = self._gather_step_outputs(self.test_step_outputs, "target_sentiments")

        hist_categories = self._gather_step_outputs(self.test_step_outputs, "hist_categories")
        hist_sentiments = self._gather_step_outputs(self.test_step_outputs, "hist_sentiments")

        cand_news_size = self._gather_step_outputs(self.test_step_outputs, "cand_news_size")
        hist_news_size = self._gather_step_outputs(self.test_step_outputs, "hist_news_size")

        cand_indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        hist_indexes = torch.arange(hist_news_size.shape[0]).repeat_interleave(hist_news_size)

        # update metrics
        self.test_rec_metrics(preds, targets, **{"indexes": cand_indexes})
        self.test_categ_div_metrics(preds, target_categories, cand_indexes)
        self.test_sent_div_metrics(preds, target_sentiments, cand_indexes)
        self.test_categ_pers_metrics(
            preds, target_categories, hist_categories, cand_indexes, hist_indexes
        )
        self.test_sent_pers_metrics(
            preds, target_sentiments, hist_sentiments, cand_indexes, hist_indexes
        )

        # log metrics
        self.log_dict(
            self.test_rec_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_categ_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_categ_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # clear memory for the next epoch
        self.test_step_outputs = self._clear_epoch_outputs(self.test_step_outputs)
