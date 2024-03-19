from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from newsreclib.data.components.batch import RecommendationBatch
from newsreclib.metrics.diversity import Diversity
from newsreclib.metrics.personalization import Personalization
from newsreclib.models.abstract_recommender import AbstractRecommneder
from newsreclib.models.components.encoders.news.category import LinearEncoder
from newsreclib.models.components.encoders.news.news import NewsEncoder
from newsreclib.models.components.encoders.news.text import PLM
from newsreclib.models.components.layers.attention import PolyAttention as UserEncoder
from newsreclib.models.components.layers.attention import TargetAwareAttention
from newsreclib.models.components.layers.click_predictor import DotProduct
from newsreclib.models.components.utils import (
    pairwise_cosine_similarity as pairwise_cosine_similarity_3d,
)


class MINERModule(AbstractRecommneder):
    """MINER: multi-interest matching network for news recommendation

    Reference: Li, Jian, Jieming Zhu, Qiwei Bi, Guohao Cai, Lifeng Shang, Zhenhua Dong, Xin Jiang, and Qun Liu. "MINER: multi-interest matching network for news recommendation." In Findings of the Association for Computational Linguistics: ACL 2022, pp. 343-352. 2022.

    For further details, please refer to the `paper <https://aclanthology.org/2022.findings-acl.29/>`_

    Attributes:
        dataset_attributes:
            List of news features available in the used dataset.
        attributes2encode:
            List of news features used as input to the news encoder.
        outputs:
            A dictionary of user-defined attributes needed for metric calculation at the end of each `*_step` of the pipeline.
        dual_loss_training:
            Whether to train with two loss functions, i.e., cross-entropy and supervised contrastive losses, aggregated with a weighted average.
        dual_loss_coef:
            The weights of each loss, in the case of dual loss training.
        loss:
            The criterion to use for training the model. Choose between `cross_entropy_loss', `sup_con_loss`, and `dual`.
        late_fusion:
            If ``True``, it trains the model with the standard `early fusion` approach (i.e., learns an explicit user embedding). If ``False``, it use the `late fusion`.
        temperature:
            The temperature parameter for the supervised contrastive loss function.
        use_plm:
            If ``True``, it will process the data for a petrained language model (PLM) in the news encoder. If ``False``, it will tokenize the news title and abstract to be used initialized with pretrained word embeddings.
        plm_model:
            Name of the pretrained language model.
        frozen_layers:
            List of layers to freeze during training.
        apply_reduce_dim:
            Whether to linearly reduce the dimensionality of the news vector.
        text_embed_dim:
            Number of features in the text vector.
        news_embed_dim:
            Number of features in the news vector.
        use_categ_bias:
            Whether to use category bias.
        pretrained_categ_embeddings_path:
            The filepath for the pretrained category embeddings.
        num_context_codes:
            The number of attention vectors in the ``PolyAttention``.
        context_code_dim:
            The number of features in a context code in the ``PolyAttention``.
        score_type:
            The type of aggregation used to compute the final user click score from the matching scores. Choose from `max`, `mean`, and `weighted`.
        dropout_probability:
            Dropout probability.
        top_k_list:
            List of positions at which to compute rank-based metrics.
        num_categ_classes:
            The number of topical categories.
        num_sent_classes:
            The number of sentiment classes.
        save_recs:
            Whether to save the recommendations (i.e., candidates news and corresponding scores) to disk in JSON format.
        recs_fpath:
            Path where to save the list of recommendations and corresponding scores for users.
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
        dual_loss_training: bool,
        dual_loss_coef: Optional[float],
        loss: str,
        late_fusion: bool,
        temperature: Optional[float],
        use_plm: bool,
        plm_model: Optional[str],
        frozen_layers: Optional[List[int]],
        apply_reduce_dim: bool,
        text_embed_dim: int,
        news_embed_dim: int,
        use_categ_bias: bool,
        pretrained_categ_embeddings_path: Optional[str],
        num_context_codes: int,
        context_code_dim: int,
        score_type: str,
        dropout_probability: float,
        top_k_list: List[int],
        num_categ_classes: int,
        num_sent_classes: int,
        save_recs: bool,
        recs_fpath: Optional[str],
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

        if self.hparams.save_recs:
            assert isinstance(self.hparams.recs_fpath, str)

        # initialize loss
        if not self.hparams.dual_loss_training:
            self.criterion = self._get_loss(self.hparams.loss)
        else:
            assert isinstance(self.hparams.dual_loss_coef, float)
            self.ce_criterion, self.scl_criterion = self._get_loss(self.hparams.loss)

        # initialize text encoder
        if self.hparams.apply_reduce_dim:
            news_embed_dim = self.hparams.news_embed_dim
        else:
            news_embed_dim = self.hparams.text_embed_dim

        text_encoder = PLM(
            plm_model=self.hparams.plm_model,
            frozen_layers=self.hparams.frozen_layers,
            embed_dim=self.hparams.text_embed_dim,
            use_mhsa=False,
            apply_reduce_dim=self.hparams.apply_reduce_dim,
            reduced_embed_dim=news_embed_dim,
            num_heads=None,
            query_dim=None,
            dropout_probability=self.hparams.dropout_probability,
        )

        # initialize category encoder
        if self.hparams.use_categ_bias:
            # load pretrained category embeddings
            pretrained_categ_embeddings = torch.from_numpy(
                np.load(pretrained_categ_embeddings_path)
            ).float()
            self.categ_encoder = LinearEncoder(
                pretrained_embeddings=pretrained_categ_embeddings,
                from_pretrained=True,
                freeze_pretrained_emb=False,
                num_categories=self.num_categ_classes,
                embed_dim=None,
                use_dropout=True,
                dropout_probability=self.hparams.dropout_probability,
                linear_transform=False,
                output_dim=None,
            )

        # initialize news encoder
        self.news_encoder = NewsEncoder(
            dataset_attributes=self.hparams.dataset_attributes,
            attributes2encode=self.hparams.attributes2encode,
            concatenate_inputs=False,
            text_encoder=text_encoder,
            category_encoder=None,
            entity_encoder=None,
            combine_vectors=False,
            combine_type=None,
            input_dim=None,
            query_dim=None,
            output_dim=None,
        )

        # initialize user encoder, if needed
        if not self.hparams.late_fusion:
            self.user_encoder = UserEncoder(
                input_dim=news_embed_dim,
                num_context_codes=self.hparams.num_context_codes,
                context_code_dim=self.hparams.context_code_dim,
            )

        # initialize click predictor
        self.click_predictor = DotProduct()
        if self.hparams.score_type == "weighted":
            self.target_aware_attn = TargetAwareAttention(input_dim=news_embed_dim)

        # collect outputs of `*_step`
        self.training_step_outputs = {key: [] for key in self.step_outputs["train"]}
        self.val_step_outputs = {key: [] for key in self.step_outputs["val"]}
        self.test_step_outputs = {key: [] for key in self.step_outputs["test"]}

        # metric objects for calculating and averaging performance across batches
        rec_metrics = MetricCollection(
            {
                "auc": AUROC(task="binary", num_classes=2),
                "mrr": RetrievalMRR(),
            }
        )
        ndcg_metrics_dict = {}
        for k in self.hparams.top_k_list:
            ndcg_metrics_dict["ndcg@" + str(k)] = RetrievalNormalizedDCG(top_k=k)
        rec_metrics.add_metrics(ndcg_metrics_dict)

        self.train_rec_metrics = rec_metrics.clone(prefix="train/")
        self.val_rec_metrics = rec_metrics.clone(prefix="val/")
        self.test_rec_metrics = rec_metrics.clone(prefix="test/")

        categ_div_metrics_dict = {}
        for k in self.hparams.top_k_list:
            categ_div_metrics_dict["categ_div@" + str(k)] = Diversity(
                num_classes=self.num_categ_classes, top_k=k
            )
        categ_div_metrics = MetricCollection(categ_div_metrics_dict)

        sent_div_metrics_dict = {}
        for k in self.hparams.top_k_list:
            sent_div_metrics_dict["sent_div@" + str(k)] = Diversity(
                num_classes=self.num_sent_classes, top_k=k
            )
        sent_div_metrics = MetricCollection(sent_div_metrics_dict)

        categ_pers_metrics_dict = {}
        for k in self.hparams.top_k_list:
            categ_pers_metrics_dict["categ_pers@" + str(k)] = Personalization(
                num_classes=self.num_categ_classes, top_k=k
            )
        categ_pers_metrics = MetricCollection(categ_pers_metrics_dict)

        sent_pers_metrics_dict = {}
        for k in self.hparams.top_k_list:
            sent_pers_metrics_dict["sent_pers@" + str(k)] = Personalization(
                num_classes=self.num_sent_classes, top_k=k
            )
        sent_pers_metrics = MetricCollection(sent_pers_metrics_dict)

        self.test_categ_div_metrics = categ_div_metrics.clone(prefix="test/")
        self.test_sent_div_metrics = sent_div_metrics.clone(prefix="test/")
        self.test_categ_pers_metrics = categ_pers_metrics.clone(prefix="test/")
        self.test_sent_pers_metrics = sent_pers_metrics.clone(prefix="test/")

    def forward(self, batch: RecommendationBatch) -> torch.Tensor:
        # encode history
        hist_news_vector = self.news_encoder(batch["x_hist"])
        hist_news_vector_agg, mask_hist = to_dense_batch(hist_news_vector, batch["batch_hist"])

        # encode candidates
        cand_news_vector = self.news_encoder(batch["x_cand"])
        cand_news_vector_agg, _ = to_dense_batch(cand_news_vector, batch["batch_cand"])

        if not self.hparams.late_fusion:
            if self.hparams.use_categ_bias:
                # encode clicked categories
                hist_categ_vector = self.categ_encoder(batch["x_hist"]["category"])

                # encode candidate categories
                cand_categ_vector = self.categ_encoder(batch["x_cand"]["category"])

                categ_bias = pairwise_cosine_similarity(hist_categ_vector, cand_categ_vector)

                # tensor mask to avoid NaN for news which were not clicked or are not candidates for a specific user
                categ_bias_agg, _ = to_dense_batch(categ_bias, batch["batch_hist"])
                mask_cand = batch["batch_cand"].unsqueeze(dim=0).repeat(categ_bias_agg.shape[0], 1)
                cand_idx = [torch.where(mask_cand[i] == i)[0] for i in range(mask_cand.shape[0])]
                mask = torch.zeros(mask_cand.shape, dtype=torch.bool, device=categ_bias_agg.device)
                for i in range(mask.shape[0]):
                    mask[i, cand_idx[i]] = True
                mask = mask.unsqueeze(dim=1).repeat(1, categ_bias_agg.shape[1], 1)
                categ_bias_agg = categ_bias_agg.masked_fill_(mask, 0)

                user_vector = self.user_encoder(
                    embeddings=hist_news_vector_agg,
                    attn_mask=mask_hist,
                    bias=categ_bias_agg,
                )
            else:
                user_vector = self.user_encoder(
                    embeddings=hist_news_vector_agg, attn_mask=mask_hist, bias=None
                )

            # click scores
            scores = self.click_predictor(cand_news_vector_agg, user_vector.permute(0, 2, 1))
            if self.hparams.score_type == "max":
                scores = scores.max(dim=2)[0]
            elif self.hparams.score_type == "mean":
                scores = scores.mean(dim=2)
            elif self.hparams.score_type == "weighted":
                scores = self.target_aware_attn(
                    query=user_vector, key=cand_news_vector_agg, value=scores
                )
            else:
                raise ValueError("Invalid method of aggregating scores.")

        else:
            # aggregate embeddings of clicked news
            hist_size = torch.tensor(
                [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
                device=self.device,
            )
            user_vector = torch.div(hist_news_vector_agg.sum(dim=1), hist_size.unsqueeze(dim=-1))

            # click scores
            scores = self.click_predictor(
                user_vector.unsqueeze(dim=1), cand_news_vector_agg.permute(0, 2, 1)
            )

        return scores, user_vector

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
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        scores, user_vector = self.forward(batch)

        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        candidate_categories, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"])
        candidate_sentiments, _ = to_dense_batch(batch["x_cand"]["sentiment"], batch["batch_cand"])

        clicked_categories, mask_hist = to_dense_batch(
            batch["x_hist"]["category"], batch["batch_hist"]
        )
        clicked_sentiments, _ = to_dense_batch(batch["x_hist"]["sentiment"], batch["batch_hist"])

        # loss computation
        if self.hparams.loss == "cross_entropy_loss":
            loss = self.criterion(scores, y_true)
        else:
            # indices of positive pairs for loss calculation
            pos_idx = [torch.where(y_true[i])[0] for i in range(mask_cand.shape[0])]
            pos_repeats = torch.tensor([len(pos_idx[i]) for i in range(len(pos_idx))])
            q_p = torch.repeat_interleave(torch.arange(mask_cand.shape[0]), pos_repeats)
            p = torch.cat(pos_idx)

            # indices of negative pairs for loss calculation
            neg_idx = [
                torch.where(~y_true[i].bool())[0][
                    : len(torch.where(mask_cand[i])[0]) - pos_repeats[i]
                ]
                for i in range(mask_cand.shape[0])
            ]
            neg_repeats = torch.tensor([len(t) for t in neg_idx])
            q_n = torch.repeat_interleave(torch.arange(mask_cand.shape[0]), neg_repeats)
            n = torch.cat(neg_idx)

            indices_tuple = (q_p, p, q_n, n)

            if not self.hparams.dual_loss_training:
                loss = self.criterion(
                    embeddings=scores,
                    labels=None,
                    indices_tuple=indices_tuple,
                    ref_emb=None,
                    ref_labels=None,
                )
            else:
                ce_loss = self.ce_criterion(scores, y_true)
                scl_loss = self.scl_criterion(
                    embeddings=scores,
                    labels=None,
                    indices_tuple=indices_tuple,
                    ref_emb=None,
                    ref_labels=None,
                )
                loss = (
                    1 - self.hparams.dual_loss_coef
                ) * ce_loss + self.hparams.dual_loss_coef * scl_loss

        if not self.hparams.late_fusion:
            disagreement_loss = pairwise_cosine_similarity_3d(
                user_vector, user_vector, zero_diagonal=True
            ).mean()
        else:
            disagreement_loss = pairwise_cosine_similarity(
                user_vector, user_vector, zero_diagonal=True
            ).mean()
        loss += disagreement_loss

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

        user_ids = batch["user_ids"]
        cand_news_ids = batch["x_cand"]["news_ids"]

        return (
            loss,
            preds,
            targets,
            cand_news_size,
            hist_news_size,
            target_categories,
            target_sentiments,
            hist_categories,
            hist_sentiments,
            user_ids,
            cand_news_ids,
        )

    def training_step(self, batch: RecommendationBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, _, _, _, _, _, _, _ = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # collect step outputs for metric computation
        self.training_step_outputs = self._collect_step_outputs(
            outputs_dict=self.training_step_outputs, local_vars=locals()
        )

        return loss

    def on_train_epoch_end(self) -> None:
        # update and log metrics
        preds = self._gather_step_outputs(self.training_step_outputs, "preds")
        targets = self._gather_step_outputs(self.training_step_outputs, "targets")
        cand_news_size = self._gather_step_outputs(self.training_step_outputs, "cand_news_size")
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        # update metrics
        self.train_rec_metrics(preds, targets, **{"indexes": indexes})

        # log metrics
        self.log_dict(
            self.train_rec_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # clear memory for the next epoch
        self.training_step_outputs = self._clear_epoch_outputs(self.training_step_outputs)

    def validation_step(self, batch: RecommendationBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, _, _, _, _, _, _, _ = self.model_step(batch)

        # update and log metrics
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

        preds = self._gather_step_outputs(self.val_step_outputs, "preds")
        targets = self._gather_step_outputs(self.val_step_outputs, "targets")
        cand_news_size = self._gather_step_outputs(self.val_step_outputs, "cand_news_size")
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        # update metrics
        self.val_rec_metrics(preds, targets, **{"indexes": indexes})

        # log metrics
        self.log_dict(
            self.val_rec_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # clear memory for the next epoch
        self.val_step_outputs = self._clear_epoch_outputs(self.val_step_outputs)

    def test_step(self, batch: RecommendationBatch, batch_idx: int):
        (
            loss,
            preds,
            targets,
            cand_news_size,
            hist_news_size,
            target_categories,
            target_sentiments,
            hist_categories,
            hist_sentiments,
            user_ids,
            cand_news_ids,
        ) = self.model_step(batch)

        # update and log metrics
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

        user_ids = self._gather_step_outputs(self.test_step_outputs, "user_ids")
        cand_news_ids = self._gather_step_outputs(self.test_step_outputs, "cand_news_ids")

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

        # save recommendations
        if self.hparams.save_recs:
            recommendations_dico = self._get_recommendations(
                user_ids=user_ids,
                news_ids=cand_news_ids,
                scores=preds,
                cand_news_size=cand_news_size,
            )
            print(recommendations_dico)
            self._save_recommendations(
                recommendations=recommendations_dico, fpath=self.hparams.recs_fpath
            )

        # clear memory for the next epoch
        self.test_step_outputs = self._clear_epoch_outputs(self.test_step_outputs)
