from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.utils import to_dense_batch
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from newsreclib.data.components.batch import RecommendationBatch
from newsreclib.metrics.diversity import Diversity
from newsreclib.metrics.personalization import Personalization
from newsreclib.models.abstract_recommender import AbstractRecommneder
from newsreclib.models.components.encoders.news.text import CNNPersAtt
from newsreclib.models.components.encoders.user.npa import UserEncoder
from newsreclib.models.components.layers.click_predictor import DotProduct
from newsreclib.models.components.layers.projection import UserProjection


class NPAModule(AbstractRecommneder):
    """NPA: neural news recommendation with personalized attention

    Reference: Wu, Chuhan, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. "NPA: neural news recommendation with personalized attention." In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2576-2584. 2019.

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3292500.3330665>`_

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
        pretrained_embeddings_path:
            The filepath for the pretrained embeddings.
        text_embed_dim:
            The number of features in the text vector.
        user_embed_dim:
            The number of features in the user vector.
        num_users:
            The number of users.
        num_filters:
            The number of filters in the ``CNN`` of the text encoder.
        window_size:
            The window size in the ``CNN`` of the text encoder.
        word_pref_query_dim:
            The number of features in the word preference query vector.
        news_pref_query_dim:
            The number of features in the news preference query vector.
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
        outputs: Dict[str, List[str]],
        dual_loss_training: bool,
        dual_loss_coef: Optional[float],
        loss: str,
        late_fusion: bool,
        temperature: Optional[float],
        pretrained_embeddings_path: str,
        text_embed_dim: int,
        user_embed_dim: int,
        num_users: int,
        num_filters: int,
        window_size: int,
        word_pref_query_dim: int,
        news_pref_query_dim: int,
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
        pretrained_embeddings = self._init_embedding(
            filepath=self.hparams.pretrained_embeddings_path
        )

        num_users = self.hparams.num_users + 1
        self.user_projection = UserProjection(
            num_users=num_users,
            user_embed_dim=self.hparams.user_embed_dim,
            dropout_probability=self.hparams.dropout_probability,
        )
        self.news_encoder = CNNPersAtt(
            pretrained_embeddings=pretrained_embeddings,
            text_embed_dim=self.hparams.text_embed_dim,
            user_embed_dim=self.hparams.user_embed_dim,
            num_filters=self.hparams.num_filters,
            window_size=self.hparams.window_size,
            query_dim=self.hparams.word_pref_query_dim,
            dropout_probability=self.hparams.dropout_probability,
        )

        # initialize user encoder, if needed
        if not self.hparams.late_fusion:
            self.user_encoder = UserEncoder(
                user_embed_dim=self.hparams.user_embed_dim,
                num_filters=self.hparams.num_filters,
                preference_query_dim=self.hparams.news_pref_query_dim,
                dropout_probability=self.hparams.dropout_probability,
            )

        # initialize click predictor
        self.click_predictor = DotProduct()

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
        # history length
        _, mask_hist = to_dense_batch(batch["x_hist"]["title"], batch["batch_hist"])
        hist_size = torch.tensor(
            [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
            device=self.device,
        )

        # candidate news length
        _, mask_cand = to_dense_batch(batch["x_cand"]["title"], batch["batch_cand"])
        cand_size = torch.tensor(
            [torch.where(mask_cand[i])[0].shape[0] for i in range(mask_cand.shape[0])],
            device=self.device,
        )

        # project users
        projected_users = self.user_projection(batch["user_idx"])

        # encode user history
        hist_news_vector = self.news_encoder(batch["x_hist"]["title"], hist_size, projected_users)
        hist_news_vector_agg, _ = to_dense_batch(hist_news_vector, batch["batch_hist"])

        # encode candidate news
        cand_news_vector = self.news_encoder(batch["x_cand"]["title"], cand_size, projected_users)
        cand_news_vector_agg, _ = to_dense_batch(cand_news_vector, batch["batch_cand"])

        if not self.hparams.late_fusion:
            # encode user
            user_vector = self.user_encoder(hist_news_vector_agg, projected_users)
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
