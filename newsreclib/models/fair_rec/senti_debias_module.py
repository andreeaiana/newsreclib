# Code adapted from https://github.com/wuch15/Sentiment-debiasing/blob/main/model.py

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torchmetrics import MaxMetric, MetricCollection
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from newsreclib.data.components.batch import RecommendationBatch
from newsreclib.metrics.diversity import Diversity
from newsreclib.metrics.personalization import Personalization
from newsreclib.models.abstract_recommender import AbstractRecommneder
from newsreclib.models.components.encoders.news.news import NewsEncoder
from newsreclib.models.components.encoders.news.text import PLM, MHSAAddAtt
from newsreclib.models.components.encoders.user.nrms import UserEncoder
from newsreclib.models.components.layers.click_predictor import DotProduct


class Discriminator(nn.Module):
    """Implements the discriminator component of the SentiDebias recommendation model.

    Reference: Wu, Chuhan, Fangzhao Wu, Tao Qi, Wei-Qiang Zhang, Xing Xie, and Yongfeng Huang. "Removing AI’s sentiment manipulation of personalized news delivery." Humanities and Social Sciences Communications 9, no. 1 (2022): 1-9.

    For further details, please refer to the `paper <https://www.nature.com/articles/s41599-022-01473-1>`_

    Attributes:
        input_dim:
            The number of input features in the first linear layer.
        hidden_dim:
            The number of input features in the hidden state.
        output_dim:
            The number of output features.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, hist_news_vector: torch.Tensor, cand_news_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_hist_sent = self.linear2(torch.tanh(self.linear1(hist_news_vector)))
        pred_cand_sent = self.linear2(torch.tanh(self.linear1(cand_news_vector)))

        return pred_hist_sent, pred_cand_sent


class Generator(nn.Module):
    """Implements the generator component of the SentiDebias recommendation model.

    Reference: Wu, Chuhan, Fangzhao Wu, Tao Qi, Wei-Qiang Zhang, Xing Xie, and Yongfeng Huang. "Removing AI’s sentiment manipulation of personalized news delivery." Humanities and Social Sciences Communications 9, no. 1 (2022): 1-9.

    For further details, please refer to the `paper <https://www.nature.com/articles/s41599-022-01473-1>`_

    Attributes:
        dataset_attributes:
            List of news features available in the used dataset.
        attributes2encode:
            List of news features used as input to the news encoder.
        late_fusion:
            If ``True``, it trains the model with the standard `early fusion` approach (i.e., learns an explicit user embedding). If ``False``, it use the `late fusion`.
        use_plm:
            If ``True``, it will process the data for a petrained language model (PLM) in the news encoder. If ``False``, it will tokenize the news title and abstract to be used initialized with pretrained word embeddings.
        pretrained_embeddings_path:
            The filepath for the pretrained embeddings.
        plm_model:
            Name of the pretrained language model.
        frozen_layers:
            List of layers to freeze during training.
        embed_dim:
            Number of features in the text vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        query_dim:
            The number of features in the query vector.
        dropout_probability:
            Dropout probability.
        sentiment_encoder:
            The sentiment encoder module.
    """

    def __init__(
        self,
        dataset_attributes: List[str],
        attributes2encode: List[str],
        late_fusion: bool,
        use_plm: bool,
        pretrained_embeddings_path: Optional[str],
        plm_model: Optional[str],
        frozen_layers: Optional[List[int]],
        embed_dim: int,
        num_heads: int,
        query_dim: int,
        dropout_probability: float,
        sentiment_encoder: nn.Module,
    ) -> None:
        super().__init__()

        self.late_fusion = late_fusion

        # initialize text encoder
        if not use_plm:
            # pretrained embeddings + contextualization
            assert isinstance(pretrained_embeddings_path, str)
            pretrained_embeddings = torch.from_numpy(np.load(pretrained_embeddings_path)).float()
            text_encoder = MHSAAddAtt(
                pretrained_embeddings=pretrained_embeddings,
                embed_dim=embed_dim,
                num_heads=num_heads,
                query_dim=query_dim,
                dropout_probability=dropout_probability,
            )
        else:
            # use PLM
            assert isinstance(plm_model, str)
            text_encoder = PLM(
                plm_model=plm_model,
                frozen_layers=frozen_layers,
                embed_dim=embed_dim,
                use_mhsa=True,
                apply_reduce_dim=False,
                reduced_embed_dim=None,
                num_heads=num_heads,
                query_dim=query_dim,
                dropout_probability=dropout_probability,
            )

        # initialize news encoder
        self.news_encoder = NewsEncoder(
            dataset_attributes=dataset_attributes,
            attributes2encode=attributes2encode,
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
        if not self.late_fusion:
            self.user_encoder = UserEncoder(
                news_embed_dim=embed_dim,
                num_heads=num_heads,
                query_dim=query_dim,
            )

        # initialize sentiment encoder
        self.sentiment_encoder = sentiment_encoder

        # initialize click predictors
        self.click_predictor_bias_free = DotProduct()
        self.click_predictor_bias_aware = DotProduct()

    def forward(
        self, batch: RecommendationBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # encode history
        hist_news_vector = self.news_encoder(batch["x_hist"])
        hist_news_vector_agg, mask_hist = to_dense_batch(hist_news_vector, batch["batch_hist"])

        # encode candidates
        cand_news_vector = self.news_encoder(batch["x_cand"])
        cand_news_vector_agg, _ = to_dense_batch(cand_news_vector, batch["batch_cand"])

        # encode clicked sentiments
        hist_sent_vector = self.sentiment_encoder(batch["x_hist"]["sentiment"])
        hist_sent_vector_agg, _ = to_dense_batch(hist_sent_vector, batch["batch_hist"])

        # encode candidate sentiments
        cand_sent_vector = self.sentiment_encoder(batch["x_cand"]["sentiment"])
        cand_sent_vector_agg, _ = to_dense_batch(cand_sent_vector, batch["batch_cand"])

        if not self.late_fusion:
            # encode user
            # bias-free user embedding
            user_bias_free_vector = self.user_encoder(hist_news_vector_agg)

            # bias-aware user embedding
            user_bias_aware_vector = self.user_encoder(hist_sent_vector_agg)
        else:
            # aggregate embeddings of clicked news
            hist_size = torch.tensor(
                [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
                device=self.device,
            )

            # bias-free user embedding
            user_bias_free_vector = torch.div(
                hist_news_vector_agg.sum(dim=1), hist_size.unsqueeze(dim=-1)
            )

            # bias-aware user embedding
            user_bias_aware_vector = torch.div(
                hist_sent_vector_agg.sum(dim=1), hist_size.unsqueeze(dim=-1)
            )

        # regularization losses
        loss_orth_hist_news = torch.mean(
            (
                torch.div(
                    torch.sum(hist_news_vector * hist_sent_vector, dim=-1),
                    1e-8
                    + torch.linalg.norm(hist_news_vector, dim=1, ord=2)
                    * torch.linalg.norm(hist_sent_vector, dim=1, ord=2),
                )
            ),
            dim=-1,
        )
        loss_orth_cand_news = torch.mean(
            (
                torch.div(
                    torch.sum(cand_news_vector * cand_sent_vector, dim=-1),
                    1e-8
                    + torch.linalg.norm(cand_news_vector, dim=1, ord=2)
                    * torch.linalg.norm(cand_sent_vector, dim=1, ord=2),
                )
            ),
            dim=-1,
        )
        loss_orth_user = torch.div(
            torch.bmm(
                user_bias_free_vector.unsqueeze(dim=1), user_bias_aware_vector.unsqueeze(dim=-1)
            ).squeeze(dim=1),
            (
                1e-8
                + torch.linalg.norm(user_bias_free_vector, dim=1, ord=2)
                * (torch.linalg.norm(user_bias_aware_vector, dim=1, ord=2))
            ).unsqueeze(dim=1),
        )

        loss_orth = (
            torch.abs(loss_orth_hist_news)
            + torch.abs(loss_orth_cand_news)
            + torch.abs(loss_orth_user)
        )
        loss_orth = torch.mean(loss_orth)

        # click scores
        bias_free_scores = self.click_predictor_bias_free(
            user_bias_free_vector.unsqueeze(dim=1), cand_news_vector_agg.permute(0, 2, 1)
        )
        bias_aware_scores = self.click_predictor_bias_aware(
            user_bias_aware_vector.unsqueeze(dim=1), cand_sent_vector_agg.permute(0, 2, 1)
        )
        combined_scores = bias_free_scores + bias_aware_scores

        return (
            combined_scores,
            bias_free_scores,
            loss_orth,
            hist_news_vector,
            cand_news_vector,
        )


class SentiDebiasModule(AbstractRecommneder):
    """Removing AI’s sentiment manipulation of personalized news delivery.

    Reference: Wu, Chuhan, Fangzhao Wu, Tao Qi, Wei-Qiang Zhang, Xing Xie, and Yongfeng Huang. "Removing AI’s sentiment manipulation of personalized news delivery." Humanities and Social Sciences Communications 9, no. 1 (2022): 1-9.

    For further details, please refer to the `paper <https://www.nature.com/articles/s41599-022-01473-1>`_

    Attributes:
        outputs:
            A dictionary of user-defined attributes needed for metric calculation at the end of each `*_step` of the pipeline.
        generator:
            The generator component.
        discriminator:
            The discriminator component.
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
        alpha_coefficient:
            The coefficient that controls the intensity of the adversarial loss.
        beta_coefficient:
            The coefficient that controls the intensity of the orthogonal regularization loss.
        optimizer_generator:
            Optimizer for the generator component.
        optimizer_discriminator:
            Optimizer for the discriminator component.
        scheduler:
            Learning rate scheduler.
    """

    def __init__(
        self,
        outputs: Dict[str, List[str]],
        generator: nn.Module,
        discriminator: nn.Module,
        top_k_list: List[int],
        num_categ_classes: int,
        num_sent_classes: int,
        save_recs: bool,
        recs_fpath: Optional[str],
        optimizer: torch.optim.Optimizer,
        alpha_coefficient: float,
        beta_coefficient: float,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__(
            outputs=outputs,
            optimizer=None,
            scheduler=scheduler,
        )

        self.automatic_optimization = False

        self.num_categ_classes = self.hparams.num_categ_classes + 1
        self.num_sent_classes = self.hparams.num_sent_classes + 1

        if self.hparams.save_recs:
            assert isinstance(self.hparams.recs_fpath, str)

        # initialize loss
        self.rec_loss = self._get_loss("cross_entropy_loss")
        self.a_loss = self._get_loss("cross_entropy_loss")

        # initialize model components
        self.generator = generator
        self.discriminator = discriminator

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

        # for tracking best so far validation loss
        self.val_acc = AUROC(task="binary", num_classes=2)
        self.val_acc_best = MaxMetric()

    def forward(
        self, batch: RecommendationBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.generator(batch)

    def adversarial_loss(self, preds, targets):
        y_true = torch.zeros(preds.shape, device=preds.device)
        for i in range(targets.shape[0]):
            y_true[i, targets[i] - 1] = 1.0

        return self.a_loss(preds, y_true)

    def on_train_start(self) -> None:
        self.val_acc.reset()
        self.val_acc_best.reset()

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
    ]:
        _, bias_free_scores, _, _, _ = self.forward(batch)

        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        candidate_categories, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"])
        candidate_sentiments, _ = to_dense_batch(batch["x_cand"]["sentiment"], batch["batch_cand"])

        clicked_categories, mask_hist = to_dense_batch(
            batch["x_hist"]["category"], batch["batch_hist"]
        )
        clicked_sentiments, _ = to_dense_batch(batch["x_hist"]["sentiment"], batch["batch_hist"])

        # model outputs for metric computation
        preds = self._collect_model_outputs(bias_free_scores, mask_cand)
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
        optimizer_g, optimizer_d = self.optimizers()

        # train generator
        self.toggle_optimizer(optimizer_g)
        (
            combined_scores,
            bias_free_scores,
            loss_orth,
            hist_news_vector,
            cand_news_vector,
        ) = self.generator(batch)
        pred_hist_sent, pred_cand_sent = self.discriminator(hist_news_vector, cand_news_vector)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])

        g_loss = (
            self.rec_loss(combined_scores, y_true)
            + self.hparams.beta_coefficient * loss_orth
            - self.hparams.alpha_coefficient
            * (
                self.adversarial_loss(pred_hist_sent, batch["x_hist"]["sentiment"])
                + self.adversarial_loss(pred_cand_sent, batch["x_cand"]["sentiment"])
            )
        )

        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)
        _, _, _, hist_news_vector, cand_news_vector = self.generator(batch)
        pred_hist_sent, pred_cand_sent = self.discriminator(hist_news_vector, cand_news_vector)
        d_loss = self.adversarial_loss(
            pred_hist_sent, batch["x_hist"]["sentiment"]
        ) + self.adversarial_loss(pred_cand_sent, batch["x_cand"]["sentiment"])

        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # model outputs for metric computation
        preds = self._collect_model_outputs(bias_free_scores, mask_cand)
        targets = self._collect_model_outputs(y_true, mask_cand)
        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )

        # collect step outputs for metric computation
        self.training_step_outputs = self._collect_step_outputs(
            outputs_dict=self.training_step_outputs, local_vars=locals()
        )

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
        preds, targets, cand_news_size, _, _, _, _, _, _, _ = self.model_step(batch)

        # collect step outputs for metric computation
        self.val_step_outputs = self._collect_step_outputs(
            outputs_dict=self.val_step_outputs, local_vars=locals()
        )

    def on_validation_epoch_end(self) -> None:
        # update and log metrics
        preds = self._gather_step_outputs(self.val_step_outputs, "preds")
        targets = self._gather_step_outputs(self.val_step_outputs, "targets")
        cand_news_size = self._gather_step_outputs(self.val_step_outputs, "cand_news_size")
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        self.val_acc(preds, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

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

    def configure_optimizers(self) -> Dict[str, Any]:
        # Override abstract class implementation to support multiple optimizers
        optimizer_generator = self.hparams.optimizer_generator(params=self.generator.parameters())
        optimizer_discriminator = self.hparams.optimizer_discriminator(
            params=self.discriminator.parameters()
        )

        return [optimizer_generator, optimizer_discriminator]
