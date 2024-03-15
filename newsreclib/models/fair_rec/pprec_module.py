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
from newsreclib.models.components.encoders.news.news import NewsEncoder
from newsreclib.models.components.encoders.news.text import PLM, MHSAAddAtt
from newsreclib.models.components.encoders.news.popularity_predictor import TimeAwareNewsPopularityPredictor
from newsreclib.models.components.encoders.user.pprec import PopularityAwareUserEncoder


class TimeDistributed(nn.Module):
    """
    Reference: https://deepinout.com/pytorch/pytorch-questions/143_pytorch_tensorflows_timedistributed_equivalent_in_pytorch.html

    TimeDistributed is originally from Keras framework, however, no official function has been 
    provided by the PyTorch. 
    """

    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        batch_size, time_steps, features = x.size()
        x = x.view(-1, features)
        outputs = self.layer(x)
        outputs = outputs.view(batch_size, time_steps, -1)
        return outputs


class PPRECModule(AbstractRecommneder):
    """PP-Rec: News Recommendation with Personalized User Interest 
    and Time-aware News Popularity

    Paper: https://aclanthology.org/2021.acl-long.424.pdf
    """

    def __init__(
        self,
        dataset_attributes: List[str],
        attributes2encode: List[str],
        outputs: Dict[str, List[str]],
        dual_loss_training: bool,
        dual_loss_coef: Optional[float],
        loss: str,
        temperature: Optional[float],
        use_plm: bool,
        text_embed_dim: int,
        pretrained_embeddings_path: Optional[str],
        plm_model: Optional[str],
        frozen_layers: Optional[List[int]],
        embed_dim: int,
        num_heads: int,
        query_dim: int,
        pop_num_embeddings: int,
        pop_embedding_dim: int,
        dropout_probability: float,
        use_entities: bool,
        pretrained_entity_embeddings_path: str,
        entity_embed_dim: int,
        top_k_list: List[int],
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

        # initialize loss
        if not self.hparams.dual_loss_training:
            self.criterion = self._get_loss(self.hparams.loss)
        else:
            assert isinstance(self.hparams.dual_loss_coef, float)
            assert self.hparams.loss == "dual_loss"
            self.ce_criterion, self.scl_criterion = self._get_loss(
                self.hparams.loss)

        # initialize text encoder
        if not self.hparams.use_plm:
            # pretrained embeddings + contextualization
            assert isinstance(self.hparams.pretrained_embeddings_path, str)
            pretrained_embeddings = self._init_embedding(
                filepath=self.hparams.pretrained_embeddings_path
            )
            text_encoder = MHSAAddAtt(
                pretrained_embeddings=pretrained_embeddings,
                embed_dim=self.hparams.embed_dim,
                num_heads=self.hparams.num_heads,
                query_dim=self.hparams.query_dim,
                dropout_probability=self.hparams.dropout_probability,
            )
        else:
            # use PLM
            assert isinstance(self.hparams.plm_model, str)
            text_encoder = PLM(
                plm_model=self.hparams.plm_model,
                frozen_layers=self.hparams.frozen_layers,
                embed_dim=self.hparams.embed_dim,
                use_mhsa=True,
                apply_reduce_dim=False,
                reduced_embed_dim=None,
                num_heads=self.hparams.num_heads,
                query_dim=self.hparams.query_dim,
                dropout_probability=self.hparams.dropout_probability,
            )

        # initialize entity encoder
        if self.hparams.use_entities:
            # load pretrained entity embeddings
            assert isinstance(
                self.hparams.pretrained_entity_embeddings_path, str)
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

        # initiliaze knowledge aware news encoder
        self.knowledge_aware_news_encoder = NewsEncoder(
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

        # initialize popularity aware user encoder
        self.popularity_aware_user_encoder = PopularityAwareUserEncoder(
            news_embed_dim=news_encoder_input_dim,
            num_heads=self.hparams.num_heads,
            query_dim=self.hparams.query_dim,
            pop_num_embeddings=self.hparams.pop_num_embeddings,
            pop_embedding_dim=self.hparams.pop_embedding_dim
        )

        # initialize popularity encoder
        self.time_aware_popularity_encoder = TimeAwareNewsPopularityPredictor(
            hidden_size=self.hparam.hidden_dim_pop_predictor,
            rec_num_embeddings=self.hparam.rec_num_emb_pop_predictor,
            rec_embedding_dim=self.hparam.rec_emb_dim_pop_predictor,
        )

        # initialize personalized matching score
        self.personalized_mat_score = DotProduct()

        # initialize personalized aggregator
        self.gate_eta = nn.Linear(hidden_size, 1)

        # collect outputs of `*_step`
        self.training_step_outputs = {key: []
                                      for key in self.step_outputs["train"]}
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
            ndcg_metrics_dict["ndcg@" +
                              str(k)] = RetrievalNormalizedDCG(top_k=k)
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
        hist_news_vector_agg, mask_hist = to_dense_batch(
            hist_news_vector, batch["batch_hist"])

        # encode candidates
        cand_news_vector = self.knowledge_aware_news_encoder(batch["x_cand"])
        cand_news_vector_agg, _ = to_dense_batch(
            cand_news_vector, batch["batch_cand"])

        # encode user
        # TODO: get ctr
        user_vector = self.popularity_aware_user_encoder(
            hist_news_vector_agg, batch["ctr"])

        # compute eta
        eta = self.sigmoid(self.gate_eta(user_vector))

        # compute popularity score (sp)
        sp = self.time_aware_popularity_encoder(
            cand_news_vector_agg, batch["recency"], batch["ctr"])

        # compute personalized matching score (sm)
        sm = self.personalized_mat_score(
            user_vector.unsqueeze(dim=1), cand_news_vector_agg.permute(0, 2, 1)
        )

        # click scores
        scores = (1 - eta) * sm + eta * sp

        return scores

    def on_train_start(self) -> None:
        pass
