import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torchmetrics import MeanMetric, MinMetric

from newsreclib.data.components.batch import RecommendationBatch
from newsreclib.models.components.losses import SupConLoss


class AbstractRecommneder(LightningModule):
    """Base class for all other recommenders.

    Implements common functionalities shared by all recommendation models.

    Attributes:
        outputs:
            A dictionary of user-defined attributes needed for metric calculation at the end of each `*_step` of the pipeline.
        optimizer:
            Optimizer used for model training.
        scheduler:
            Learning rate scheduler.
    """

    def __init__(
        self,
        outputs: Dict[str, List[str]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # collect outputs of `*_step`
        self.step_outputs = {}
        for stage in outputs:
            stage_outputs = outputs[stage]
            self.step_outputs[stage] = {key: [] for key in stage_outputs}

    def forward(self, batch: RecommendationBatch) -> torch.Tensor:
        raise NotImplementedError

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        if "valid" in self.metrics.keys():
            for metric_group in self.metrics["valid"].keys():
                self.metrics["valid"][metric_group].reset()

    def model_step(self, batch: RecommendationBatch):
        raise NotImplementedError

    def training_step(self, batch: RecommendationBatch, batch_idx: int):
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        raise NotImplementedError

    def validation_step(self, batch: RecommendationBatch, batch_idx: int):
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        raise NotImplementedError

    def test_step(self, batch: RecommendationBatch, batch_idx: int):
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _init_embedding(self, filepath: str) -> torch.Tensor:
        return torch.from_numpy(np.load(filepath)).float().to(self.device)

    def _get_loss(self, criterion) -> Union[Callable, Tuple[Callable, Callable]]:
        """Returns an instantiated loss object based on the specified criterion."""
        if criterion == "cross_entropy_loss":
            loss = CrossEntropyLoss()
        elif criterion == "sup_con_loss":
            loss = SupConLoss()
        elif criterion == "dual_loss":
            loss = CrossEntropyLoss(), SupConLoss()
        else:
            raise ValueError(f"Loss not defined: {self.hparams.loss}")

        return loss

    def _collect_model_outputs(self, vector: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Concatenates model outputs for metric computation."""
        model_output = torch.cat([vector[n][mask[n]] for n in range(mask.shape[0])], dim=0)

        return model_output

    def _collect_step_outputs(
        self, outputs_dict: Dict[str, List[torch.Tensor]], local_vars
    ) -> Dict[str, List[torch.Tensor]]:
        """Collects user-defined attributes of outputs at the end of a `*_step` in dict."""
        for key in outputs_dict.keys():
            val = local_vars.get(key, [])
            outputs_dict[key].append(val)
        return outputs_dict

    def _gather_step_outputs(
        self, outputs_dict: Optional[Dict[str, List[torch.Tensor]]], key: str
    ) -> torch.Tensor:
        if key not in outputs_dict.keys():
            raise AttributeError(f"{key} not in {outputs_dict}")

        outputs = torch.cat([output for output in outputs_dict[key]])
        return outputs

    def _clear_epoch_outputs(
        self, outputs_dict: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, List[torch.Tensor]]:
        """Clears the outputs collected during each epoch."""
        for key in outputs_dict.keys():
            outputs_dict[key].clear()

        return outputs_dict

    def _get_recommendations(
        self,
        user_ids: torch.Tensor,
        news_ids: torch.Tensor,
        scores: torch.Tensor,
        cand_news_size: torch.Tensor,
    ) -> Dict[int, Dict[str, List[Any]]]:
        """Returns the recommendations and corresponding scores for the given users.

        Attributes:
            user_ids (torch.Tensor): IDs of users.
            news_ids (torch.Tensor): IDs of the candidates news.
            scores (torch.Tensor): Predicted scores for the candidate news.
            cand_news_size (torch.Tensor): Number of candidate news for each user.

        Returns:
            Dict[int, Dict[str, List[Any]]]: A dictionary with user IDs as keys and an inner dictionary of recommendations and corresponding scores as values.
        """
        users = torch.repeat_interleave(user_ids.detach().cpu(), cand_news_size).tolist()
        users = ["U" + str(uid) for uid in users]
        news = ["N" + str(nid) for nid in news_ids.detach().cpu().tolist()]
        scores = scores.detach().cpu().tolist()

        # dictionary of recommendations and scores for each user
        recommendations_dico = {}
        for user, news, score in zip(users, news, scores):
            if user not in recommendations_dico:
                recommendations_dico[user] = {}
            recommendations_dico[user][news] = score

        return recommendations_dico

    def _save_recommendations(self, recommendations: Dict[int, Dict[str, List[Any]]], fpath: str):
        with open(fpath, "w") as f:
            json.dump(recommendations, f)
