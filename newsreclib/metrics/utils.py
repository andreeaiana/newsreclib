from typing import Dict, Optional, Union

from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from newsreclib.metrics.diversity import Diversity
from newsreclib.metrics.personalization import Personalization


def get_metric(metric_name: str, metric_params: Optional[Dict[str, Union[str, int]]]):
    """Returns a metric object for the specified name.

    Args:
        metric_name:
            Name of the metric.
        metric_params:
            Dictionary of parameters for instantiating the metric object.
    """

    if metric_name == "auc":
        return AUROC(task=metric_params["task"], num_classes=metric_params["num_classes"])
    elif metric_name == "mrr":
        return RetrievalMRR()
    elif "ndcg" in metric_name:
        return RetrievalNormalizedDCG(top_k=metric_params["top_k"])
    elif "categ_div" in metric_name:
        return Diversity(num_classes=metric_params["num_classes"], top_k=metric_params["top_k"])
    elif "sent_div" in metric_name:
        return Diversity(num_classes=metric_params["num_classes"], top_k=metric_params["top_k"])
    elif "categ_pers" in metric_name:
        return Personalization(
            num_classes=metric_params["num_classes"], top_k=metric_params["top_k"]
        )
    elif "sent_pers" in metric_name:
        return Personalization(
            num_classes=metric_params["num_classes"], top_k=metric_params["top_k"]
        )
    else:
        raise ValueError(f"Metric {metric_name} not supported.")
