Metrics
=======

NewsRecLib provides several evaluation metrics, for evaluating recommendation models
on the following dimensions: classification, ranking, diversity, and personalization.
Note that NewsRecLib relies on `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_
for the metric implementation. Custom metrics are built by extending
the `Metric <https://torchmetrics.readthedocs.io/en/latest/pages/implement.html#torchmetrics.Metric>`_ class.

The user can add any metric available in `All TorchMetrics <https://torchmetrics.readthedocs.io/en/stable/all-metrics.html>`_
or implement a new one, following this `guide <https://torchmetrics.readthedocs.io/en/stable/pages/implement.html>`_.
