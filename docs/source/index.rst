.. NewsRecLib documentation master file, created by
   sphinx-quickstart on Fri Aug  4 13:11:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NewsRecLib's documentation!
======================================

NewsRecLib is a library based on `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_
and `Hydra <https://hydra.cc/>`_
for the development and evaluation of neural news recommenders (NNR).
The framework is highly configurable and modularized,
decoupling core model components from one another. It enables running experiments from
a single configuration file that navigates the pipeline from dataset selection and loading
to model evaluation.
NewsRecLib provides implementations of several neural news recommenders,
training methods, standard evaluation benchmarks, hypeparameter optimization algorithms,
extensive logging functionalities, and evaluation metrics
(ranging from accuracy-based to beyond accuracy performance evaluation).

The foremost goals of NewsRecLib are to promote *reproducible research* and
*rigorous experimental evaluation*.

.. figure:: ./_static/framework.png
   :alt: system schema

   NewsrecLib's schema

.. toctree::
   :maxdepth: 1
   :caption: GET STARTED

   guide/introduction
   guide/installation
   guide/quick_start

.. toctree::
   :maxdepth: 1
   :caption: DATASETS

   guide/datasets_intro

.. toctree::
   :maxdepth: 1
   :caption: RECOMMENDERS

   guide/recommenders_intro
   guide/click_behavior_fusion
   guide/training_objectives
   guide/recommenders

.. toctree::
   :maxdepth: 1
   :caption: TRAINING

   guide/callbacks
   guide/hyperparameter_optimization

.. toctree::
   :maxdepth: 1
   :caption: EVALUATION

   guide/metrics_intro

.. toctree::
   :maxdepth: 6
   :caption: API REFERENCE

   newsreclib/newsreclib
   newsreclib/newsreclib.data
   newsreclib/newsreclib.metrics
   newsreclib/newsreclib.models
   newsreclib/newsreclib.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
