Quick Start
===========

NewsRecLib's entry point is the function ``train``, which accepts a
configuration file that drives the entire experiment.

Basic Configuration
-------------------
The following example shows how to train a `NRMS` model on the
`MINDsmall` dataset with the original configurations (i.e., news
encoder contextualizing pretrained embeddings, model trained by
optimizing cross-entropy loss), using an existing configuration file.

.. code:: python

    python newsreclib/train.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent

In the basic experiment, the experiment configuration only specifies
required hyperparameter values which are not set in the configurations
of the corresponding modules.

.. code:: yaml

    defaults:
        - override /data: mind_rec_bert_sent.yaml
        - override /model: nrms.yaml
        - override /callbacks: default.yaml
        - override /logger: many_loggers.yaml
        - override /trainer: gpu.yaml
    data:
        dataset_size: "small"
    model:
        use_plm: False
        pretrained_embeddings_path: ${paths.data_dir}MINDsmall_train/transformed_word_embeddings.npy
        embed_dim: 300
        num_heads: 15

Advanced Configuration
----------------------
The advanced scenario depicts a more complex experimental setting.
Users cn overwrite from the main experiment configuration file any of the
predefined module configurations. The following code snippet shows how
to train a `NRMS` model with a PLM-based news encoder,
and a supervised contrastive loss objective instead of the default settings.

.. code:: python

    python newsreclib/train.py experiment=nrms_mindsmall_plm_supconloss_bertsent

This is achieved by creating an experiment configuration file with the
following specifications:

.. code:: yaml

    defaults:
        - override /data: mind_rec_bert_sent.yaml
        - override /model: nrms.yaml
        - override /callbacks: default.yaml
        - override /logger: many_loggers.yaml
        - override /trainer: gpu.yaml
    data:
        dataset_size: "small"
        use_plm: True
        tokenizer_name: "roberta-base"
        tokenizer_use_fast: True
        tokenizer_max_len: 96
    model:
        loss: "sup_con_loss"
        temperature: 0.1
        use_plm: True
        plm_model: "roberta-base"
        frozen_layers: [0, 1, 2, 3, 4, 5, 6, 7]
        pretrained_embeddings_path: None
        embed_dim: 768
        num_heads: 16

Alternatively, configurations can be overridden from the command line, as follows:

.. code:: python

    python newsreclib/train.py experiment=nrms_mindsmall_plm_supconloss_bertsent data.batch_size=128
