<div align="center">

# <img src="docs/source/_static/newsreclib_header.png" alt="NewsRecLib: A PyTorch-Lightning Library for Neural News Recommendation" width="20%">

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![torchmetrics](https://img.shields.io/badge/-TorchMetrics_2.0+-792ee5?logo=torchmetrics&logoColor=white)](https://lightning.ai/docs/pytorch/stable/ecosystem/metrics.html)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![optuna](https://img.shields.io/badge/Optimization-Optuna_1.3-89b8cd)](https://optuna.org/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

</div>

NewsRecLib is a library based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Hydra](https://hydra.cc/)
for the development and evaluation of neural news recommenders (NNR).
The framework is highly configurable and modularized, decoupling core model components from one another.
It enables running experiments from a single configuration file that navigates the pipeline from dataset selection and loading
to model evaluation. NewsRecLib provides implementations of several neural news recommenders, training methods,
standard evaluation benchmarks, hypeparameter optimization algorithms, extensive logging functionalities, and evaluation metrics
(ranging from accuracy-based to beyond accuracy performance evaluation).

The foremost goals of NewsRecLib are to promote *reproducible research* and *rigorous experimental evaluation*.

![NewsRecLib schema](docs/source/_static/framework.png)

## Installation

NewsRecLib requires Python version 3.9 or later.

NewsRecLib requires PyTorch, PyTorch Lightning, and TorchMetrics version 2.0 or later.
If you want to use NewsRecLib with GPU, please ensure CUDA or cudatoolkit version of 11.8.

### Install from source

#### CONDA

```bash
   git clone https://github.com/andreeaiana/newsreclib.git
   cd newsreclib
   conda create --name newsreclib_env python=3.9
   conda activate newsreclib_env
   pip install -e .
```

## Quick Start

NewsRecLib's entry point is the function `train`, which accepts a
configuration file that drives the entire experiment.

## Basic Configuration

The following example shows how to train a `NRMS` model on the
`MINDsmall` dataset with the original configurations (i.e., news
encoder contextualizing pretrained embeddings, model trained by
optimizing cross-entropy loss), using an existing configuration file.

```python
    python newsreclib/train.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent
```

In the basic experiment, the experiment configuration only specifies
required hyperparameter values which are not set in the configurations
of the corresponding modules.

```yaml
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
```

For training the `NRMS` model on the `MINDlarge` dataset, execute the following command:

```python
python newsreclib/train.py experiment=nrms_mindlarge_pretrainedemb_celoss_bertsent
```

To understand how to adjust configuration files when transitioning from smaller to larger datasets, refer to the examples provided in `nrms_mindsmall_pretrainedemb_celoss_bertsent` and `nrms_mindlarge_pretrainedemb_celoss_bertsent`. These files will guide you in scaling your configurations appropriately.

*Note:* The same procedure applies for the advanced configuration shown below.

## Advanced Configuration

The advanced scenario depicts a more complex experimental setting.
Users cn overwrite from the main experiment configuration file any of the
predefined module configurations. The following code snippet shows how
to train a `NRMS` model with a PLM-based news encoder,
and a supervised contrastive loss objective instead of the default settings.

```python
    python newsreclib/train.py experiment=nrms_mindsmall_plm_supconloss_bertsent
```

This is achieved by creating an experiment configuration file with the
following specifications:

```yaml
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
```

Alternatively, configurations can be overridden from the command line, as follows:

```python
    python newsreclib/train.py experiment=nrms_mindsmall_plm_supconloss_bertsent data.batch_size=128
```

## Features

- **Training**
  - Click behavior fusion strategies: early fusion, [late fusion](https://dl.acm.org/doi/pdf/10.1145/3539618.3592062)
  - Training objectives: cross-entropy loss, supervised contrastive loss, dual
  - All optimizers and learning rate schedulers of PyTorch
  - Early stopping
  - Model checkpointing
- **Hyperparameter optimization**
  - Integrated using [Optuna](https://optuna.org/) and Hydra's [Optuna Sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/)
- **Datasets**
  - [Adreesa](https://dl.acm.org/doi/pdf/10.1145/3106426.3109436): 1-week and 3-months
  - [MIND](https://aclanthology.org/2020.acl-main.331.pdf): MINDsmall and MINDlarge
  - [xMIND](https://github.com/andreeaiana/xMIND): all languages, dataset sizes and splits
- **Recommendation Models**
  - General recommenders (GeneralRec)
    - [CAUM](https://dl.acm.org/doi/pdf/10.1145/3477495.3531778) ([code](newsreclib/models/general_rec/caum_module.py), [config](configs/model/caum.yaml))
    - [CenNewsRec](https://aclanthology.org/2020.findings-emnlp.128.pdf) ([code](newsreclib/models/general_rec/cen_news_rec_module.py), [config](configs/model/cen_news_rec.yaml))
    - [DKN](https://dl.acm.org/doi/pdf/10.1145/3178876.3186175) ([code](newsreclib/models/general_rec/dkn_module.py), [config](configs/model/dkn.yaml))
    - [LSTUR](https://aclanthology.org/P19-1033.pdf) ([code](newsreclib/models/general_rec/lstur_module.py), [config](configs/model/lstur.yaml))
    - [MINER](https://aclanthology.org/2022.findings-acl.29.pdf) ([code](newsreclib/models/general_rec/miner_module.py), [config](configs/model/miner.yaml))
    - [MINS](https://ieeexplore.ieee.org/abstract/document/9747149) ([code](newsreclib/models/general_rec/mins_module.py), [config](configs/model/mins.yaml))
    - [NAML](https://www.ijcai.org/proceedings/2019/0536.pdf) ([code](newsreclib/models/general_rec/naml_module.py), [config](configs/model/naml.yaml))
    - [NPA](https://dl.acm.org/doi/pdf/10.1145/3292500.3330665) ([code](newsreclib/models/general_rec/npa_module.py), [config](configs/model/npa.yaml))
    - [NRMS](https://aclanthology.org/D19-1671.pdf) ([code](newsreclib/models/general_rec/nrms_module.py), [config](configs/model/nrms.yaml))
    - [TANR](https://aclanthology.org/P19-1110.pdf) ([code](newsreclib/models/general_rec/tanr_module.py), [config](configs/model/tanr.yaml))
  - Fairness-aware recommenders (FairRec)
    - [MANNeR](https://arxiv.org/pdf/2307.16089.pdf) ([code](newsreclib/models/fair_rec/manner_module.py), [config](configs/model/manner_module.yaml))
    - [SentiDebias](https://www.nature.com/articles/s41599-022-01473-1) ([code](newsreclib/models/fair_rec/senti_debias_module.py), [config](configs/model/senti_debias.yaml))
    - [SentiRec](https://aclanthology.org/2020.aacl-main.6.pdf) ([code](newsreclib/models/fair_rec/sentirec_module.py), [config](configs/model/sentirec.yaml))
- **Evaluation**
  - Integration with [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
  - Accuracy-based metrics: AUROC, MRR, nDCG@k
  - Diversity: entropy
  - Personalization: generalized Jaccard
- **Extensive logging**
  - Logging  and visualization with [WandB](https://wandb.ai/site)
  - Quick export to CSV files
  - Detailed information about training, hyperparmeters, evaluation, metadata

## Contributing

We welcome all contributions to NewsRecLib! You can get involved by contributing code, making improvements to the documentation,
reporting or investigating [bugs and issues](https://github.com/andreeaiana/newsreclib/issues).

## Resources

This repository was inspired by:

- https://github.com/ashleve/lightning-hydra-template
- https://github.com/MeteSertkan/newsrec

Other useful repositories:

- https://github.com/recommenders-team/recommenders

## License

NewsRecLib uses a [MIT License](./LICENSE).

## Citation

We did our best to provide all the bibliographic information of the methods, models, datasets, and techniques available in NewsRecLib
to credit their authors. Please remember to cite them if you use NewsRecLib in your research.

If you use NewsRecLib, please cite the following publication:

```
@inproceedings{iana2023newsreclib,
  title={NewsRecLib: A PyTorch-Lightning Library for Neural News Recommendation},
  author={Iana, Andreea and Glava{\v{s}}, Goran and Paulheim, Heiko},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={296--310},
  year={2023}
}
```
