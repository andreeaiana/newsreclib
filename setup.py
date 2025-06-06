#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="newsreclib",
    version="0.1",
    description="NewsRecLib: A Pytorch-Lightning Library for Benchmarking Neural News Recommenders",
    author="Andreea Iana",
    author_email="andreea.iana@uni-mannheim.de",
    url="https://github.com/andreeaiana/newsreclib",
    install_requires=[
        "colorcet==3.0.1",
        "cmake==3.18.4",
        "hydra-core==1.3.2",
        "lightning==2.2.1",
        # "MulticoreTSNE==0.1",
        "numpy==1.25.0",
        "omegaconf==2.3.0",
        "pandas==1.5.3",
        "torch_geometric==2.3.0",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        "pytorch-lightning==2.0.3",
        "pytorch-metric-learning==2.2.0",
        "requests==2.31.0",
        "rich==13.3.5",
        "scikit-learn==1.2.2",
        "seaborn==0.12.2",
        "sphinx==5.0.2",
        "tokenizers==0.15.2",
        "torchmetrics==1.3.1",
        "tqdm==4.66.3",
        "transformers==4.38.0",
        "wandb==0.15.3",
        "hydra-colorlog==1.2.0",
        "hydra-optuna-sweeper==1.2.0",
        "optuna==2.10.1",
        "pyrootutils==1.0.4",
        "retrying==1.3.4",
        "sentencepiece==0.1.99",
        "vadersentiment==3.3.2",
    ],
    tests_require=["pytest"],
    setup_requires=["flake8", "pre-commit"],
    packages=find_packages(),
    python_requires=">=3.9.16",
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = newsreclib.train:main",
            "eval_command = newsreclib.eval:main",
        ]
    },
)
