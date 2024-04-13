import warnings
from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

from newsreclib.utils import pylogger, rich_utils

from halo import Halo
from functools import wraps
from collections import defaultdict
from datetime import datetime
import pandas as pd

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        cfg: Configuration composed by Hydra.
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```

    Args:
        task_func: Function to be wrapped with the decorator.
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict: Dictionary of metrics and corresponding values.
        metric_name: Name of the metric to be retrieved.

    Returns:
        Value of the specified metric.
    """

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def metrics_spinner(func):
    """
    Wraper function to show to the user that the evaluation metrics 
    are being calculated.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Halo(text="Calculating Evaluation Metrics...", spinner="dots"):
            return func(*args, **kwargs)
    return wrapper


def get_article2clicks(behaviors_path_train: str, behaviors_path_dev: str):
    """
    Get a dictionary that maps the number of clicks per hour per news article

    Arguments:
        - behaviors_path_train: path for the train behaviors file
        - behaviors_path_dev: path for the dev behaviors file

    we need both behaviors to compute the estimate of their publish time
    """
    column_names = ["impid", "uid", "time", "history", "impressions"]

    df_behaviors_train = pd.read_table(
        filepath_or_buffer=behaviors_path_train,
        header=None,
        names=column_names,
        usecols=range(len(column_names))
    )

    df_behaviors_val = pd.read_table(
        filepath_or_buffer=behaviors_path_dev,
        header=None,
        names=column_names,
        usecols=range(len(column_names))
    )

    # ---- Join all news behaviors
    total_behaviors = pd.concat([df_behaviors_train, df_behaviors_val])
    total_behaviors['history'] = total_behaviors['history'].fillna(" ")
    total_behaviors.impressions = total_behaviors.impressions.str.split()

    article2published = {}
    article2clicks = defaultdict(list)
    article2impression = {}
    for _, behavior in total_behaviors.iterrows():
        time = datetime.strptime(behavior["time"], "%m/%d/%Y %I:%M:%S %p")
        for article_id_and_clicked in behavior["impressions"]:
            article_id = article_id_and_clicked[:-2]  # article id ex: N113723
            article_clicked = article_id_and_clicked[
                -1
            ]  # 0 (not clicked) and 1 (clicked)
            
            # Keep a dict to know which news articles appeared on the impression list
            article2impression[article_id] = 1

            # Track the first time an article appears and add that time occurance as publish time
            if (
                article_id not in article2published
                or time < article2published[article_id]
            ):
                article2published[article_id] = time

            # Append the hour when the article was clicked
            if article_clicked == "1":
                article2clicks[article_id].append(time)

    # --- Sort article2clicks dictionary
    for article_id, clicks in article2clicks.items():
        clicks.sort()
        for i, click in enumerate(clicks):
            clicks[i] = (click - article2published[article_id]).total_seconds() // 3600

    return article2published, article2clicks, article2impression