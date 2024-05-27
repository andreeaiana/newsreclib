import os
import pandas as pd

from newsreclib.utils import pylogger

log = pylogger.get_pylogger(__name__)


def _get_nmb(bucket_info):
    # --- Compute ptb logic first
    # Compute num_clicks and exposures by grouping by 'time_bucket' and 'news_id'
    news_metrics_bucket = (
        bucket_info.groupby([
            "time_bucket", 
            "news_id", 
            "time_bucket_end_hour", 
            "time_bucket_start_hour"
        ])
        .agg(num_clicks=("clicked", "sum"), exposures=("clicked", "count"))
        .reset_index()
    )

    # Compute total number of impressions per time bucket
    total_impressions = (
        bucket_info.groupby("time_bucket").size().reset_index(name="total_impressions")
    )

    # Merge to get the total impressions per time bucket alongside the news_metrics
    news_metrics_bucket = pd.merge(
        news_metrics_bucket, total_impressions, on="time_bucket"
    )

    # --- Compute acc logic now
    # Sort the DataFrame by 'news_id' and 'time_bucket' to ensure correct order for cumulative sum
    news_metrics_bucket = news_metrics_bucket.sort_values(by=["news_id", "time_bucket"])

    # Calculate cumulative sums for 'num_clicks' and 'exposures' to get 'num_clicks' and 'exposures'
    news_metrics_bucket["num_clicks"] = news_metrics_bucket.groupby("news_id")[
        "num_clicks"
    ].cumsum()

    return news_metrics_bucket


def get_news_metrics_bucket(bucket_info, path, article2published, matrix_size=5):
    # -- Select logic type
    news_metrics_bucket = _get_nmb(bucket_info)

    # -- Clicks ratio
    total_clicks = (
        news_metrics_bucket.groupby("time_bucket")["num_clicks"]
        .sum()
        .reset_index(name="total_clicks")
    )

    # -- Merge this information back with the original DataFrame
    news_metrics_bucket = pd.merge(news_metrics_bucket, total_clicks, on="time_bucket")

    # -- Calculate 'clicks_ratio' as a percentage
    news_metrics_bucket["clicks_ratio"] = (
        news_metrics_bucket["num_clicks"] / news_metrics_bucket["total_clicks"]
    )

    # -- Replace any potential NaN values with 0 (in case there are time buckets with 0 clicks leading to division by 0)
    news_metrics_bucket["clicks_ratio"] = news_metrics_bucket["clicks_ratio"].fillna(0)

    # -- Normalize variables before processing
    news_metrics_bucket["clicks_ratio"] = (
        news_metrics_bucket["clicks_ratio"] - news_metrics_bucket["clicks_ratio"].min()
    ) / (
        news_metrics_bucket["clicks_ratio"].max()
        - news_metrics_bucket["clicks_ratio"].min()
    )

    # -- map publish time for articles
    news_metrics_bucket["news_pb_time"] = news_metrics_bucket["news_id"].map(
        article2published
    )

    # -- Save news metrics bucket into csv and pickle
    news_metrics_bucket.to_pickle(path)
    log.info(f"(News metrics bucket file created!")

    return news_metrics_bucket
