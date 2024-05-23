import pandas as pd
import numpy as np

from newsreclib.utils import pylogger

log = pylogger.get_pylogger(__name__)


def aux_lst_f(series):
    """
    Define a custom aggregation function to create tuples of (history_news_id, num_clicks)
    """
    return list(series)


# ---- Click Through Rate variables
def _get_history_ctr(behaviors: pd.DataFrame, news_metrics_bucket: pd.DataFrame) -> any:
    # Explode the 'history' column to individual rows for easier processing
    bhv_hist_explode = behaviors.explode("history")
    bhv_hist_explode = bhv_hist_explode.rename(columns={"history": "history_news_id"})
    bhv_hist_explode["time"] = pd.to_datetime(bhv_hist_explode["time"])

    # Filter the metrics bucket to include only news_ids present in bhv_hist_explode for efficiency
    unique_ids_metrics_bucket = news_metrics_bucket["news_id"].unique().tolist()
    bhv_hist_explode_filter = bhv_hist_explode[
        bhv_hist_explode["history_news_id"].isin(unique_ids_metrics_bucket)
    ]

    # Merge filtered behaviors with metrics on news_id
    merged_df = pd.merge(
        bhv_hist_explode_filter,
        news_metrics_bucket,
        left_on="history_news_id",
        right_on="news_id",
        how="left",
    )

    # Select entries where the behavioral event time is within the time bucket range
    valid_entries = merged_df[
        (merged_df["time"] >= merged_df["time_bucket_start_hour"])
        & (merged_df["time"] <= merged_df["time_bucket_end_hour"])
    ]
    valid_entries = valid_entries.sort_values(by="time", ascending=False)
    final_df = valid_entries.drop_duplicates(
        subset=["history_news_id", "time"], keep="first"
    )

    # Reintegrate CTR data back to behaviors, filling gaps with zero where no data was found
    bhv_hist_ctr = pd.merge(
        bhv_hist_explode_filter,
        final_df[
            [
                "history_news_id",
                "time",
                "num_clicks",
                "clicks_ratio",
            ]
        ],
        on=["history_news_id", "time"],
        how="left",
    )
    bhv_hist_ctr[["num_clicks", "clicks_ratio"]] = (
        bhv_hist_ctr[["num_clicks", "clicks_ratio"]]
        .fillna(0)
        .astype(int)
    )
    bhv_hist_ctr = bhv_hist_ctr.drop_duplicates(
        subset=["history_news_id", "time", "num_clicks"]
    )

    # Reaggregate to match the original data granularity and form a history column with CTR data
    final_df = pd.merge(
        bhv_hist_explode,
        bhv_hist_ctr[
            [
                "history_news_id",
                "impid",
                "uid",
                "user",
                "time",
                "num_clicks",
                "clicks_ratio",
            ]
        ],
        on=["history_news_id", "impid", "uid", "user", "time"],
        how="left",
    )

    final_df[["num_clicks", "clicks_ratio"]] = (
        final_df[["num_clicks", "clicks_ratio"]].fillna(0).astype(int)
    )

    result_df = (
        final_df.groupby(["impid", "uid", "user", "time"])
        .agg(
            {
                "history_news_id": list,
                "num_clicks": aux_lst_f,
                "clicks_ratio": aux_lst_f,
            }
        )
        .reset_index()
    )
    result_df["history_ctr"] = result_df.apply(
        lambda x: list(
            zip(
                x["history_news_id"],
                x["num_clicks"],
                x["clicks_ratio"],
            )
        ),
        axis=1,
    )
    result_df = result_df.rename(columns={"history_news_id": "history"})

    # Validate that merged data matches original behaviors data
    behaviors["time"] = pd.to_datetime(behaviors["time"])
    result_df["time"] = pd.to_datetime(result_df["time"])
    behaviors_ = pd.merge(
        behaviors, result_df, on=["impid", "uid", "user", "time"], how="inner"
    )
    diff_mask = behaviors_["history_x"] != behaviors_["history_y"]
    different_indexes = behaviors_.index[diff_mask].tolist()

    # Ensure no discrepancies exist
    assert len(different_indexes) == 0

    # Drop the 'history_y' column
    behaviors_ = behaviors_.drop(columns=["history_y"])

    # Rename 'history_x' to 'history'
    behaviors_ = behaviors_.rename(columns={"history_x": "history"})
    behaviors_ = behaviors_.rename(
        columns={
            "num_clicks": "hist_num_clicks",
            "clicks_ratio": "hist_clicks_ratio",
        }
    )

    return behaviors_


def _get_candidate_ctr(
    behaviors: pd.DataFrame,
    news_metrics_bucket: pd.DataFrame,
    article2published: any,
) -> any:
    # Explode the 'candidates' column to individual rows for easier processing
    bhv_cand_explode = behaviors.explode("candidates")
    bhv_cand_explode = bhv_cand_explode.rename(columns={"candidates": "cand_news_id"})
    bhv_cand_explode["pb_time"] = bhv_cand_explode["cand_news_id"].map(
        article2published
    )
    # -- In case of empty pb_time
    min_time = bhv_cand_explode["pb_time"].min()
    bhv_cand_explode["pb_time"] = bhv_cand_explode["pb_time"].fillna(min_time)
    bhv_cand_explode["time"] = pd.to_datetime(bhv_cand_explode["time"])

    # Filter the metrics bucket to include only news_ids present in bhv_cand_explode for efficiency
    unique_ids_metrics_bucket = news_metrics_bucket["news_id"].unique().tolist()
    bhv_cand_explode_filter = bhv_cand_explode[
        bhv_cand_explode["cand_news_id"].isin(unique_ids_metrics_bucket)
    ]

    # Merge filtered behaviors with metrics on news_id
    merged_df = pd.merge(
        bhv_cand_explode_filter,
        news_metrics_bucket,
        left_on="cand_news_id",
        right_on="news_id",
        how="left",
    )

    # Select entries where the behavioral event time is within the time bucket range
    valid_entries = merged_df[
        (merged_df["time"] >= merged_df["time_bucket_start_hour"])
        & (merged_df["time"] <= merged_df["time_bucket_end_hour"])
    ]
    valid_entries = valid_entries.sort_values(by="time", ascending=False)
    final_df = valid_entries.drop_duplicates(
        subset=["cand_news_id", "time"], keep="first"
    )

    # Reintegrate CTR data back to behaviors, filling gaps with zero where no data was found
    bhv_cand_ctr = pd.merge(
        bhv_cand_explode_filter,
        final_df[["cand_news_id", "time", "num_clicks"]],
        on=["cand_news_id", "time"],
        how="left",
    )
    bhv_cand_ctr["num_clicks"] = bhv_cand_ctr["num_clicks"].fillna(0)
    bhv_cand_ctr = bhv_cand_ctr.drop_duplicates(
        subset=["cand_news_id", "time", "num_clicks"]
    )

    # Reaggregate to match the original data granularity and form a cand column with CTR data
    final_df = pd.merge(
        bhv_cand_explode,
        bhv_cand_ctr[["cand_news_id", "impid", "uid", "user", "time", "num_clicks"]],
        on=["cand_news_id", "impid", "uid", "user", "time"],
        how="left",
    )
    final_df["num_clicks"] = final_df["num_clicks"].fillna(0).astype(int)

    # Get recency column
    final_df["time"] = pd.to_datetime(final_df["time"])
    final_df["cand_recency"] = (
        final_df["time"] - final_df["pb_time"]
    ).dt.total_seconds() / 3600
    # Deal with inconsistencies of negative recency values
    final_df["cand_recency"] = final_df["cand_recency"].clip(lower=3600)
    final_df["cand_recency"] = final_df["cand_recency"].astype(int)

    # check if there's any negative value on the recency column
    assert False == (final_df["cand_recency"] < 0).any()

    result_df = (
        final_df.groupby(["impid", "uid", "user", "time"])
        .agg(
            {
                "cand_news_id": list,
                "num_clicks": aux_lst_f,
                "cand_recency": aux_lst_f,
            }
        )
        .reset_index()
    )

    result_df = result_df.rename(columns={"cand_news_id": "candidates"})
    result_df = result_df.rename(columns={"num_clicks": "cand_num_clicks"})

    # compute candidates_ctr
    result_df["candidates_ctr"] = result_df.apply(
        lambda x: list(zip(x["candidates"], x["cand_num_clicks"], x["cand_recency"])),
        axis=1,
    )

    # Validate that merged data matches original behaviors data
    behaviors["time"] = pd.to_datetime(behaviors["time"])
    behaviors_ = pd.merge(
        behaviors, result_df, on=["impid", "uid", "user", "time"], how="inner"
    )
    diff_mask = behaviors_["candidates_x"] != behaviors_["candidates_y"]
    different_indexes = behaviors_.index[diff_mask].tolist()

    # Ensure no discrepancies exist
    assert len(different_indexes) == 0

    # Drop the 'candidates_y' column
    behaviors_ = behaviors_.drop(columns=["candidates_y"])

    # Rename 'cand_x' to 'cand'
    behaviors_ = behaviors_.rename(columns={"candidates_x": "candidates"})
    behaviors_ = behaviors_.rename(columns={"num_clicks": "cand_num_clicks"})

    return behaviors_


def _get_ctr(
    behaviors: pd.DataFrame,
    news_metrics_bucket: pd.DataFrame,
    article2published: any,
) -> any:
    """
    Calculate the CTR for each news article over its respective time buckets from the news_metrics_bucket DataFrame.
    It matches news articles by ID and checks that the behavioral event time falls within the designated time buckets.

    Parameters:
        behaviors (pd.DataFrame): DataFrame containing user behavior data.
        news_metrics_bucket (pd.DataFrame): DataFrame with metrics for each news article over specific time buckets.
        row (any): Unused in this snippet, but typically used for row-specific operations.
        article2published (any): Unused in this snippet, could be used for mapping articles to published info.

    Returns:
        pd.DataFrame: Behaviors DataFrame enriched with the CTR information and checks for data consistency.

    Example usage:
        - Input DataFrame row: {'news_id': 'N3128', 'time_bucket': '11/13/2019 14:00 to 15:00', 'num_clicks': 152}
        - Output: CTR values merged back into the original behaviors DataFrame.
    """
    # Assign impid to the index
    behaviors = behaviors.reset_index(names="impid")

    # Split the 'time_bucket' into two separate columns for start and end times
    time_bounds = news_metrics_bucket["time_bucket"].str.split(" to ", expand=True)
    news_metrics_bucket["time_bucket_start_hour"] = time_bounds[0]

    # Convert the new start and end time columns to datetime
    news_metrics_bucket['time_bucket_start_hour'] = pd.to_datetime(
        news_metrics_bucket['time_bucket_start_hour'], 
        errors='coerce'
    )
    news_metrics_bucket['time_bucket_end_hour'] = pd.to_datetime(
        news_metrics_bucket['time_bucket_end_hour'], 
        errors='coerce'
    )

    # Get CTR for history column
    df_history_ctr = _get_history_ctr(
        behaviors=behaviors, news_metrics_bucket=news_metrics_bucket
    )

    # Get CTR for candidates column
    df_candidate_ctr = _get_candidate_ctr(
        behaviors=behaviors,
        news_metrics_bucket=news_metrics_bucket,
        article2published=article2published,
    )

    # Join informations
    behaviors = pd.merge(
        behaviors,
        df_history_ctr[
            [
                "impid",
                "history_ctr",
                "hist_num_clicks",
                "hist_clicks_ratio",
            ]
        ],
        on="impid",
        how="left",
    )
    behaviors = pd.merge(
        behaviors,
        df_candidate_ctr[
            ["impid", "cand_num_clicks", "cand_recency", "candidates_ctr"]
        ],
        on="impid",
        how="left",
    )

    return behaviors