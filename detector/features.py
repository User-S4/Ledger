"""Per-query and per-account feature extraction from request logs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _embeddings_from_logs(logs: pd.DataFrame) -> np.ndarray:
    """Stack query_input vectors into a 2-D array."""
    return np.vstack(logs["query_input"].to_numpy())


def compute_query_rate(
    logs: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    account_col: str = "account_id",
) -> pd.Series:
    """
    Queries per minute for each account.

    Uses the span from first to last query per account. Accounts with a
    single query are assigned a rate of 1.0 query/minute.
    """
    ts = pd.to_datetime(logs[time_col])
    grouped = logs.assign(_ts=ts).groupby(account_col)["_ts"]

    counts = grouped.count()
    spans_min = grouped.apply(lambda s: max((s.max() - s.min()).total_seconds() / 60.0, 1 / 60.0))

    return (counts / spans_min).rename("query_rate")


def compute_input_diversity(
    logs: pd.DataFrame,
    *,
    account_col: str = "account_id",
    metric: str = "std",
) -> pd.Series:
    """
    Input diversity per account, measured as mean std or variance across
    embedding dimensions over all queries for that account.
    """
    if metric not in {"std", "var"}:
        raise ValueError("metric must be 'std' or 'var'")

    def _diversity(group: pd.DataFrame) -> float:
        emb = _embeddings_from_logs(group)
        if emb.shape[0] < 2:
            return 0.0
        per_dim = np.std(emb, axis=0) if metric == "std" else np.var(emb, axis=0)
        return float(np.mean(per_dim))

    return logs.groupby(account_col, group_keys=False).apply(_diversity).rename("input_entropy")


def compute_consecutive_distances(
    logs: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    account_col: str = "account_id",
    aggregate: str = "mean",
) -> pd.Series:
    """
    Mean (or other aggregate) Euclidean distance between consecutive
    query embeddings per account, ordered by timestamp.
    """
    if aggregate not in {"mean", "median", "max"}:
        raise ValueError("aggregate must be 'mean', 'median', or 'max'")

    def _distances(group: pd.DataFrame) -> float:
        ordered = group.sort_values(time_col)
        emb = _embeddings_from_logs(ordered)
        if emb.shape[0] < 2:
            return 0.0
        diffs = np.diff(emb, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        if aggregate == "mean":
            return float(np.mean(dists))
        if aggregate == "median":
            return float(np.median(dists))
        return float(np.max(dists))

    return (
        logs.groupby(account_col, group_keys=False)
        .apply(_distances)
        .rename("consecutive_distance")
    )


def compute_low_confidence_rate(
    logs: pd.DataFrame,
    *,
    account_col: str = "account_id",
    confidence_col: str = "confidence_score",
    threshold: float = 0.6,
) -> pd.Series:
    """Fraction of responses with confidence below threshold, per account."""
    low = logs[confidence_col] < threshold
    return (
        logs.assign(_low=low.astype(float))
        .groupby(account_col)["_low"]
        .mean()
        .rename("low_conf_rate")
    )


def compute_account_features(
    logs: pd.DataFrame,
    *,
    low_conf_threshold: float = 0.6,
    diversity_metric: str = "std",
) -> pd.DataFrame:
    """Compute all per-account features and return a single DataFrame."""
    features = pd.concat(
        [
            compute_query_rate(logs),
            compute_input_diversity(logs, metric=diversity_metric),
            compute_consecutive_distances(logs),
            compute_low_confidence_rate(logs, threshold=low_conf_threshold),
        ],
        axis=1,
    )
    features.index.name = "account_id"
    return features.reset_index()
