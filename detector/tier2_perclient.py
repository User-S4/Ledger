"""Tier-2 per-account threshold detector for model-extraction traffic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import compute_account_features


@dataclass(frozen=True)
class Tier2Thresholds:
    """Configurable per-account feature thresholds."""

    query_rate: float = 30.0
    input_entropy: float = 0.05
    low_conf_rate: float = 0.4


def detect_accounts(
    logs: pd.DataFrame,
    *,
    thresholds: Tier2Thresholds | None = None,
    query_rate_threshold: float | None = None,
    input_entropy_threshold: float | None = None,
    low_conf_rate_threshold: float | None = None,
    low_conf_feature_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Flag accounts whose features exceed any configured threshold.

    Returns a DataFrame with account_id, feature values, and a boolean
    ``flagged`` column. Thresholds can be passed via ``Tier2Thresholds`` or
    individual keyword arguments (which override the dataclass defaults).
    """
    cfg = thresholds or Tier2Thresholds()
    if query_rate_threshold is not None:
        cfg = Tier2Thresholds(
            query_rate=query_rate_threshold,
            input_entropy=input_entropy_threshold or cfg.input_entropy,
            low_conf_rate=low_conf_rate_threshold or cfg.low_conf_rate,
        )
    elif input_entropy_threshold is not None or low_conf_rate_threshold is not None:
        cfg = Tier2Thresholds(
            query_rate=cfg.query_rate,
            input_entropy=input_entropy_threshold or cfg.input_entropy,
            low_conf_rate=low_conf_rate_threshold or cfg.low_conf_rate,
        )

    features = compute_account_features(logs, low_conf_threshold=low_conf_feature_threshold)

    features["flagged"] = (
        (features["query_rate"] > cfg.query_rate)
        | (features["input_entropy"] < cfg.input_entropy)
        | (features["low_conf_rate"] > cfg.low_conf_rate)
    )

    features["flag_reasons"] = features.apply(
        lambda row: _reasons(row, cfg),
        axis=1,
    )

    return features


def flagged_account_ids(results: pd.DataFrame) -> list:
    """Return account IDs marked as suspicious."""
    return results.loc[results["flagged"], "account_id"].tolist()


def _reasons(row: pd.Series, cfg: Tier2Thresholds) -> list[str]:
    reasons: list[str] = []
    if row["query_rate"] > cfg.query_rate:
        reasons.append("high_query_rate")
    if row["input_entropy"] < cfg.input_entropy:
        reasons.append("low_input_entropy")
    if row["low_conf_rate"] > cfg.low_conf_rate:
        reasons.append("high_low_conf_rate")
    return reasons


def _normal_user_rows(
    rng: np.random.Generator,
    base_time: pd.Timestamp,
    dim: int,
    n_normal_accounts: int = 5,
    queries_per_normal: int = 20,
    *,
    repetitive: bool = False,
    cluster_noise: float = 0.10,
    start_offset_minutes: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    for i in range(n_normal_accounts):
        account_id = f"user_{i:03d}"
        anchor = rng.normal(0, 0.02, size=dim) + i * 0.5
        user_start = start_offset_minutes + (i * 2 if repetitive else 0)
        for q in range(queries_per_normal):
            if repetitive:
                # Mostly revisit a tight cluster; occasional wider probe lifts entropy
                # enough to stay above tier-2 thresholds while cell reuse stays high.
                if rng.random() < 0.92:
                    query_input = (anchor + rng.normal(0, 0.02, size=dim)).tolist()
                else:
                    query_input = (anchor + rng.normal(0, 0.06, size=dim)).tolist()
            rows.append(
                {
                    "timestamp": base_time
                    + pd.Timedelta(
                        minutes=user_start + q * 3 + rng.integers(0, 2)
                    ),
                    "account_id": account_id,
                    "query_input": query_input,
                    "confidence_score": float(rng.uniform(0.75, 0.98)),
                }
            )
    return rows


def _single_attacker_rows(
    rng: np.random.Generator,
    base_time: pd.Timestamp,
    dim: int,
    attacker_queries: int = 120,
) -> list[dict]:
    rows: list[dict] = []
    anchor = rng.normal(0, 0.02, size=dim)
    for q in range(attacker_queries):
        rows.append(
            {
                "timestamp": base_time + pd.Timedelta(seconds=q * 2),
                "account_id": "attacker_007",
                "query_input": (anchor + rng.normal(0, 0.01, size=dim)).tolist(),
                "confidence_score": float(rng.uniform(0.25, 0.55)),
            }
        )
    return rows


def _distributed_attack_rows(
    rng: np.random.Generator,
    base_time: pd.Timestamp,
    dim: int,
    n_attackers: int = 25,
) -> list[dict]:
    """
    Many low-volume attacker keys with similar embeddings and mostly low
    confidence, spread over time so no single account crosses tier-2 limits.
    """
    rows: list[dict] = []
    for i in range(n_attackers):
        account_id = f"attacker_dist_{i:03d}"
        n_queries = int(rng.integers(4, 7))
        start_offset = int(rng.integers(0, 90))
        anchor = rng.normal(0, 0.02, size=dim)
        low_conf_slots = {int(rng.integers(0, n_queries))}

        for q in range(n_queries):
            is_low_conf = q in low_conf_slots
            rows.append(
                {
                    "timestamp": base_time
                    + pd.Timedelta(minutes=start_offset + q * int(rng.integers(10, 16))),
                    "account_id": account_id,
                    "query_input": (anchor + rng.normal(0, 0.12, size=dim)).tolist(),
                    "confidence_score": float(
                        rng.uniform(0.25, 0.55) if is_low_conf else rng.uniform(0.72, 0.95)
                    ),
                }
            )
    return rows


def _make_fake_logs(
    rng: np.random.Generator,
    n_normal_accounts: int = 5,
    queries_per_normal: int = 20,
    attacker_queries: int = 120,
) -> pd.DataFrame:
    """Build synthetic logs with one obvious attacker account."""
    base_time = pd.Timestamp("2026-09-08 12:00:00")
    dim = 8
    rows = _normal_user_rows(rng, base_time, dim, n_normal_accounts, queries_per_normal)
    rows.extend(_single_attacker_rows(rng, base_time, dim, attacker_queries))
    return pd.DataFrame(rows)


def _make_distributed_attack_logs(rng: np.random.Generator) -> pd.DataFrame:
    """Build synthetic logs with 25 low-volume coordinated attacker accounts."""
    base_time = pd.Timestamp("2026-09-08 12:00:00")
    dim = 8
    rows = _normal_user_rows(
        rng,
        base_time,
        dim,
        repetitive=True,
        cluster_noise=0.10,
        start_offset_minutes=150,
    )
    rows.extend(_distributed_attack_rows(rng, base_time, dim))
    return pd.DataFrame(rows)


def _run_scenario(label: str, logs: pd.DataFrame) -> None:
    results = detect_accounts(
        logs,
        query_rate_threshold=25.0,
        input_entropy_threshold=0.08,
        low_conf_rate_threshold=0.35,
    )
    flagged = flagged_account_ids(results)
    print(label)
    print("=" * len(label))
    print(results.to_string(index=False))
    print()
    print(f"Flagged accounts: {flagged}")
    print()


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("Scenario 1: Single-account attack")
    print("=" * 40)
    _run_scenario("Tier-2 per-client detection results", _make_fake_logs(rng))

    rng = np.random.default_rng(42)
    print("Scenario 2: Distributed attack")
    print("=" * 40)
    _run_scenario(
        "Tier-2 per-client detection results (25 low-volume attacker keys)",
        _make_distributed_attack_logs(rng),
    )
