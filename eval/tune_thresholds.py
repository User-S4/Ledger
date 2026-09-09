"""Hyperparameter tuning and grid search for Tier-2 detector thresholds.

Explores combinations of query_rate_fraction, input_entropy, and low_conf_rate
thresholds on calibration logs ('cal_*') and ranks them by F1 score.

CRITICAL FIREWALL RULE:
Tuning MUST be performed strictly on calibration data ('cal_*'), NEVER on
evaluation data ('eval_*'). The ground-truth owner column is accessed only
via eval/metrics.py for scoring candidate configurations.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DETECTOR_DIR = ROOT / "detector"
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))

from eval.metrics import evaluate_detections, load_keys_ground_truth
from detector.adapter import load_calibration_logs
from detector.features import compute_account_features
from detector.tier2_perclient import Tier2Thresholds, detect_accounts, load_tier_limits


DEFAULT_QUERY_RATE_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
DEFAULT_INPUT_ENTROPIES = [0.03, 0.05, 0.08, 0.12]
DEFAULT_LOW_CONF_RATES = [0.25, 0.35, 0.45, 0.55]


def grid_search_thresholds(
    logs_df: pd.DataFrame,
    keys_df: pd.DataFrame,
    *,
    query_rate_fractions: Sequence[float] | None = None,
    query_rates: Sequence[float] | None = None,
    input_entropies: Sequence[float] | None = None,
    low_conf_rates: Sequence[float] | None = None,
    low_conf_feature_threshold: float = 0.6,
    tier_limits: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run grid search over Tier-2 thresholds and rank combinations by F1 score.

    Args:
        logs_df: Calibration logs in detector format (from load_calibration_logs).
        keys_df: Ground truth keys table from load_keys_ground_truth.
        query_rate_fractions: List of query_rate_fraction thresholds to try.
        query_rates: Alias for query_rate_fractions.
        input_entropies: List of input_entropy thresholds to try.
        low_conf_rates: List of low_conf_rate thresholds to try.
        low_conf_feature_threshold: Threshold below which a confidence score
            is considered 'low' when computing per-account features.
        tier_limits: Optional dictionary of tier rate limits (req/min).

    Returns:
        pd.DataFrame of all combinations sorted by F1 descending.
    """
    qr_grid = (
        query_rate_fractions
        if query_rate_fractions is not None
        else (query_rates if query_rates is not None else DEFAULT_QUERY_RATE_FRACTIONS)
    )
    ie_grid = input_entropies if input_entropies is not None else DEFAULT_INPUT_ENTROPIES
    lc_grid = low_conf_rates if low_conf_rates is not None else DEFAULT_LOW_CONF_RATES

    if logs_df.empty:
        return pd.DataFrame(
            columns=[
                "query_rate_fraction",
                "input_entropy",
                "low_conf_rate",
                "detection_rate",
                "false_positive_rate",
                "precision",
                "F1",
                "f1_score",
                "tp_count",
                "fp_count",
                "tn_count",
                "fn_count",
            ]
        )

    # Compute account features once to avoid redundant feature extraction
    features = compute_account_features(
        logs_df,
        low_conf_threshold=low_conf_feature_threshold,
    )

    limits = tier_limits or load_tier_limits()
    free_limit = limits.get("free", 60.0)

    if "tier" in features.columns:
        tier_col = features["tier"].fillna("free").astype(str)
        account_limits = tier_col.map(lambda t: limits.get(t, free_limit)).astype(float)
    else:
        account_limits = pd.Series(free_limit, index=features.index)

    rows: list[dict] = []

    for qr, ie, lc in itertools.product(qr_grid, ie_grid, lc_grid):
        cfg = Tier2Thresholds(
            query_rate_fraction=float(qr),
            input_entropy=float(ie),
            low_conf_rate=float(lc),
        )

        f_df = features.copy()
        rate_threshold = account_limits * cfg.query_rate_fraction
        has_multiple_queries = f_df["query_rate"].notna() & f_df["input_entropy"].notna()
        joint_rate_entropy = (
            has_multiple_queries
            & (f_df["query_rate"] > rate_threshold)
            & (f_df["input_entropy"] < cfg.input_entropy)
        )
        low_conf = f_df["low_conf_rate"].notna() & (f_df["low_conf_rate"] > cfg.low_conf_rate)
        f_df["flagged"] = low_conf | joint_rate_entropy

        metrics = evaluate_detections(f_df, keys_df)

        rows.append(
            {
                "query_rate_fraction": float(qr),
                "input_entropy": float(ie),
                "low_conf_rate": float(lc),
                "detection_rate": metrics["detection_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
                "precision": metrics["precision"],
                "F1": metrics["f1_score"],
                "f1_score": metrics["f1_score"],
                "tp_count": metrics["tp_count"],
                "fp_count": metrics["fp_count"],
                "tn_count": metrics["tn_count"],
                "fn_count": metrics["fn_count"],
            }
        )

    results_df = pd.DataFrame(rows)
    return results_df.sort_values(
        ["F1", "detection_rate", "false_positive_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def signal_contribution(
    logs_df: pd.DataFrame,
    keys_df: pd.DataFrame,
    *,
    low_conf_feature_threshold: float = 0.6,
    tier_limits: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Test each Tier-2 signal in isolation to measure its individual value.

    For each of the three signals (query_rate, input_entropy, low_conf_rate),
    runs detection with only that signal active — the other two are disabled
    by setting their thresholds to impossible values that can never fire:
      - query_rate_fraction disabled: threshold = inf  (nothing > inf)
      - input_entropy disabled:       threshold = -1   (nothing < -1)
      - low_conf_rate disabled:       threshold = inf  (nothing > inf)

    Also includes an "all_combined" row using the best-grid defaults for
    comparison, and a "none" baseline where all signals are disabled.

    Args:
        logs_df: Calibration logs in detector format.
        keys_df: Ground truth keys table.
        low_conf_feature_threshold: Feature computation threshold for low confidence.
        tier_limits: Optional dictionary of tier rate limits (req/min).

    Returns:
        pd.DataFrame with one row per signal configuration.
    """
    if logs_df.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "query_rate_fraction_thresh",
                "input_entropy_thresh",
                "low_conf_rate_thresh",
                "detection_rate",
                "false_positive_rate",
                "precision",
                "F1",
                "tp_count",
                "fp_count",
            ]
        )

    features = compute_account_features(
        logs_df,
        low_conf_threshold=low_conf_feature_threshold,
    )

    limits = tier_limits or load_tier_limits()
    free_limit = limits.get("free", 60.0)

    if "tier" in features.columns:
        tier_col = features["tier"].fillna("free").astype(str)
        account_limits = tier_col.map(lambda t: limits.get(t, free_limit)).astype(float)
    else:
        account_limits = pd.Series(free_limit, index=features.index)

    # Disabled-value constants
    INF = float("inf")
    DISABLED_ENTROPY = -1.0  # input_entropy is always >= 0

    # Each entry: (label, query_rate_fraction_thresh, input_entropy_thresh, low_conf_rate_thresh)
    configs = [
        ("query_rate only",   DEFAULT_QUERY_RATE_FRACTIONS[3], DISABLED_ENTROPY, INF),
        ("input_entropy only", INF,                            DEFAULT_INPUT_ENTROPIES[1], INF),
        ("low_conf_rate only", INF,                            DISABLED_ENTROPY, DEFAULT_LOW_CONF_RATES[0]),
        ("all_combined",       DEFAULT_QUERY_RATE_FRACTIONS[3], DEFAULT_INPUT_ENTROPIES[1], DEFAULT_LOW_CONF_RATES[0]),
        ("none (baseline)",    INF,                            DISABLED_ENTROPY, INF),
    ]

    rows: list[dict] = []
    for label, qr, ie, lc in configs:
        f_df = features.copy()
        rate_threshold = account_limits * qr
        has_multiple_queries = f_df["query_rate"].notna() & f_df["input_entropy"].notna()
        joint_rate_entropy = (
            has_multiple_queries
            & (f_df["query_rate"] > rate_threshold)
            & (f_df["input_entropy"] < ie)
        )
        low_conf = f_df["low_conf_rate"].notna() & (f_df["low_conf_rate"] > lc)
        f_df["flagged"] = low_conf | joint_rate_entropy

        metrics = evaluate_detections(f_df, keys_df)
        rows.append(
            {
                "signal": label,
                "query_rate_fraction_thresh": qr,
                "input_entropy_thresh": ie,
                "low_conf_rate_thresh": lc,
                "detection_rate": metrics["detection_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
                "precision": metrics["precision"],
                "F1": metrics["f1_score"],
                "tp_count": metrics["tp_count"],
                "fp_count": metrics["fp_count"],
            }
        )

    return pd.DataFrame(rows)


def fpr_by_tier(
    logs_df: pd.DataFrame,
    keys_df: pd.DataFrame,
    *,
    thresholds: Tier2Thresholds | None = None,
) -> pd.DataFrame:
    """Break down false positive rate by account tier (free / pro / enterprise).

    Runs detect_accounts() on the full logs, joins the per-account tier from
    the adapted logs and the ground-truth owner from the keys table, then
    computes FPR separately for each tier.

    Args:
        logs_df: Adapted logs with a 'tier' column.
        keys_df: Ground-truth keys table with 'api_key_id' and 'owner'.
        thresholds: Optional Tier2Thresholds; uses defaults if omitted.

    Returns:
        pd.DataFrame with columns: tier, normal_accounts, false_positives,
        true_negatives, false_positive_rate.
    """
    cfg = thresholds or Tier2Thresholds()

    if logs_df.empty:
        return pd.DataFrame(
            columns=[
                "tier",
                "normal_accounts",
                "false_positives",
                "true_negatives",
                "false_positive_rate",
            ]
        )

    results = detect_accounts(logs_df, thresholds=cfg)
    merged = results.copy()

    if "tier" not in merged.columns:
        if "tier" in logs_df.columns:
            account_tiers = (
                logs_df.groupby("account_id")["tier"]
                .agg(lambda s: s.mode().iloc[0] if len(s) else "free")
                .reset_index()
            )
            merged = merged.merge(account_tiers, on="account_id", how="left")
        else:
            merged["tier"] = "free"
    merged["tier"] = merged["tier"].fillna("free").astype(str)

    # Merge ground truth
    df_keys = keys_df.copy()
    df_keys["api_key_id"] = df_keys["api_key_id"].astype(str)
    merged = merged.merge(
        df_keys[["api_key_id", "owner"]],
        left_on="account_id",
        right_on="api_key_id",
        how="left",
    )

    # Classify: attacker vs normal
    is_attacker = (
        merged["owner"].fillna("").astype(str).str.lower().str.contains("attacker")
    )
    merged["is_normal"] = ~is_attacker

    # Compute FPR per tier
    rows: list[dict] = []
    for tier_name, group in merged.groupby("tier"):
        normals = group[group["is_normal"]]
        n_normal = len(normals)
        n_fp = int(normals["flagged"].sum())
        n_tn = n_normal - n_fp
        fpr = float(n_fp / n_normal) if n_normal > 0 else 0.0
        rows.append(
            {
                "tier": tier_name,
                "normal_accounts": n_normal,
                "false_positives": n_fp,
                "true_negatives": n_tn,
                "false_positive_rate": fpr,
            }
        )

    return pd.DataFrame(rows).sort_values("tier").reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Tune Tier-2 thresholds on calibration logs")
    ap.add_argument("--db", default="data/fake.db", help="Path to database")
    ap.add_argument("--run-id", default="cal_seed1", help="Calibration run_id (must start with cal_)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Generate it with: python -m api.fake_logs --db data/fake.db --n 20000")
        sys.exit(1)

    print("=" * 70)
    print("TIER-2 THRESHOLD GRID SEARCH (CALIBRATION TUNING)")
    print("=" * 70)
    print(f"Loading calibration logs from {db_path} (run_id: {args.run_id})...")

    # Enforce firewall: calibration data ONLY
    logs_df = load_calibration_logs(run_id=args.run_id, db_path=db_path)
    print(f"Loaded {len(logs_df)} requests across {logs_df['account_id'].nunique()} accounts.")

    print("Loading ground truth keys table...")
    keys_df = load_keys_ground_truth(db_path=db_path)
    print(f"Loaded {len(keys_df)} keys for scoring.")

    print("\nRunning grid search across threshold combinations...")
    grid_results = grid_search_thresholds(logs_df, keys_df)
    print(f"Evaluated {len(grid_results)} threshold combinations.\n")

    print("Top 10 Threshold Combinations (Ranked by F1 Descending):")
    print("-" * 70)
    display_cols = [
        "query_rate_fraction",
        "input_entropy",
        "low_conf_rate",
        "detection_rate",
        "false_positive_rate",
        "precision",
        "F1",
    ]
    formatted = grid_results[display_cols].head(10).copy()
    formatted["detection_rate"] = (formatted["detection_rate"] * 100).round(2).astype(str) + "%"
    formatted["false_positive_rate"] = (formatted["false_positive_rate"] * 100).round(2).astype(str) + "%"
    formatted["precision"] = (formatted["precision"] * 100).round(2).astype(str) + "%"
    formatted["F1"] = formatted["F1"].round(4)
    print(formatted.to_string(index=False))

    print("\n" + "=" * 70)
    best = grid_results.iloc[0]
    print("SINGLE BEST THRESHOLD CONFIGURATION:")
    print("=" * 70)
    print(f"  query_rate_fraction : {best['query_rate_fraction']}")
    print(f"  input_entropy       : {best['input_entropy']}")
    print(f"  low_conf_rate       : {best['low_conf_rate']}")
    print("-" * 70)
    print(f"  F1 Score            : {best['F1']:.4f}")
    print(f"  Detection Rate (TPR): {best['detection_rate'] * 100:.2f}%  ({int(best['tp_count'])} attackers caught)")
    print(f"  False Positive Rate : {best['false_positive_rate'] * 100:.2f}%  ({int(best['fp_count'])} false alarms)")
    print(f"  Precision           : {best['precision'] * 100:.2f}%")
    print("=" * 70)

    # Signal contribution analysis
    print("\n\n" + "=" * 70)
    print("SIGNAL CONTRIBUTION ANALYSIS (each signal tested in isolation)")
    print("=" * 70)
    contrib = signal_contribution(logs_df, keys_df)

    fmt = contrib.copy()
    fmt["detection_rate"] = (fmt["detection_rate"] * 100).round(2).astype(str) + "%"
    fmt["false_positive_rate"] = (fmt["false_positive_rate"] * 100).round(2).astype(str) + "%"
    fmt["precision"] = (fmt["precision"] * 100).round(2).astype(str) + "%"
    fmt["F1"] = fmt["F1"].round(4)

    show_cols = [
        "signal",
        "detection_rate",
        "false_positive_rate",
        "precision",
        "F1",
        "tp_count",
        "fp_count",
    ]
    print(fmt[show_cols].to_string(index=False))
    print("=" * 70)

    # FPR by tier analysis
    print("\n\n" + "=" * 70)
    print("FALSE POSITIVE RATE BY TIER")
    print("=" * 70)
    tier_results = fpr_by_tier(logs_df, keys_df)

    tier_fmt = tier_results.copy()
    tier_fmt["false_positive_rate"] = (
        (tier_fmt["false_positive_rate"] * 100).round(2).astype(str) + "%"
    )
    print(tier_fmt.to_string(index=False))
    print("=" * 70)
