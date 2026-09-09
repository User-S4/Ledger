"""Evaluation metrics for extraction attack detection.

Calculates detection rate (recall), false positive rate (FPR), precision,
and confusion breakdowns against ground-truth owner labels.

CRITICAL FIREWALL RULE:
This evaluation module is the ONLY place in the project allowed to read or
reference the `owner` column. Detector code (features.py, tier2_perclient.py,
tier3_ledger.py, attribution.py, adapter.py) must NEVER import or reference it.
This file evaluates detectors from the outside; it does not feed them answers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_keys_ground_truth(db_path: str | Path | None = None) -> pd.DataFrame:
    """Load the ground-truth keys table (api_key_id, owner, tier).

    This helper is strictly for evaluation and benchmark scoring.
    """
    from api.logstore import LogStore

    resolved_path = Path(db_path) if db_path is not None else ROOT / "data" / "ledger.db"
    if not resolved_path.is_absolute():
        resolved_path = ROOT / resolved_path

    store = LogStore(resolved_path)
    keys_df = store.owners_df()
    store.close()
    return keys_df


def evaluate_detections(
    flagged_results_df: pd.DataFrame,
    keys_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate detector performance against ground-truth account labels.

    Args:
        flagged_results_df: Output of detector (e.g. tier2_perclient.detect_accounts).
            Must contain an account identifier column ('account_id' or 'api_key_id')
            and a boolean 'flagged' column.
        keys_df: Ground-truth DataFrame from the keys table containing
            'api_key_id' (or 'account_id') and 'owner'.

    Returns:
        dict containing:
            - detection_rate: float (recall on actual attackers, TP / (TP + FN))
            - false_positive_rate: float (FP / (FP + TN))
            - precision: float (TP / (TP + FP))
            - f1_score: float
            - tp_count: int
            - fp_count: int
            - tn_count: int
            - fn_count: int
            - total_evaluated: int
            - total_attackers: int
            - total_normal: int
            - true_positives: list[str] (account_ids)
            - false_positives: list[str] (account_ids)
            - true_negatives: list[str] (account_ids)
            - false_negatives: list[str] (account_ids)
    """
    if flagged_results_df.empty:
        return {
            "detection_rate": 0.0,
            "false_positive_rate": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "tp_count": 0,
            "fp_count": 0,
            "tn_count": 0,
            "fn_count": 0,
            "total_evaluated": 0,
            "total_attackers": 0,
            "total_normal": 0,
            "true_positives": [],
            "false_positives": [],
            "true_negatives": [],
            "false_negatives": [],
        }

    # Normalize account ID column in flagged results
    df_results = flagged_results_df.copy()
    if "account_id" not in df_results.columns:
        if "api_key_id" in df_results.columns:
            df_results["account_id"] = df_results["api_key_id"]
        else:
            raise KeyError("flagged_results_df must contain 'account_id' or 'api_key_id' column")

    if "flagged" not in df_results.columns:
        raise KeyError("flagged_results_df must contain 'flagged' boolean column")

    df_results["account_id"] = df_results["account_id"].astype(str)
    df_results["flagged"] = df_results["flagged"].astype(bool)

    # Normalize keys DataFrame
    df_keys = keys_df.copy()
    if "api_key_id" not in df_keys.columns:
        if "account_id" in df_keys.columns:
            df_keys["api_key_id"] = df_keys["account_id"]
        else:
            raise KeyError("keys_df must contain 'api_key_id' or 'account_id' column")

    if "owner" not in df_keys.columns:
        raise KeyError("keys_df must contain 'owner' ground-truth column")

    df_keys["api_key_id"] = df_keys["api_key_id"].astype(str)

    # Left join results to ground truth
    merged = df_results.merge(
        df_keys[["api_key_id", "owner"]],
        left_on="account_id",
        right_on="api_key_id",
        how="left",
    )

    # Ground truth: owner containing "attacker" is positive, all others negative
    # Missing/null owners are treated as normal (negative)
    is_attacker = (
        merged["owner"].fillna("").astype(str).str.lower().str.contains("attacker")
    )
    is_flagged = merged["flagged"]

    tp_mask = is_attacker & is_flagged
    fp_mask = (~is_attacker) & is_flagged
    tn_mask = (~is_attacker) & (~is_flagged)
    fn_mask = is_attacker & (~is_flagged)

    true_positives = merged.loc[tp_mask, "account_id"].tolist()
    false_positives = merged.loc[fp_mask, "account_id"].tolist()
    true_negatives = merged.loc[tn_mask, "account_id"].tolist()
    false_negatives = merged.loc[fn_mask, "account_id"].tolist()

    tp = len(true_positives)
    fp = len(false_positives)
    tn = len(true_negatives)
    fn = len(false_negatives)

    total_attackers = tp + fn
    total_normal = fp + tn
    total_evaluated = len(merged)

    # Metric computations with safe zero-division handling
    detection_rate = float(tp / total_attackers) if total_attackers > 0 else 0.0
    false_positive_rate = float(fp / total_normal) if total_normal > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    if (precision + detection_rate) > 0:
        f1_score = float(2 * (precision * detection_rate) / (precision + detection_rate))
    else:
        f1_score = 0.0

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "f1_score": f1_score,
        "tp_count": tp,
        "fp_count": fp,
        "tn_count": tn,
        "fn_count": fn,
        "total_evaluated": total_evaluated,
        "total_attackers": total_attackers,
        "total_normal": total_normal,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate detector on ground-truth logs")
    ap.add_argument("--db", default="data/fake.db", help="Path to database")
    ap.add_argument("--run-id", default=None, help="Optional run_id filter")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    if not db_path.exists():
        print(f"Database file not found: {db_path}")
        print("Run: python -m api.fake_logs --db data/fake.db --n 20000")
        sys.exit(1)

    print(f"Loading logs from {db_path}...")
    from detector.adapter import load_real_logs
    from detector.tier2_perclient import detect_accounts

    logs_df = load_real_logs(run_id=args.run_id, db_path=db_path)
    print(f"Loaded {len(logs_df)} requests across {logs_df['account_id'].nunique()} unique accounts.")

    print("\nRunning tier2_perclient.detect_accounts()...")
    results_df = detect_accounts(logs_df)
    flagged_count = int(results_df["flagged"].sum())
    print(f"Detector processed {len(results_df)} accounts and flagged {flagged_count} accounts.")

    print("\nLoading ground-truth keys table...")
    keys_df = load_keys_ground_truth(db_path=db_path)
    print(f"Loaded {len(keys_df)} total provisioned keys.")

    print("\nComputing detection evaluation metrics...")
    metrics = evaluate_detections(results_df, keys_df)

    print("\n" + "=" * 60)
    print("DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Detection Rate (Recall) : {metrics['detection_rate'] * 100:.2f}%  ({metrics['tp_count']}/{metrics['total_attackers']} attackers caught)")
    print(f"  False Positive Rate     : {metrics['false_positive_rate'] * 100:.2f}%  ({metrics['fp_count']}/{metrics['total_normal']} legitimate accounts flagged)")
    print(f"  Precision               : {metrics['precision'] * 100:.2f}%  ({metrics['tp_count']}/{metrics['tp_count'] + metrics['fp_count']} flagged accounts are attackers)")
    print(f"  F1 Score                : {metrics['f1_score']:.4f}")
    print("-" * 60)
    print("  Confusion Breakdown:")
    print(f"    True Positives  (TP)  : {metrics['tp_count']:4d}  (attackers correctly flagged)")
    print(f"    False Positives (FP)  : {metrics['fp_count']:4d}  (normal accounts wrongly flagged)")
    print(f"    True Negatives  (TN)  : {metrics['tn_count']:4d}  (normal accounts unflagged)")
    print(f"    False Negatives (FN)  : {metrics['fn_count']:4d}  (attackers missed)")
    print(f"    Total Evaluated       : {metrics['total_evaluated']:4d}")
    print("=" * 60)

