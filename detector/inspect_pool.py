"""Read-only diagnostic script inspecting 'attacker_pool' accounts and features.

Analyzes:
1. Per-key query counts and traffic distribution for attacker_pool accounts.
2. Computed Tier-2 features (query_rate, input_entropy, low_conf_rate).
3. Fraction of attacker_pool keys that individually trigger Tier-2 per-account
   thresholds vs those requiring Tier-3 global coverage detection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DETECTOR_DIR = Path(__file__).resolve().parent
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))

from api.logstore import LogStore
from detector.adapter import load_real_logs
from detector.features import compute_account_features
from detector.tier2_perclient import Tier2Thresholds, detect_accounts
from eval.metrics import load_keys_ground_truth


def inspect_attacker_pool(db_path: str | Path, run_id: str | None = None) -> None:
    resolved_db = Path(db_path)
    if not resolved_db.is_absolute():
        resolved_db = ROOT / resolved_db

    print("=" * 80)
    print("ATTACKER POOL DIAGNOSTIC INSPECTION")
    print("=" * 80)
    print(f"Database: {resolved_db}")

    if not resolved_db.exists():
        print(f"[ERROR] Database file not found: {resolved_db}")
        return

    # 1. Load keys table and isolate attacker_pool accounts
    keys_df = load_keys_ground_truth(db_path=resolved_db)
    pool_keys_all = keys_df[keys_df["owner"] == "attacker_pool"]["api_key_id"].tolist()
    pool_keys_set = set(pool_keys_all)

    print(f"Total 'attacker_pool' keys provisioned in keys table: {len(pool_keys_all)}")

    # 2. Load logs
    logs_df = load_real_logs(run_id=run_id, db_path=resolved_db)
    if logs_df.empty:
        print("[NOTICE] No log records found in database.")
        return

    total_requests = len(logs_df)
    unique_accounts = logs_df["account_id"].nunique()
    print(f"Total requests in log: {total_requests} across {unique_accounts} unique accounts")

    # 3. Compute query counts and per-account features
    counts = logs_df.groupby("account_id").size().rename("query_count")
    features_df = compute_account_features(logs_df)
    features_df = features_df.merge(counts, on="account_id", how="left")

    # Run Tier-2 detection
    t2_results = detect_accounts(logs_df)
    features_df["flagged"] = t2_results["flagged"]
    features_df["flag_reasons"] = t2_results["flag_reasons"]

    # Filter to attacker_pool
    pool_df = features_df[features_df["account_id"].isin(pool_keys_set)].copy()
    pool_df = pool_df.sort_values("account_id").reset_index(drop=True)

    active_pool_count = len(pool_df)
    print(f"Active 'attacker_pool' keys in this log run: {active_pool_count} / {len(pool_keys_all)}")

    if pool_df.empty:
        print("[NOTICE] No queries from attacker_pool found in the log.")
        return

    # 4. Print per-key metrics
    print("\n" + "-" * 80)
    print("INDIVIDUAL ATTACKER_POOL KEYS (First 25 shown):")
    print("-" * 80)
    display_cols = [
        "account_id",
        "tier",
        "query_count",
        "query_rate",
        "input_entropy",
        "low_conf_rate",
        "flagged",
        "flag_reasons",
    ]
    formatted = pool_df[display_cols].head(25).copy()
    formatted["query_rate"] = formatted["query_rate"].round(2)
    formatted["input_entropy"] = formatted["input_entropy"].round(4)
    formatted["low_conf_rate"] = (formatted["low_conf_rate"] * 100).round(1).astype(str) + "%"
    print(formatted.to_string(index=False))

    if len(pool_df) > 25:
        print(f"... and {len(pool_df) - 25} more attacker_pool keys.")

    # 5. Diagnostic Summary & Metrics
    print("\n" + "=" * 80)
    print("ATTACKER POOL TRAFFIC CHARACTERISTICS")
    print("=" * 80)
    total_pool_queries = int(pool_df["query_count"].sum())
    pool_query_share = (total_pool_queries / total_requests) * 100

    print(f"  Total queries sent by pool: {total_pool_queries} ({pool_query_share:.1f}% of total API volume)")
    print(f"  Queries per key (min/median/max): {pool_df['query_count'].min()} / {pool_df['query_count'].median():.1f} / {pool_df['query_count'].max()}")
    print(f"  Single-query keys count: {(pool_df['query_count'] == 1).sum()} / {active_pool_count} ({(pool_df['query_count'] == 1).mean() * 100:.1f}%)")

    # 6. Tier-2 Isolation Detection Analysis
    flagged_pool = pool_df[pool_df["flagged"]]
    flagged_count = len(flagged_pool)
    flagged_fraction = flagged_count / active_pool_count

    print("\n" + "-" * 80)
    print("TIER-2 DETECTION IN ISOLATION")
    print("-" * 80)
    print(f"  Individually flagged keys : {flagged_count} / {active_pool_count} ({flagged_fraction * 100:.2f}%)")
    print(f"  Unflagged (stealthy) keys : {active_pool_count - flagged_count} / {active_pool_count} ({(1 - flagged_fraction) * 100:.2f}%)")

    # Breakdown of trigger reasons
    reasons_series = pool_df["flag_reasons"].explode()
    reasons_counts = reasons_series[reasons_series.notna()].value_counts()
    print("\n  Trigger Reasons Breakdown (keys may match multiple):")
    for reason, count in reasons_counts.items():
        print(f"    - {reason:25s}: {count:3d} keys ({count / active_pool_count * 100:.1f}%)")

    # Single-query artifact note
    single_q_flagged = ((pool_df["query_count"] == 1) & pool_df["flagged"]).sum()
    multi_q_flagged = ((pool_df["query_count"] > 1) & pool_df["flagged"]).sum()
    print(f"\n  Flagged single-query keys : {single_q_flagged} (flagged due to 1-sec span default rate & 0.0 entropy / low conf)")
    print(f"  Flagged multi-query keys  : {multi_q_flagged}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC CONCLUSION:")
    print("=" * 80)
    if pool_df["query_count"].max() <= 10:
        print("  - Volume Profile: GENUINELY DISTRIBUTED / LOW VOLUME PER KEY")
        print(f"    Each key sent at most {pool_df['query_count'].max()} queries across the entire run.")
    else:
        print("  - Volume Profile: MODERATE TO HIGH VOLUME PER KEY")

    if flagged_fraction < 0.5:
        print("  - Tier-2 Capability: Most keys evade per-client limits. Tier-3 Global Ledger is ESSENTIAL.")
    else:
        print(f"  - Tier-2 Capability: {flagged_fraction * 100:.1f}% flagged in isolation (primarily from low_conf_rate or single-query rate defaults).")
        print(f"  - Remaining { (1 - flagged_fraction) * 100:.1f}% stealth keys bypass Tier-2 and require Tier-3 coverage ledger.")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Inspect attacker_pool accounts in logs")
    ap.add_argument("--db", default="data/fake.db", help="Database path (default: data/fake.db)")
    ap.add_argument("--run-id", default=None, help="Optional run_id filter")
    args = ap.parse_args()

    inspect_attacker_pool(args.db, run_id=args.run_id)


if __name__ == "__main__":
    main()

