"""new1.py -- Enhanced Tier-2 Per-Account Extraction Detector.

Provides per-account anomaly scoring with explicit activity categorization,
risk levels, and defense action recommendations (ALLOW vs. DEGRADE vs. BLOCK).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DETECTOR_DIR = Path(__file__).resolve().parent
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))

try:
    from detector.features import compute_account_features
except ImportError:
    from features import compute_account_features

DEFAULT_TIERS: dict[str, float] = {"free": 60.0, "pro": 600.0, "enterprise": 6000.0}


def load_tier_limits(config_path: str | Path | None = None) -> dict[str, float]:
    """Load rate limits per tier from config.yaml's tiers section."""
    cfg_path = Path(config_path or os.environ.get("LEDGER_CONFIG", ROOT / "config.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            tiers = cfg.get("tiers")
            if isinstance(tiers, dict):
                return {k: float(v) for k, v in tiers.items()}
        except Exception:
            pass
    return dict(DEFAULT_TIERS)


@dataclass(frozen=True)
class Tier2Thresholds:
    """Configurable thresholds for Tier-2 per-account anomaly detection."""

    query_rate_fraction: float = 0.5
    input_entropy: float = 0.05
    consecutive_distance: float = 0.1
    low_conf_rate: float = 0.35


def _categorize_activity(row: pd.Series) -> str:
    """Assign human-readable activity classification to an account."""
    flagged = bool(row.get("flagged", False))
    qr = float(row.get("query_rate", 0.0)) if pd.notna(row.get("query_rate")) else 0.0
    ie = float(row.get("input_entropy", 0.0)) if pd.notna(row.get("input_entropy")) else 0.0
    lc = float(row.get("low_conf_rate", 0.0)) if pd.notna(row.get("low_conf_rate")) else 0.0

    if flagged:
        if lc > 0.5:
            return "SUSPICIOUS_BOUNDARY_PROBING"
        if qr > 10.0 and ie < 0.05:
            return "SUSPICIOUS_HIGH_SPEED_EXTRACTION"
        return "SUSPICIOUS_ANOMALOUS_CLIENT"
    else:
        if qr > 5.0 and ie > 0.08:
            return "NORMAL_BATCH_PIPELINE"
        if ie < 0.03 and qr < 2.0:
            return "NORMAL_REPETITIVE_ROUTINE"
        if lc > 0.3:
            return "NORMAL_EXPLORATORY_RESEARCH"
        return "NORMAL_BENIGN_USER"


def _determine_action(row: pd.Series) -> str:
    """Determine recommended security action for the account."""
    flagged = bool(row.get("flagged", False))
    if not flagged:
        return "ALLOW"
    reasons = row.get("flag_reasons", [])
    if len(reasons) >= 2:
        return "DEGRADE_LEVEL_2_TOP3_ROUNDED"
    return "DEGRADE_LEVEL_1_ROUNDED"


def _determine_risk_level(row: pd.Series) -> str:
    """Assign risk level based on signals."""
    flagged = bool(row.get("flagged", False))
    reasons = row.get("flag_reasons", [])
    if not flagged:
        return "LOW"
    if len(reasons) >= 2:
        return "HIGH"
    return "MEDIUM"


def detect_accounts(
    logs: pd.DataFrame,
    *,
    thresholds: Tier2Thresholds | None = None,
    query_rate_fraction: float | None = None,
    query_rate_threshold: float | None = None,
    input_entropy_threshold: float | None = None,
    low_conf_rate_threshold: float | None = None,
    low_conf_feature_threshold: float = 0.6,
    tier_limits: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Analyze logs, extract features, and classify accounts with actions."""
    if logs.empty:
        cols = [
            "account_id",
            "query_rate",
            "input_entropy",
            "consecutive_distance",
            "low_conf_rate",
            "tier",
            "flagged",
            "activity_label",
            "action",
            "risk_level",
            "flag_reasons",
        ]
        return pd.DataFrame(columns=cols)

    cfg = thresholds or Tier2Thresholds()
    qr_frac = query_rate_fraction if query_rate_fraction is not None else query_rate_threshold
    if qr_frac is not None:
        cfg = Tier2Thresholds(
            query_rate_fraction=qr_frac,
            input_entropy=input_entropy_threshold if input_entropy_threshold is not None else cfg.input_entropy,
            low_conf_rate=low_conf_rate_threshold if low_conf_rate_threshold is not None else cfg.low_conf_rate,
        )

    features = compute_account_features(logs, low_conf_threshold=low_conf_feature_threshold)

    limits = tier_limits or load_tier_limits()
    free_limit = limits.get("free", 60.0)

    if "tier" in features.columns:
        tier_col = features["tier"].fillna("free").astype(str)
        account_limits = tier_col.map(lambda t: limits.get(t, free_limit)).astype(float)
    else:
        account_limits = pd.Series(free_limit, index=features.index)

    rate_threshold = account_limits * cfg.query_rate_fraction

    has_multiple_queries = features["query_rate"].notna() & features["input_entropy"].notna()
    joint_rate_entropy = (
        has_multiple_queries
        & (features["query_rate"] > rate_threshold)
        & (features["input_entropy"] < cfg.input_entropy)
    )
    low_conf = features["low_conf_rate"].notna() & (features["low_conf_rate"] > cfg.low_conf_rate)

    features["flagged"] = low_conf | joint_rate_entropy
    features["flag_reasons"] = features.apply(lambda row: _reasons(row, cfg, limits), axis=1)
    features["activity_label"] = features.apply(_categorize_activity, axis=1)
    features["action"] = features.apply(_determine_action, axis=1)
    features["risk_level"] = features.apply(_determine_risk_level, axis=1)

    return features


def _reasons(row: pd.Series, cfg: Tier2Thresholds, tier_limits: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    tier_name = str(row.get("tier", "free"))
    rate_thresh = tier_limits.get(tier_name, 60.0) * cfg.query_rate_fraction

    qr = row.get("query_rate")
    if pd.notna(qr) and qr > rate_thresh:
        reasons.append("high_query_rate")

    ie = row.get("input_entropy")
    if pd.notna(ie) and ie < cfg.input_entropy:
        reasons.append("low_input_entropy")

    lc = row.get("low_conf_rate")
    if pd.notna(lc) and lc > cfg.low_conf_rate:
        reasons.append("high_low_conf_rate")

    return reasons


def _main() -> None:
    import argparse
    from detector.adapter import load_real_logs

    ap = argparse.ArgumentParser(description="new1: Tier-2 Per-Client Anomaly Detector with Activity Labeling")
    ap.add_argument("--db", default="data/ledger.db")
    ap.add_argument("--run-id", default="eval_seed2")
    a = ap.parse_args()

    df = load_real_logs(run_id=a.run_id, db_path=a.db)
    print(f"[new1] Loaded {len(df)} requests across {df['account_id'].nunique()} accounts from {a.db} ({a.run_id})")

    results = detect_accounts(df)
    print(f"\n[new1] Total evaluated accounts: {len(results)}")
    print(f"[new1] Activity Label Breakdown:")
    print(results["activity_label"].value_counts().to_string())
    print(f"\n[new1] Recommended Action Breakdown:")
    print(results["action"].value_counts().to_string())

    print("\nSample Output (First 10 Accounts):")
    display_cols = ["account_id", "tier", "activity_label", "action", "risk_level", "flagged"]
    print(results[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    _main()
