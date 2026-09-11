"""new2.py -- Enhanced Tier-3 Global Coverage Ledger with Activity Labeling & Actions.

Computes embedding space spatial discovery metrics, categorizes account activity
(NORMAL_OFFICE, NORMAL_ROUTINE, NORMAL_CASUAL, SUSPICIOUS_EXTRACTION_SWEEP),
and recommends graduated Stage 7 defense actions.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.cells import assign_cell

DEFAULT_DIMS = 8
DEFAULT_CELL_SIZE = 1.0
MIN_REQUESTS_FOR_EFFICIENCY = 30


@dataclass
class Tier3Config:
    """Configurable parameters for Tier-3 spatial coverage detection."""

    cell_size: float = DEFAULT_CELL_SIZE
    dims: int | None = DEFAULT_DIMS
    spike_window: int = 20
    spike_threshold: int = 10
    rate_reference: float = 20.0
    coverage_reference: float = 0.25
    min_requests: int = MIN_REQUESTS_FOR_EFFICIENCY


def score_account(
    key_cells: int,
    total_cells: int,
    discovery_rate: float,
    requests: int,
    cfg: Tier3Config | None = None,
) -> float:
    """Compute 0.0 - 1.0 suspicion score for an account."""
    cfg = cfg or Tier3Config()
    if requests <= 0:
        return 0.0

    rate_term = min(1.0, discovery_rate / cfg.rate_reference)

    if requests < cfg.min_requests or key_cells <= 0:
        efficiency_term = 0.0
    else:
        spent = requests / key_cells
        efficiency_term = max(0.0, min(1.0, (5.0 - spent) / 4.0))

    coverage_term = 0.0
    if total_cells > 0:
        coverage_term = min(1.0, (key_cells / total_cells) / cfg.coverage_reference)

    score = 0.45 * rate_term + 0.30 * efficiency_term + 0.25 * coverage_term
    return round(min(1.0, score), 4)


def score_global(account_scores: dict[str, float]) -> float:
    """Calculate system-wide extraction pressure (worst 10th percentile)."""
    if not account_scores:
        return 0.0
    vals = sorted(account_scores.values(), reverse=True)
    top = vals[: max(1, len(vals) // 10)]
    return round(sum(top) / len(top), 4)


def classify_account(
    score: float,
    requests: int,
    key_cells: int,
    threshold: float = 0.30,
) -> dict[str, str]:
    """Classify account into explicit activity labels, actions, and risk tiers."""
    spent = requests / max(1, key_cells)

    if score >= threshold:
        # High suspicion extraction behavior
        if score >= 0.70:
            activity = "SUSPICIOUS_HIGH_INTENSITY_EXTRACTION"
            action = "DEGRADE_LEVEL_3_LABEL_ONLY"
            risk = "CRITICAL"
        elif score >= 0.50:
            activity = "SUSPICIOUS_SPATIAL_SWEEP"
            action = "DEGRADE_LEVEL_2_TOP3_ROUNDED"
            risk = "HIGH"
        else:
            activity = "SUSPICIOUS_DISTRIBUTED_THEFT"
            action = "DEGRADE_LEVEL_1_ROUNDED_PROBS"
            risk = "MEDIUM_HIGH"
    else:
        # Normal behavior
        if requests >= 200:
            activity = "NORMAL_ENTERPRISE_BATCH"
            action = "ALLOW_FULL_ACCESS"
            risk = "LOW"
        elif requests >= 50 and spent >= 3.0:
            activity = "NORMAL_OFFICE_WORKFLOW"
            action = "ALLOW_FULL_ACCESS"
            risk = "LOW"
        elif requests < 30:
            activity = "NORMAL_CASUAL_QUERY"
            action = "ALLOW_FULL_ACCESS"
            risk = "LOW"
        else:
            activity = "NORMAL_BENIGN_ACTIVITY"
            action = "ALLOW_FULL_ACCESS"
            risk = "LOW"

    return {
        "activity_label": activity,
        "action": action,
        "risk_level": risk,
    }


def analyze_ledger_run(
    db_path: str | Path,
    run_id: str = "eval_seed2",
    threshold: float = 0.30,
    dims: int = DEFAULT_DIMS,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> pd.DataFrame:
    """Analyze a database run and return full labeled results table."""
    from api.logstore import LogStore
    from detector.cell_index import rebuild_from_log

    store = LogStore(db_path)
    idx = rebuild_from_log(
        store,
        run_id,
        cell_fn=lambda e: assign_cell(e, cell_size, dims),
    )
    cfg = Tier3Config(cell_size=cell_size, dims=dims)

    keys = list(idx.accounts())
    scores = {k: score_account(
        key_cells=idx.key_coverage(k),
        total_cells=idx.total_coverage(),
        discovery_rate=idx.discovery_rate(k),
        requests=idx.requests_for(k),
        cfg=cfg,
    ) for k in keys}

    owners = store.owners_df().set_index("api_key_id")["owner"].to_dict()
    store.close()

    rows = []
    for k in keys:
        s = scores[k]
        reqs = idx.requests_for(k)
        cells = idx.key_coverage(k)
        clf = classify_account(s, reqs, cells, threshold=threshold)
        rows.append({
            "account_id": k,
            "owner": owners.get(k, "unknown"),
            "suspicion_score": s,
            "requests": reqs,
            "cells_covered": cells,
            "efficiency_reqs_per_cell": round(reqs / max(1, cells), 2),
            "flagged": s >= threshold,
            "activity_label": clf["activity_label"],
            "action": clf["action"],
            "risk_level": clf["risk_level"],
        })

    return pd.DataFrame(rows)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="new2: Enhanced Tier-3 Global Coverage Ledger with Activity Labeling")
    ap.add_argument("--db", default="data/ledger.db")
    ap.add_argument("--run-id", default="eval_seed2")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    ap.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE)
    ap.add_argument("--out", default=None, help="Optional output JSON/CSV path")
    a = ap.parse_args()

    print(f"[new2] Running Tier-3 spatial coverage analysis on {a.db} ({a.run_id})...")
    df = analyze_ledger_run(a.db, run_id=a.run_id, threshold=a.threshold, dims=a.dims, cell_size=a.cell_size)

    print(f"\nTotal Accounts Analyzed: {len(df)}")
    print("\n[new2] Activity Label Breakdown:")
    print(df["activity_label"].value_counts().to_string())

    print("\n[new2] Recommended Security Actions:")
    print(df["action"].value_counts().to_string())

    print("\n[new2] Persona Group Summary:")
    summary = df.groupby("owner").agg(
        accounts=("suspicion_score", "size"),
        median_score=("suspicion_score", "median"),
        median_reqs=("requests", "median"),
        median_cells=("cells_covered", "median"),
        flagged=("flagged", "sum"),
    )
    summary["caught_pct"] = (summary["flagged"] / summary["accounts"]).map("{:.1%}".format)
    print(summary.to_string())

    print("\nSample Output (First 10 Accounts):")
    cols = ["account_id", "owner", "suspicion_score", "activity_label", "action", "risk_level"]
    print(df[cols].head(10).to_string(index=False))

    if a.out:
        out_p = Path(a.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if str(out_p).endswith(".json"):
            import json
            out_p.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
        else:
            df.to_csv(out_p, index=False)
        print(f"\n[Saved results to {out_p}]")


if __name__ == "__main__":
    _main()
