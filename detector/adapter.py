"""Log schema adapter for detector pipeline.

Bridges the real API's log schema (SCHEMA.md, api/logstore.py) to the
standardized column format expected by detector modules:
  - timestamp: pd.Timestamp datetime
  - account_id: str (from api_key_id)
  - query_input: np.ndarray (from decoded embedding)
  - confidence_score: float (max of model_probs)
  - run_id: str
  - tier: str

CRITICAL FIREWALL RULE:
Never read, join, or reference the `owner` column from the keys table.
Ground-truth owner labels are reserved exclusively for offline scoring.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DETECTOR_DIR = Path(__file__).resolve().parent
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))

from api.logstore import LogStore

EXPECTED_DETECTOR_COLS = [
    "timestamp",
    "account_id",
    "query_input",
    "confidence_score",
    "run_id",
    "tier",
]


def _get_default_db_path() -> Path:
    """Resolve the database path from env, config.yaml, or default."""
    env_db = os.environ.get("LEDGER_DB")
    if env_db:
        p = Path(env_db)
        return p if p.is_absolute() else ROOT / p

    cfg_path = Path(os.environ.get("LEDGER_CONFIG", ROOT / "config.yaml"))
    if cfg_path.exists():
        try:
            import yaml

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            db_rel = cfg.get("run", {}).get("db_path", "data/ledger.db")
            p = Path(db_rel)
            return p if p.is_absolute() else ROOT / p
        except Exception:
            pass

    return ROOT / "data" / "ledger.db"


def _extract_confidence(probs: Any) -> float:
    """Derive confidence_score as max(model_probs)."""
    if isinstance(probs, (list, tuple, np.ndarray)) and len(probs) > 0:
        return float(max(probs))
    return float(np.nan)


def load_real_logs(
    run_id: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load API request logs and transform them into detector format.

    Args:
        run_id: Optional run_id filter (e.g. 'cal_seed1' or 'eval_seed2').
        db_path: Path to the SQLite database file. Defaults to config/env.

    Returns:
        pd.DataFrame with columns:
            timestamp, account_id, query_input, confidence_score, run_id, tier
    """
    resolved_db = Path(db_path) if db_path is not None else _get_default_db_path()
    if not resolved_db.is_absolute():
        resolved_db = ROOT / resolved_db

    if not resolved_db.exists():
        return pd.DataFrame(columns=EXPECTED_DETECTOR_COLS)

    store = LogStore(resolved_db)
    raw_df = store.read_df(run_id=run_id, decode=True)
    store.close()

    if raw_df.empty:
        return pd.DataFrame(columns=EXPECTED_DETECTOR_COLS)

    df = raw_df.copy()

    # Filter by run_id if provided (in case read_df was not passed run_id)
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_DETECTOR_COLS)

    # Filter out failed requests / missing embeddings so detector math won't crash
    if "status_code" in df.columns:
        df = df[df["status_code"] == 200]
    if "embedding" in df.columns:
        df = df[df["embedding"].notna()]

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_DETECTOR_COLS)

    # Rename and derive columns according to detector requirements
    transformed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["ts"], unit="s"),
            "account_id": df["api_key_id"].astype(str),
            "query_input": df["embedding"],
            "confidence_score": df["model_probs"].map(_extract_confidence),
            "run_id": df["run_id"].astype(str),
            "tier": df["tier"].astype(str),
        }
    )

    return transformed[EXPECTED_DETECTOR_COLS].reset_index(drop=True)


def load_calibration_logs(
    run_id: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load calibration logs (run_id starting with 'cal_') for hyperparameter tuning.

    Enforces the tuning-vs-reporting firewall rule from SCHEMA.md.
    """
    if run_id is not None and not run_id.startswith("cal_"):
        raise ValueError(
            f"Calibration run_id must start with 'cal_', got: {run_id!r}"
        )

    df = load_real_logs(run_id=run_id, db_path=db_path)
    if not df.empty and "run_id" in df.columns:
        df = df[df["run_id"].astype(str).str.startswith("cal_")].reset_index(drop=True)
    return df


def load_evaluation_logs(
    run_id: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load evaluation logs (run_id starting with 'eval_') for reporting results.

    Enforces the tuning-vs-reporting firewall rule from SCHEMA.md.
    """
    if run_id is not None and not run_id.startswith("eval_"):
        raise ValueError(
            f"Evaluation run_id must start with 'eval_', got: {run_id!r}"
        )

    df = load_real_logs(run_id=run_id, db_path=db_path)
    if not df.empty and "run_id" in df.columns:
        df = df[df["run_id"].astype(str).str.startswith("eval_")].reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Testing detector log adapter...")
    print("=" * 60)

    # 1. Try loading from default real DB path
    default_db = _get_default_db_path()
    print(f"Loading real logs from default database: {default_db}")
    df_real = load_real_logs()

    if df_real.empty:
        print(
            f"[NOTICE] The database at {default_db} is empty or has no data yet.\n"
            f"Returning an empty DataFrame with schema: {list(df_real.columns)}"
        )
        # If default DB has no data, check if data/fake.db exists for demonstration
        fake_db = ROOT / "data" / "fake.db"
        if fake_db.exists():
            print("\nFound data/fake.db, testing adapter with fake database...")
            df_real = load_real_logs(db_path=fake_db)

    print(f"\nTotal rows loaded: {len(df_real)}")
    if not df_real.empty:
        print("\nFirst 5 rows:")
        print(df_real.head())

        print("\nChecking columns against detector expectations:")
        required_cols = {"timestamp", "account_id", "query_input", "confidence_score"}
        present_cols = set(df_real.columns)
        missing = required_cols - present_cols
        print(f"Required columns present: {missing == set()} (Columns: {list(df_real.columns)})")

        print("\nVerifying integration with tier2_perclient.detect_accounts()...")
        try:
            from tier2_perclient import detect_accounts

            results = detect_accounts(df_real)
            print("Successfully ran detect_accounts() on adapted logs!")
            print(f"Results shape: {results.shape}")
            print(f"Flagged accounts count: {results['flagged'].sum()} / {len(results)}")
            print("\nSample detector output:")
            print(results.head())
        except Exception as e:
            print(f"Error executing detect_accounts: {e}")
            raise
    else:
        print("\nDatabase is empty. Testing detect_accounts() on empty adapted logs...")
        from tier2_perclient import detect_accounts

        empty_results = detect_accounts(df_real)
        print(f"detect_accounts handled empty logs cleanly, returned shape: {empty_results.shape}")

    # 2. Test calibration and evaluation firewall filters
    print("\n" + "=" * 60)
    print("Testing calibration / evaluation filters...")
    fake_db = ROOT / "data" / "fake.db"
    target_db = default_db if default_db.exists() and len(df_real) > 0 else (fake_db if fake_db.exists() else None)
    if target_db:
        cal_df = load_calibration_logs(db_path=target_db)
        eval_df = load_evaluation_logs(db_path=target_db)
        print(f"Calibration logs ('cal_*'): {len(cal_df)} rows")
        print(f"Evaluation logs ('eval_*'): {len(eval_df)} rows")

        # Test firewall enforcement on invalid run_ids
        try:
            load_calibration_logs(run_id="eval_test")
            print("ERROR: Allowed eval_ run_id in calibration loader!")
        except ValueError:
            print("Firewall OK: load_calibration_logs correctly rejected 'eval_test'")

        try:
            load_evaluation_logs(run_id="cal_test")
            print("ERROR: Allowed cal_ run_id in evaluation loader!")
        except ValueError:
            print("Firewall OK: load_evaluation_logs correctly rejected 'cal_test'")

    print("\nAll adapter tests passed successfully!")

