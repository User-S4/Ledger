"""Unit and regression tests for time-based evaluation criteria (P3).

Tests:
1. LogStore.read_timing_df and timing_stats calculation.
2. Inter-query interval (delta_t) computation across accounts.
3. Traditional Tier 1 rate-bypass behavior under pacing.
4. Tier 3 spatial efficiency temporal invariance.
5. Attack session pacing mechanism.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.logstore import LogStore
from eval.time_criteria import (
    evaluate_rate_bypass_frontier,
    evaluate_temporal_invariance_tier3,
    evaluate_duration_to_succeed,
)


def test_logstore_read_timing_empty(tmp_path):
    """Empty database returns empty DataFrame with correct column types."""
    db_path = tmp_path / "empty.db"
    store = LogStore(db_path)
    tdf = store.read_timing_df()
    stats = store.timing_stats()
    store.close()

    assert tdf.empty
    assert "delta_t" in tdf.columns
    assert "global_delta_t" in tdf.columns
    assert stats["total_requests"] == 0
    assert stats["duration_s"] == 0.0


def test_logstore_timing_computation(tmp_path):
    """Inter-query intervals (delta_t) are accurately derived per-account and globally."""
    db_path = tmp_path / "test_timing.db"
    store = LogStore(db_path, batch_size=1)

    # Key A queries at t = 10.0, 11.5 (delta_t = 1.5)
    # Key B queries at t = 10.5, 13.0 (delta_t = 2.5)
    records = [
        {"run_id": "eval_test", "ts": 10.0, "api_key_id": "key_A", "tier": "free", "ip": "1.1.1.1", "status_code": 200},
        {"run_id": "eval_test", "ts": 10.5, "api_key_id": "key_B", "tier": "free", "ip": "1.1.1.2", "status_code": 200},
        {"run_id": "eval_test", "ts": 11.5, "api_key_id": "key_A", "tier": "free", "ip": "1.1.1.1", "status_code": 200},
        {"run_id": "eval_test", "ts": 13.0, "api_key_id": "key_B", "tier": "free", "ip": "1.1.1.2", "status_code": 200},
    ]

    for r in records:
        store.log(**r)
    store.flush()

    tdf = store.read_timing_df(run_id="eval_test")
    stats = store.timing_stats(run_id="eval_test")
    store.close()

    assert len(tdf) == 4
    # Check per-account delta_t
    a_deltas = tdf[tdf["api_key_id"] == "key_A"]["delta_t"].tolist()
    assert np.isnan(a_deltas[0])
    assert abs(a_deltas[1] - 1.5) < 1e-4

    b_deltas = tdf[tdf["api_key_id"] == "key_B"]["delta_t"].tolist()
    assert np.isnan(b_deltas[0])
    assert abs(b_deltas[1] - 2.5) < 1e-4

    # Check global delta_t (ordered: 10.0, 10.5, 11.5, 13.0 -> diffs: nan, 0.5, 1.0, 1.5)
    g_deltas = tdf["global_delta_t"].tolist()
    assert np.isnan(g_deltas[0])
    assert abs(g_deltas[1] - 0.5) < 1e-4
    assert abs(g_deltas[2] - 1.0) < 1e-4
    assert abs(g_deltas[3] - 1.5) < 1e-4

    # Check summary stats
    assert stats["total_requests"] == 4
    assert abs(stats["duration_s"] - 3.0) < 1e-4
    assert abs(stats["effective_qps"] - (4.0 / 3.0)) < 1e-4


def test_rate_bypass_frontier():
    """Verify that pacing API calls beyond free tier limits evades Tier 1."""
    results = evaluate_rate_bypass_frontier(delays=[0.5, 1.0, 1.5, 2.0], rate_threshold=0.8)
    res_dict = {r["delta_t_s"]: r for r in results}

    # At 0.5s interval -> 120 req/min -> util = 2.0 (>= 0.8) -> flagged (NOT evaded)
    assert not res_dict[0.5]["tier1_evaded"]
    assert res_dict[0.5]["rate_utilisation"] > 1.0

    # At 1.5s interval -> 40 req/min -> util = 40/60 = 0.67 (< 0.8) -> EVADED
    assert res_dict[1.5]["tier1_evaded"]

    # At 2.0s interval -> 30 req/min -> util = 30/60 = 0.50 (< 0.8) -> EVADED
    assert res_dict[2.0]["tier1_evaded"]


def test_tier3_temporal_invariance():
    """Verify that Tier 3 efficiency term is completely invariant to query delays."""
    results = evaluate_temporal_invariance_tier3(
        requests_count=50,
        unique_cells_found=40,
        delays=[0.1, 1.0, 10.0, 60.0],
    )

    efficiency_terms = [r["efficiency_term"] for r in results]
    # All delays must produce the exact same efficiency score
    assert len(set(efficiency_terms)) == 1
    # 50 requests / 40 cells = 1.25 spent per cell -> efficiency term = (5 - 1.25)/4 = 0.9375
    assert abs(efficiency_terms[0] - 0.9375) < 1e-4


def test_duration_to_succeed_structure():
    """Check that duration-to-succeed produces coherent metrics."""
    dur = evaluate_duration_to_succeed(budget=10000, qps_raw=10.0, safe_delta_t=1.5, n_keys_distributed=100)
    assert dur["budget_queries"] == 10000
    assert dur["single_key_unthrottled"]["tier1_caught"] is True
    assert dur["single_key_paced"]["tier1_caught"] is False
    assert dur["single_key_paced"]["tier3_caught"] is True
    assert dur["distributed_sybil"]["tier1_caught"] is False
    assert dur["distributed_sybil"]["tier3_caught"] is True


def test_attack_session_pacing():
    """Verify that run_session populates timing_out and records pacing elapsed time."""
    from attack.session import run_session
    from attack.service import LocalService

    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    service = LocalService()
    timing_info = {}

    answers, keys = run_session(
        images,
        key_for=lambda i: f"key_{i}",
        service=service,
        timing_out=timing_info,
        verbose=False,
    )

    assert len(answers) == 3
    assert len(keys) == 3
    assert "elapsed_s" in timing_info
    assert timing_info["total_queries"] == 3
    assert timing_info["elapsed_s"] >= 0.0

