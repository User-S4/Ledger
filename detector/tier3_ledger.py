"""Tier-3 global coverage ledger for detecting distributed extraction attacks.

P1 owns this file. P3 edited it on Day 2 with P1's agreement. Three changes:

  1. Column names now match SCHEMA.md (ts / api_key_id / embedding). The
     originals were written before the schema was frozen and could not read
     a real log row.

  2. Added score_account() and score_global() -- one number per account plus
     one global number, which is what api/defense.py reads in Stage 7. The
     original produced a single system-wide boolean, which cannot tell you
     WHOSE answers to degrade.

  3. Defaults changed to 8 dimensions at cell_size 1.0, measured on 46k real
     logged embeddings (python -m victim.embed diagnose). The old 0.2 across
     all 32 dimensions put every query in its own cell, so the spike alarm
     was permanently on for everyone.

The batch functions at the bottom remain P1's tool for after-the-fact
analysis. They are NOT what Stage 7 calls -- they read a whole DataFrame,
which cannot happen inside a request.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from detector.cells import assign_cell

# Measured on real logged embeddings, not invented ones. See the module note.
DEFAULT_DIMS = 8
DEFAULT_CELL_SIZE = 1.0

# Below this many requests an account has not had the CHANCE to revisit
# ground, so its efficiency is meaningless. Without this gate, every casual
# user who asks nine questions scores as a ruthlessly efficient prober.
MIN_REQUESTS_FOR_EFFICIENCY = 30


@dataclass
class Tier3Config:
    """Configurable parameters. P1 owns the values; config.yaml holds them."""

    cell_size: float = DEFAULT_CELL_SIZE
    dims: int | None = DEFAULT_DIMS
    spike_window: int = 20
    spike_threshold: int = 10
    rate_reference: float = 20.0        # new cells/min that saturates the term
    coverage_reference: float = 0.25    # share of the map that saturates it
    min_requests: int = MIN_REQUESTS_FOR_EFFICIENCY


# ---------------------------------------------------------------- scoring

def score_account(key_cells: int, total_cells: int, discovery_rate: float,
                  requests: int, cfg: Tier3Config | None = None) -> float:
    """One suspicion number for one account, 0 (ordinary) to 1 (sweeping).

    This is the Stage 7 contract. Facts in, judgement out -- no database, no
    DataFrame, no I/O -- so it runs inside a request in microseconds. Every
    input comes straight out of CellIndex.

    Three ingredients, and why none of them works alone:

    discovery_rate  new cells per minute. Moves first when a sweep starts,
                    and unlike raw request rate it does not punish an
                    enterprise customer for asking lots of repetitive
                    questions. Per MINUTE, not per N queries: ten new cells
                    in twenty queries means something completely different
                    at 5 req/min than at 500.

    efficiency      requests spent per new cell. Low means almost nothing is
                    repeated. Alone it convicts researchers, who probe edge
                    cases for a living, so it is gated on volume.

    coverage share  how much of the known map this account alone has seen.
                    Catches the loud single-key thief, who is WASTEFUL and
                    so scores low on efficiency. The two signals cover each
                    other's blind spot.

    Deliberately excluded: how many accounts share an IP. That is tier 1's
    job, and alone it flags every office, university and mobile network.
    """
    cfg = cfg or Tier3Config()
    if requests <= 0:
        return 0.0

    rate_term = min(1.0, discovery_rate / cfg.rate_reference)

    if requests < cfg.min_requests or key_cells <= 0:
        efficiency_term = 0.0
    else:
        spent = requests / key_cells
        # 1 request per new cell -> 1.0.  5 or more -> 0.
        efficiency_term = max(0.0, min(1.0, (5.0 - spent) / 4.0))

    coverage_term = 0.0
    if total_cells > 0:
        coverage_term = min(1.0, (key_cells / total_cells) / cfg.coverage_reference)

    score = 0.45 * rate_term + 0.30 * efficiency_term + 0.25 * coverage_term
    return round(min(1.0, score), 4)


def score_global(account_scores: dict) -> float:
    """One number for the whole system, as agreed for the Stage 7 handoff.

    Mean of the worst tenth, not the mean of everyone: a handful of hostile
    accounts among thousands of honest ones should move this, and an overall
    average would drown them.
    """
    if not account_scores:
        return 0.0
    vals = sorted(account_scores.values(), reverse=True)
    top = vals[: max(1, len(vals) // 10)]
    return round(sum(top) / len(top), 4)


def score_from_index(index, api_key_id: str, now=None,
                     cfg: Tier3Config | None = None) -> float:
    """Pull the four facts out of a live CellIndex and score them.

    This is the single call api/defense.py makes per request.
    """
    return score_account(
        key_cells=index.key_coverage(api_key_id),
        total_cells=index.total_coverage(),
        discovery_rate=index.discovery_rate(api_key_id, now=now),
        requests=index.requests_for(api_key_id),
        cfg=cfg,
    )


# ---------------------------------------------------------------- ledger

@dataclass
class GlobalCoverageLedger:
    """
    Running ledger of unique embedding cells seen across all accounts.

    Kept for P1's batch analysis. The live path used by the API is
    detector/cell_index.py -- one tally, ~3 microseconds per request. Two
    tallies that can disagree would be worse than either.
    """

    cell_size: float = DEFAULT_CELL_SIZE
    dims: int | None = DEFAULT_DIMS
    spike_window: int = 20
    _cells: set = field(default_factory=set, init=False, repr=False)
    _cell_accounts: dict = field(default_factory=lambda: defaultdict(set),
                                 init=False, repr=False)
    _recent_new: deque = field(default_factory=deque, init=False, repr=False)
    _recent_meta: deque = field(default_factory=deque, init=False, repr=False)
    _query_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._recent_new = deque(maxlen=self.spike_window)
        self._recent_meta = deque(maxlen=self.spike_window)

    def update(self, account_id: str, embedding) -> tuple:
        """Record a query embedding and return its assigned cell."""
        cell = assign_cell(embedding, self.cell_size, self.dims)
        is_new = cell not in self._cells
        self._cells.add(cell)
        self._cell_accounts[cell].add(account_id)
        self._recent_new.append(is_new)
        self._recent_meta.append(
            {"account_id": account_id, "cell": cell, "is_new_cell": is_new})
        self._query_count += 1
        return cell

    def coverage(self) -> int:
        """Total number of unique cells covered so far."""
        return len(self._cells)

    def spike_rate(self, window_size: int | None = None) -> int:
        """Count of new unique cells added in the last ``window_size`` queries."""
        window = window_size or self.spike_window
        return int(sum(list(self._recent_new)[-window:]))

    def flag_if_spiking(self, threshold) -> bool:
        """Return True when recent new-cell count exceeds ``threshold``."""
        if len(self._recent_new) < self.spike_window:
            return False
        return self.spike_rate() > threshold

    def accounts_for_cell(self, cell) -> set:
        """Return account IDs that have touched a cell."""
        return set(self._cell_accounts.get(cell, set()))


def detect_distributed_attack(
    logs_df: pd.DataFrame,
    *,
    cell_size: float = DEFAULT_CELL_SIZE,
    dims: int | None = DEFAULT_DIMS,
    spike_window: int = 20,
    spike_threshold: int = 10,
    time_col: str = "ts",             # SCHEMA.md column 3
    account_col: str = "api_key_id",  # SCHEMA.md column 4
    embedding_col: str = "embedding", # SCHEMA.md column 12
) -> pd.DataFrame:
    """
    Process logs in timestamp order and flag windows with fast coverage growth.

    Returns a DataFrame of triggered windows with window boundaries, spike
    rate, and cumulative coverage at detection time.

    A caution on `accounts_in_window`: it lists everyone who appeared in the
    window, not everyone responsible. Under interleaved traffic an office
    worker querying during an attacker's burst lands in that list. Treat it
    as a shortlist to investigate, never as an accusation -- for real
    attribution use CellIndex.cohort(), which asks who shared CELLS rather
    than who shared a moment in time.
    """
    ledger = GlobalCoverageLedger(cell_size=cell_size, dims=dims,
                                  spike_window=spike_window)
    df = logs_df
    if "status_code" in df.columns:
        df = df[df.status_code == 200]
    df = df[df[embedding_col].notna()]
    ordered = df.sort_values(time_col).reset_index(drop=True)

    flagged, in_spike = [], False
    for idx, row in ordered.iterrows():
        ledger.update(row[account_col], row[embedding_col])
        spiking = ledger.flag_if_spiking(spike_threshold)
        if spiking and not in_spike:
            start = max(0, idx - spike_window + 1)
            flagged.append({
                "window_start": ordered.loc[start, time_col],
                "window_end": row[time_col],
                "query_index_start": start,
                "query_index_end": idx,
                "spike_rate": ledger.spike_rate(),
                "coverage": ledger.coverage(),
                "accounts_in_window": sorted(
                    {e["account_id"] for e in list(ledger._recent_meta)}),
            })
        in_spike = spiking

    cols = ["window_start", "window_end", "query_index_start",
            "query_index_end", "spike_rate", "coverage", "accounts_in_window"]
    return pd.DataFrame(flagged) if flagged else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------- CLI

def _main() -> None:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.logstore import LogStore
    from detector.cell_index import rebuild_from_log

    ap = argparse.ArgumentParser(description="Tier-3 scores from a logged run")
    ap.add_argument("--db", default="data/fake.db")
    ap.add_argument("--run-id", default="cal_seed1")
    ap.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    ap.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    store = LogStore(a.db)
    idx = rebuild_from_log(store, a.run_id,
                           cell_fn=lambda e: assign_cell(e, a.cell_size, a.dims))
    cfg = Tier3Config(cell_size=a.cell_size, dims=a.dims)

    keys = list(idx.accounts())
    scores = {k: score_from_index(idx, k, cfg=cfg) for k in keys}
    print(f"{len(scores)} accounts, {idx.total_coverage()} cells, "
          f"global tier-3 score {score_global(scores):.3f}")
    print(f"dims={a.dims} cell_size={a.cell_size} threshold={a.threshold}\n")

    owners = store.owners_df().set_index("api_key_id")["owner"].to_dict()
    df = pd.DataFrame([{
        "owner": owners.get(k, "?"), "score": s,
        "cells": idx.key_coverage(k), "reqs": idx.requests_for(k),
        "flagged": s >= a.threshold} for k, s in scores.items()])

    summary = df.groupby("owner").agg(
        accounts=("score", "size"),
        median_score=("score", "median"),
        median_cells=("cells", "median"),
        median_reqs=("reqs", "median"),
        flagged=("flagged", "sum"))
    summary["caught"] = (summary.flagged / summary.accounts).map("{:.0%}".format)
    print(summary.to_string())

    attackers = df.owner.str.startswith("attacker")
    if attackers.any() and (~attackers).any():
        print(f"\ndetection rate on attackers : {df[attackers].flagged.mean():.1%}")
        print(f"false alarm rate on honest  : {df[~attackers].flagged.mean():.1%}")
    else:
        print("\nNo honest accounts in this run -- detection and false-alarm")
        print("rates are meaningless until P4's traffic shares the same log.")


if __name__ == "__main__":
    _main()