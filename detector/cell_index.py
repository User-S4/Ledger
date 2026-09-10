"""Where the ledger lives, and how it stays fast.

P1 decides how the map is drawn (detector/cells.py). This file is P3's half
of Stage 6: the structure that holds the tally and updates on every single
request without slowing the API down.

Why a live structure at all
    A batch job that reads the log afterwards is fine for P1's analysis. It
    is useless for Stage 7, which has to decide *during* a request whether
    to degrade the answer. So the tally has to be current at all times, and
    updating it has to cost microseconds.

What it costs
    Every operation is a dict or set lookup -- no scans, no sorting, nothing
    proportional to how many requests have been seen. Cells are interned to
    small integers on first sight, so per-account storage is a set of ints
    rather than a set of tuples.

What it holds
    per account   which cells that account has revealed
    globally      which cells anyone has revealed, and when first seen
    per cell      which accounts touched it, for P1's attribution (Stage 8)
    recent        a sliding window of discoveries, for rate-of-discovery

Rate matters more than total. An attacker sweeping the map shows up as a
spike in *new cells per minute* long before their total coverage looks
unusual -- and coverage totals also rise slowly for honest heavy users,
which is exactly the false positive we are trying to avoid.
"""

from __future__ import annotations

import sys
import threading
from collections import defaultdict, deque

DEFAULT_WINDOW_S = 300.0     # 5 minutes of history for the discovery rate
DEFAULT_MAX_KEYS_PER_CELL = 64   # attribution detail cap, see note below


class CellIndex:
    """Live coverage tally. Thread-safe, O(1) per observation."""

    def __init__(self, window_s: float = DEFAULT_WINDOW_S,
                 max_keys_per_cell: int = DEFAULT_MAX_KEYS_PER_CELL):
        self.window_s = window_s
        self.max_keys_per_cell = max_keys_per_cell
        self._lock = threading.Lock()

        # Cells are interned: the tuple from cells.assign_cell is seen once
        # and given a small int. Everything downstream stores the int.
        self._cell_ids: dict[tuple, int] = {}
        self._first_seen: list[float] = []      # indexed by cell id

        self._by_key: dict[str, set[int]] = defaultdict(set)
        self._requests: dict[str, int] = defaultdict(int)
        self._new_cells: dict[str, int] = defaultdict(int)

        # Which accounts touched each cell. This is what lets P1 say "these
        # 43 accounts collectively swept the map" in Stage 8. Capped per
        # cell: past a few dozen accounts the cell is plainly public and the
        # extra names add memory without adding evidence.
        self._cell_keys: dict[int, set[str]] = defaultdict(set)

        self._recent: deque[tuple[float, str]] = deque()   # (ts, key) discoveries

    # ------------------------------------------------------------ writing

    def observe(self, api_key_id: str, cell: tuple, ts: float) -> dict:
        """Record one request. Returns what changed, for the caller to log.

        Everything here is a dict or set operation. Nothing scans.
        """
        with self._lock:
            cid = self._cell_ids.get(cell)
            if cid is None:
                cid = len(self._first_seen)
                self._cell_ids[cell] = cid
                self._first_seen.append(ts)
                globally_new = True
            else:
                globally_new = False

            seen = self._by_key[api_key_id]
            new_for_key = cid not in seen
            if new_for_key:
                seen.add(cid)
                self._new_cells[api_key_id] += 1
                self._recent.append((ts, api_key_id))

            self._requests[api_key_id] += 1

            holders = self._cell_keys[cid]
            if len(holders) < self.max_keys_per_cell:
                holders.add(api_key_id)

            self._trim(ts)

            return {
                "cell_id": cid,
                "new_globally": globally_new,
                "new_for_key": new_for_key,
                "key_cells": len(seen),
                "total_cells": len(self._first_seen),
            }

    def _trim(self, now: float) -> None:
        """Drop discoveries that fell out of the window. Caller holds the lock."""
        cutoff = now - self.window_s
        recent = self._recent
        while recent and recent[0][0] < cutoff:
            recent.popleft()

    # ------------------------------------------------------------ reading

    def key_coverage(self, api_key_id: str) -> int:
        with self._lock:
            return len(self._by_key.get(api_key_id, ()))

    def total_coverage(self) -> int:
        with self._lock:
            return len(self._first_seen)

    def requests_for(self, api_key_id: str) -> int:
        with self._lock:
            return self._requests.get(api_key_id, 0)

    def accounts(self) -> list:
        with self._lock:
            return list(self._by_key)

    def efficiency(self, api_key_id: str) -> float:
        """Requests spent per new cell. Low means almost nothing is repeated.

        On its own this does NOT separate attackers from honest users: a
        researcher probing edge cases is efficient too. It is one input, not
        a verdict.
        """
        with self._lock:
            n = self._new_cells.get(api_key_id, 0)
            r = self._requests.get(api_key_id, 0)
            return (r / n) if n else float(r)

    def discovery_rate(self, api_key_id: str | None = None,
                       now: float | None = None) -> float:
        """New cells per minute, over the window. The signal that moves first."""
        with self._lock:
            if now is not None:
                self._trim(now)
            if api_key_id is None:
                n = len(self._recent)
            else:
                n = sum(1 for _, k in self._recent if k == api_key_id)
            return n / (self.window_s / 60.0)

    def cohort(self, api_key_id: str, min_shared: int = 1) -> set[str]:
        """Accounts whose cells overlap this one's. Raw material for P1's
        Stage 8 attribution -- which accounts moved together."""
        with self._lock:
            mine = self._by_key.get(api_key_id, set())
            counts: dict[str, int] = defaultdict(int)
            for cid in mine:
                for other in self._cell_keys.get(cid, ()):
                    if other != api_key_id:
                        counts[other] += 1
            return {k for k, c in counts.items() if c >= min_shared}

    def snapshot(self) -> dict:
        """Cheap summary for /stats and P5's dashboard."""
        with self._lock:
            return {
                "cells_revealed": len(self._first_seen),
                "accounts_seen": len(self._by_key),
                "requests_seen": sum(self._requests.values()),
                "discoveries_in_window": len(self._recent),
                "window_s": self.window_s,
            }

    def memory_estimate(self) -> int:
        """Rough bytes held. Worth watching during a long attack run."""
        with self._lock:
            n_cells = len(self._first_seen)
            n_key_entries = sum(len(s) for s in self._by_key.values())
            n_cell_keys = sum(len(s) for s in self._cell_keys.values())
            return (n_cells * 120 + n_key_entries * 60 + n_cell_keys * 60
                    + sys.getsizeof(self._recent))

    def reset(self) -> None:
        with self._lock:
            self._cell_ids.clear()
            self._first_seen.clear()
            self._by_key.clear()
            self._requests.clear()
            self._new_cells.clear()
            self._cell_keys.clear()
            self._recent.clear()


# ------------------------------------------------------------------ rebuild

def rebuild_from_log(store, run_id: str, cell_fn=None, **index_kwargs) -> CellIndex:
    """Replay a logged run into a fresh index.

    Two uses. The API restarts mid-experiment and needs its tally back --
    the log is the source of truth, so it can always be rebuilt. And P1 can
    replay the same run under a different map design without regenerating a
    single request, because column 12 stores the point rather than the cell.
    """
    if cell_fn is None:
        from detector.cells import assign_cell as cell_fn

    idx = CellIndex(**index_kwargs)
    df = store.read_df(run_id=run_id)
    df = df[(df.status_code == 200) & df.embedding.notna()]
    for key, ts, emb in zip(df.api_key_id, df.ts, df.embedding):
        idx.observe(key, cell_fn(emb), float(ts))
    return idx


# ------------------------------------------------------------------ CLI

def _main() -> None:
    import argparse
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.logstore import LogStore
    from detector.cells import DEFAULT_CELL_SIZE, DEFAULT_DIMS, assign_cell

    ap = argparse.ArgumentParser(description="Replay a run into the live index")
    ap.add_argument("--db", default="data/fake.db")
    ap.add_argument("--run-id", default="cal_seed1")
    ap.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    ap.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE)
    a = ap.parse_args()

    store = LogStore(a.db)
    t0 = time.perf_counter()
    idx = rebuild_from_log(
        store, a.run_id,
        cell_fn=lambda e: assign_cell(e, a.cell_size, a.dims),
    )
    elapsed = time.perf_counter() - t0
    snap = idx.snapshot()

    print(f"replayed {snap['requests_seen']} requests in {elapsed:.2f}s "
          f"({snap['requests_seen']/max(elapsed,1e-9):,.0f}/sec)")
    print(f"dims={a.dims} cell_size={a.cell_size}")
    for k, v in snap.items():
        print(f"  {k}: {v}")
    print(f"  memory: ~{idx.memory_estimate()/1e6:.1f} MB")

    owners = store.owners_df().set_index("api_key_id")["owner"].to_dict()
    per_owner: dict[str, dict] = {}
    for key in idx._by_key:
        o = per_owner.setdefault(owners.get(key, "?"),
                                 {"accounts": 0, "cells": 0, "reqs": 0})
        o["accounts"] += 1
        o["cells"] += idx.key_coverage(key)
        o["reqs"] += idx._requests[key]

    print(f"\n  {'owner':18s} {'accts':>6} {'reqs':>8} {'cells/acct':>11} "
          f"{'reqs/new cell':>14}")
    for owner, v in sorted(per_owner.items(), key=lambda kv: -kv[1]["cells"]):
        print(f"  {owner:18s} {v['accounts']:>6} {v['reqs']:>8} "
              f"{v['cells']/v['accounts']:>11.1f} "
              f"{v['reqs']/max(v['cells'],1):>14.1f}")


if __name__ == "__main__":
    _main()