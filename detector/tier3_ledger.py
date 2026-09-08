"""Tier-3 global coverage ledger for detecting distributed extraction attacks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cells import assign_cell


@dataclass
class Tier3Config:
    """Configurable parameters for global coverage spike detection."""

    cell_size: float = 0.2
    spike_window: int = 20
    spike_threshold: int = 10


@dataclass
class GlobalCoverageLedger:
    """
    Running ledger of unique embedding cells seen across all accounts.

    Tracks which accounts touched each cell and detects bursts of new-cell
    discovery that may indicate coordinated distributed probing.
    """

    cell_size: float = 0.2
    spike_window: int = 20
    _cells: set[tuple[int, ...]] = field(default_factory=set, init=False, repr=False)
    _cell_accounts: dict[tuple[int, ...], set[str]] = field(
        default_factory=lambda: defaultdict(set),
        init=False,
        repr=False,
    )
    _recent_new: deque[bool] = field(default_factory=deque, init=False, repr=False)
    _recent_meta: deque[dict] = field(default_factory=deque, init=False, repr=False)
    _query_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._recent_new = deque(maxlen=self.spike_window)
        self._recent_meta = deque(maxlen=self.spike_window)

    def update(self, account_id: str, embedding: list[float] | np.ndarray) -> tuple[int, ...]:
        """Record a query embedding and return its assigned cell."""
        cell = assign_cell(embedding, self.cell_size)
        is_new = cell not in self._cells
        self._cells.add(cell)
        self._cell_accounts[cell].add(account_id)
        self._recent_new.append(is_new)
        self._recent_meta.append(
            {
                "account_id": account_id,
                "cell": cell,
                "is_new_cell": is_new,
            }
        )
        self._query_count += 1
        return cell

    def coverage(self) -> int:
        """Total number of unique cells covered so far."""
        return len(self._cells)

    def spike_rate(self, window_size: int | None = None) -> int:
        """Count of new unique cells added in the last ``window_size`` queries."""
        window = window_size or self.spike_window
        recent = list(self._recent_new)[-window:]
        return int(sum(recent))

    def flag_if_spiking(self, threshold: float | int) -> bool:
        """Return True when recent new-cell count exceeds ``threshold``."""
        if len(self._recent_new) < self.spike_window:
            return False
        return self.spike_rate() > threshold

    def accounts_for_cell(self, cell: tuple[int, ...]) -> set[str]:
        """Return account IDs that have touched a cell."""
        return set(self._cell_accounts.get(cell, set()))


def detect_distributed_attack(
    logs_df: pd.DataFrame,
    *,
    cell_size: float = 0.2,
    spike_window: int = 20,
    spike_threshold: int = 10,
    time_col: str = "timestamp",
    account_col: str = "account_id",
    embedding_col: str = "query_input",
) -> pd.DataFrame:
    """
    Process logs in timestamp order and flag windows with fast coverage growth.

    Returns a DataFrame of triggered windows with window boundaries, spike
    rate, and cumulative coverage at detection time.
    """
    ledger = GlobalCoverageLedger(cell_size=cell_size, spike_window=spike_window)
    ordered = logs_df.sort_values(time_col).reset_index(drop=True)

    flagged_windows: list[dict] = []
    in_spike = False

    for idx, row in ordered.iterrows():
        ledger.update(row[account_col], row[embedding_col])
        currently_spiking = ledger.flag_if_spiking(spike_threshold)

        if currently_spiking and not in_spike:
            window_start_idx = max(0, idx - spike_window + 1)
            window_accounts = {
                entry["account_id"] for entry in list(ledger._recent_meta)
            }
            flagged_windows.append(
                {
                    "window_start": ordered.loc[window_start_idx, time_col],
                    "window_end": row[time_col],
                    "query_index_start": window_start_idx,
                    "query_index_end": idx,
                    "spike_rate": ledger.spike_rate(),
                    "coverage": ledger.coverage(),
                    "accounts_in_window": sorted(window_accounts),
                }
            )

        in_spike = currently_spiking

    if not flagged_windows:
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "query_index_start",
                "query_index_end",
                "spike_rate",
                "coverage",
                "accounts_in_window",
            ]
        )

    return pd.DataFrame(flagged_windows)


if __name__ == "__main__":
    from tier2_perclient import _make_distributed_attack_logs

    rng = np.random.default_rng(42)
    logs = _make_distributed_attack_logs(rng)

    print("Scenario 2: Distributed attack - Tier-3 global coverage ledger")
    print("=" * 62)

    windows = detect_distributed_attack(
        logs,
        cell_size=0.2,
        spike_window=20,
        spike_threshold=10,
    )

    detected = not windows.empty
    print(f"Coverage spike detected: {detected}")
    print(f"Total queries processed: {len(logs)}")
    print()

    if detected:
        print("Flagged windows:")
        print(windows.to_string(index=False))
        first = windows.iloc[0]
        print()
        print(
            "First spike roughly at "
            f"{first['window_start']} -> {first['window_end']} "
            f"({first['spike_rate']} new cells in window, "
            f"{first['coverage']} total cells covered)"
        )
    else:
        print("No coverage spike windows exceeded the threshold.")
