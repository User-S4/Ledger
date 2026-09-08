"""Attribute coverage-spike windows to the accounts that drove new-cell discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cells import assign_cell


@dataclass(frozen=True)
class AttributionConfig:
    """Configurable parameters for spike-window attribution."""

    cell_size: float = 0.2
    ratio_threshold: float = 0.5


def _pre_window_cells(
    logs_df: pd.DataFrame,
    window_start: pd.Timestamp,
    *,
    cell_size: float,
    time_col: str,
    embedding_col: str,
) -> set[tuple[int, ...]]:
    ts = pd.to_datetime(logs_df[time_col])
    pre_window = logs_df.loc[ts < pd.to_datetime(window_start)]
    cells: set[tuple[int, ...]] = set()
    for embedding in pre_window[embedding_col]:
        cells.add(assign_cell(embedding, cell_size))
    return cells


def score_contributions(
    logs_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    cell_size: float = 0.2,
    time_col: str = "timestamp",
    account_col: str = "account_id",
    embedding_col: str = "query_input",
) -> pd.DataFrame:
    """
    Rank accounts by how much they expanded global coverage during a spike window.

    ``new_cells_contributed`` counts distinct cells queried by the account in
    the window that were not covered by anyone before ``window_start``.
    """
    ts = pd.to_datetime(logs_df[time_col])
    start = pd.to_datetime(window_start)
    end = pd.to_datetime(window_end)

    pre_window_cells = _pre_window_cells(
        logs_df,
        start,
        cell_size=cell_size,
        time_col=time_col,
        embedding_col=embedding_col,
    )

    in_window = logs_df.loc[(ts >= start) & (ts <= end)]
    rows: list[dict] = []

    for account_id, group in in_window.groupby(account_col):
        queries_sent = len(group)
        new_cells: set[tuple[int, ...]] = set()

        for embedding in group[embedding_col]:
            cell = assign_cell(embedding, cell_size)
            if cell not in pre_window_cells:
                new_cells.add(cell)

        new_cells_contributed = len(new_cells)
        contribution_ratio = (
            new_cells_contributed / queries_sent if queries_sent else 0.0
        )
        rows.append(
            {
                "account_id": account_id,
                "new_cells_contributed": new_cells_contributed,
                "queries_sent": queries_sent,
                "contribution_ratio": contribution_ratio,
            }
        )

    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored

    return scored.sort_values(
        ["contribution_ratio", "new_cells_contributed", "queries_sent"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def rank_suspects(
    scored_df: pd.DataFrame,
    *,
    top_n: int | None = None,
    ratio_threshold: float = 0.5,
) -> list[str]:
    """
    Return suspect account IDs from a scored attribution table.

    When ``top_n`` is set, returns the highest-ranked ``top_n`` accounts.
    Otherwise returns accounts whose ``contribution_ratio`` exceeds
    ``ratio_threshold``.
    """
    if scored_df.empty:
        return []

    ordered = scored_df.sort_values(
        ["contribution_ratio", "new_cells_contributed", "queries_sent"],
        ascending=[False, False, False],
    )

    if top_n is not None:
        return ordered.head(top_n)["account_id"].tolist()

    return ordered.loc[
        ordered["contribution_ratio"] > ratio_threshold,
        "account_id",
    ].tolist()


if __name__ == "__main__":
    from tier2_perclient import _make_distributed_attack_logs
    from tier3_ledger import detect_distributed_attack

    rng = np.random.default_rng(42)
    logs = _make_distributed_attack_logs(rng)

    windows = detect_distributed_attack(
        logs,
        cell_size=0.2,
        spike_window=20,
        spike_threshold=10,
    )

    print("Scenario 2: Distributed attack - spike-window attribution")
    print("=" * 58)

    if windows.empty:
        print("No flagged coverage window found; skipping attribution.")
    else:
        attacker_windows = windows[
            windows["accounts_in_window"].apply(
                lambda accounts: any(str(a).startswith("attacker_dist_") for a in accounts)
            )
        ]
        flagged = (
            attacker_windows.iloc[0]
            if not attacker_windows.empty
            else windows.iloc[0]
        )
        print(
            "Using flagged window: "
            f"{flagged['window_start']} -> {flagged['window_end']}"
        )
        print()

        scored = score_contributions(
            logs,
            flagged["window_start"],
            flagged["window_end"],
            cell_size=0.2,
        )
        suspects = rank_suspects(scored, ratio_threshold=0.5)

        print("Contribution scores (sorted by contribution_ratio):")
        print(scored.to_string(index=False))
        print()
        print(f"Suspect accounts (ratio > 0.5): {suspects}")

        attacker_suspects = [s for s in suspects if str(s).startswith("attacker_dist_")]
        normal_suspects = [s for s in suspects if str(s).startswith("user_")]
        print()
        print(
            f"Attacker keys in suspect list: {len(attacker_suspects)} / "
            f"{scored['account_id'].astype(str).str.startswith('attacker_dist_').sum()}"
        )
        print(f"Normal users in suspect list: {normal_suspects}")
