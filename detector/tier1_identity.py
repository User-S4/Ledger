"""Tier 1: the cheap identity checks. Rate, IP, subnet.

These are the signals any team would reach for first, and they are computed
from columns 3, 4, 5 and 6 alone -- no model output, no embeddings, nothing
expensive. They run in microseconds and they are the first thing a real
service would deploy.

They are also, deliberately, not enough. Two known failures, both visible in
our own data before P2 has written an attack:

  - An office of sixty legitimate accounts behind one connection looks
    structurally identical to a distributed attack. Flagging it means
    turning away a paying enterprise customer.

  - A distributed attacker running nine requests per minute per account is
    quieter than that office. No rate threshold reaches it.

Building this properly is what makes Stage 5 an honest result rather than a
claim. You cannot show the ordinary guard failing unless you built one.

Two consumers, one definition:
    score_from_log(df)      batch, over a logged run -- what P1 analyses
    LiveTier1               streaming, per request -- what feeds Stage 7

Both produce the same shape: one score per account in 0..1, plus one global
score for the run. That is the format agreed with P1 for the suspicion
handoff.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

DEFAULT_TIERS = {"free": 60, "pro": 600, "enterprise": 6000}


def subnet24(ip: str) -> str:
    """Column 6 is the full address. The /24 is one line, so we derive it
    rather than storing a second copy -- see SCHEMA.md."""
    parts = str(ip).split(".")
    return ".".join(parts[:3]) if len(parts) == 4 else str(ip)


# ------------------------------------------------------------------ signals

def rate_utilisation(df: pd.DataFrame, tiers=None, window_s: float = 60.0) -> pd.Series:
    """Peak requests per minute as a fraction of what the tier allows.

    Utilisation, not raw rate. An enterprise account at 3000/min is using
    half its allowance; a free account at 70/min is over its own. Raw rate
    would flag the paying customer and miss the cheat.
    """
    tiers = tiers or DEFAULT_TIERS
    out = {}
    for key, g in df.groupby("api_key_id"):
        ts = np.sort(g["ts"].to_numpy())
        limit = tiers.get(g["tier"].iloc[0], 60)
        if len(ts) < 2:
            out[key] = len(ts) / limit
            continue
        # Busiest window: for each request, how many fall in the preceding
        # window. searchsorted keeps this linear rather than quadratic.
        left = np.searchsorted(ts, ts - window_s, side="left")
        peak = int((np.arange(len(ts)) - left + 1).max())
        out[key] = peak / limit
    return pd.Series(out, name="rate_utilisation", dtype=float)


def keys_per_ip(df: pd.DataFrame) -> pd.Series:
    """How many accounts share this account's address.

    High is suspicious in theory. In practice it is also every office, every
    university, and every mobile network. This is the signal that produces
    P4's false positive, on purpose.
    """
    per_ip = df.groupby("ip")["api_key_id"].nunique()
    return df.groupby("api_key_id")["ip"].first().map(per_ip).rename("keys_per_ip")


def keys_per_subnet(df: pd.DataFrame) -> pd.Series:
    """The same question one level wider, because an attacker who reads a
    security blog will spread across nearby addresses rather than reuse one."""
    sn = df["ip"].map(subnet24)
    per_sn = df.assign(subnet=sn).groupby("subnet")["api_key_id"].nunique()
    return (df.assign(subnet=sn).groupby("api_key_id")["subnet"].first()
              .map(per_sn).rename("keys_per_subnet"))


def error_fraction(df: pd.DataFrame) -> pd.Series:
    """Probing shows up as failures. Cheap, and it costs nothing to include."""
    return (df.assign(bad=df["status_code"] >= 400)
              .groupby("api_key_id")["bad"].mean().rename("error_fraction"))


# ------------------------------------------------------------------ scoring

def score_from_log(df: pd.DataFrame, tiers=None,
                   rate_threshold: float = 0.8,
                   ip_threshold: int = 25,
                   subnet_threshold: int = 40) -> pd.DataFrame:
    """Per-account tier-1 signals and a score in 0..1.

    Thresholds live in config.yaml under detector.tier1 -- P1 owns the
    values, P3 owns the shape.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "requests", "rate_utilisation", "keys_per_ip", "keys_per_subnet",
            "error_fraction", "tier1_score", "tier1_flag"])

    feats = pd.concat([
        df.groupby("api_key_id").size().rename("requests"),
        rate_utilisation(df, tiers),
        keys_per_ip(df),
        keys_per_subnet(df),
        error_fraction(df),
    ], axis=1)

    # Each component saturates at 1. Deliberately blunt -- this tier is meant
    # to be the obvious approach, not a tuned one.
    rate_c = (feats["rate_utilisation"] / rate_threshold).clip(0, 1)
    ip_c = (feats["keys_per_ip"] / ip_threshold).clip(0, 1)
    sn_c = (feats["keys_per_subnet"] / subnet_threshold).clip(0, 1)
    err_c = (feats["error_fraction"] / 0.25).clip(0, 1)

    feats["tier1_score"] = (0.45 * rate_c + 0.25 * ip_c
                            + 0.20 * sn_c + 0.10 * err_c).round(4)
    feats["tier1_flag"] = feats["tier1_score"] >= 0.5
    return feats.sort_values("tier1_score", ascending=False)


def global_score(feats: pd.DataFrame) -> float:
    """One number for the whole run, as agreed with P1 for Stage 7.

    The mean of the top decile rather than the overall mean: a handful of
    hostile accounts among thousands of honest ones should move this, and an
    average over everyone would drown them.
    """
    if feats.empty:
        return 0.0
    s = feats["tier1_score"].sort_values(ascending=False)
    top = s.head(max(1, len(s) // 10))
    return float(top.mean().round(4))


# ------------------------------------------------------------------ live

class LiveTier1:
    """Streaming version, for the API to update on every request.

    Keeps a per-account sliding window and a running account-per-address
    count. Bounded memory, no database reads on the hot path. This is what
    Stage 7's defence will read.
    """

    def __init__(self, tiers=None, window_s: float = 60.0,
                 rate_threshold: float = 0.8, ip_threshold: int = 25):
        self.tiers = tiers or DEFAULT_TIERS
        self.window_s = window_s
        self.rate_threshold = rate_threshold
        self.ip_threshold = ip_threshold
        self._hits: dict[str, deque] = defaultdict(deque)
        self._ip_keys: dict[str, set] = defaultdict(set)
        self._scores: dict[str, float] = {}

    def observe(self, api_key_id: str, tier: str, ip: str, ts: float) -> float:
        """Record one request, return this account's updated score."""
        q = self._hits[api_key_id]
        q.append(ts)
        cutoff = ts - self.window_s
        while q and q[0] < cutoff:
            q.popleft()

        self._ip_keys[ip].add(api_key_id)

        limit = self.tiers.get(tier, 60)
        rate_c = min(1.0, (len(q) / limit) / self.rate_threshold)
        ip_c = min(1.0, len(self._ip_keys[ip]) / self.ip_threshold)

        score = round(0.65 * rate_c + 0.35 * ip_c, 4)
        self._scores[api_key_id] = score
        return score

    def score(self, api_key_id: str) -> float:
        return self._scores.get(api_key_id, 0.0)

    def global_score(self) -> float:
        if not self._scores:
            return 0.0
        vals = sorted(self._scores.values(), reverse=True)
        top = vals[: max(1, len(vals) // 10)]
        return round(sum(top) / len(top), 4)

    def reset(self) -> None:
        self._hits.clear()
        self._ip_keys.clear()
        self._scores.clear()


# ------------------------------------------------------------------ CLI

def _main() -> None:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.logstore import LogStore

    ap = argparse.ArgumentParser(description="Tier-1 identity signals")
    ap.add_argument("--db", default="data/fake.db")
    ap.add_argument("--run-id", default="cal_seed1")
    ap.add_argument("--score-only", action="store_true",
                    help="skip the ground-truth breakdown")
    a = ap.parse_args()

    store = LogStore(a.db)
    df = store.read_df(run_id=a.run_id)
    feats = score_from_log(df)

    print(f"{len(df)} rows, {len(feats)} accounts, "
          f"global tier-1 score {global_score(feats):.3f}\n")

    if a.score_only:
        print(feats.head(15).to_string())
        return

    owners = store.owners_df().set_index("api_key_id")["owner"]
    feats = feats.join(owners)

    by_owner = feats.groupby("owner").agg(
        accounts=("requests", "size"),
        median_rate_util=("rate_utilisation", "median"),
        median_keys_per_ip=("keys_per_ip", "median"),
        median_score=("tier1_score", "median"),
        flagged=("tier1_flag", "sum"),
    )
    by_owner["caught"] = (by_owner.flagged / by_owner.accounts).map("{:.0%}".format)
    print(by_owner.to_string())

    attackers = {"attacker_single", "attacker_pool"}
    hostile = feats[feats.owner.isin(attackers)]
    honest = feats[~feats.owner.isin(attackers)]
    print(f"\ndetection rate on attackers : {hostile.tier1_flag.mean():.1%}")
    print(f"false alarm rate on honest  : {honest.tier1_flag.mean():.1%}")
    print("\nnumbers from fake data. Not reportable -- see SCHEMA.md.")


if __name__ == "__main__":
    _main()