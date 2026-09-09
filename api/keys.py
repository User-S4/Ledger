"""Accounts, tiers and rate limits.

Why tiers exist
    If every account had the same limit, high volume would be an obvious
    giveaway and there would be no project. Real services sell tiers, and an
    enterprise customer legitimately sends thousands of requests a minute.
    That is what makes a request counter useless as a detector -- you cannot
    separate the paying batch job from the thief by volume alone.

    The tier is logged (column 5) so P1 can ask "is this volume paid for?"
    rather than just "is this volume high?".

Why rate limits exist
    They are not the defence. They are the reason the attack has to be
    distributed in the first place: a single free-tier key cannot pull a
    million answers, so a serious thief must spread across accounts -- which
    is precisely the case Stage 5 shows the ordinary guard missing.

Bulk provisioning
    P2 needs a few hundred accounts for the distributed attack. P4 needs
    sixty for the office. Nobody creates those by hand:

        python -m api.keys --count 300 --tier free --owner attacker_pool \\
            --out data/attacker_keys.json
        python -m api.keys --count 60 --tier free --owner office_acme \\
            --out data/office_keys.json

    `owner` is ground truth for scoring. The detector must never read it.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

# Requests per minute. Overridden by the `tiers` block in config.yaml.
DEFAULT_TIERS = {"free": 60, "pro": 600, "enterprise": 6000}
DEFAULT_TIER = "free"


def load_tiers(cfg: dict | None = None) -> dict:
    if cfg and isinstance(cfg.get("tiers"), dict):
        return {str(k): int(v) for k, v in cfg["tiers"].items()}
    return dict(DEFAULT_TIERS)


def new_secret(prefix: str = "sk") -> str:
    """Unguessable. Attackers in this project are given keys, not stealing
    them, but a guessable key would make the logs meaningless."""
    return f"{prefix}_{secrets.token_urlsafe(24)}"


def format_key_id(n: int) -> str:
    return f"k_{n:05d}"


class RateLimiter:
    """Sliding 60-second window per account.

    In memory, so it resets when the API restarts. That is fine here and
    worth knowing: a restart mid-experiment gives everyone a clean slate.
    """

    def __init__(self, tiers: dict | None = None, window_s: float = 60.0):
        self.tiers = tiers or dict(DEFAULT_TIERS)
        self.window_s = window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def limit_for(self, tier: str) -> int:
        return self.tiers.get(tier, self.tiers.get(DEFAULT_TIER, 60))

    def allow(self, api_key_id: str, tier: str, now: float | None = None) -> bool:
        """True if this request is within the account's limit, and records it."""
        limit = self.limit_for(tier)
        now = time.time() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            q = self._hits[api_key_id]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def current_rate(self, api_key_id: str, now: float | None = None) -> int:
        """Requests in the last window. P3's Stage 4 tier-1 signal."""
        now = time.time() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            q = self._hits.get(api_key_id)
            if not q:
                return 0
            while q and q[0] < cutoff:
                q.popleft()
            return len(q)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class KeyRegistry:
    """Maps a secret to an account. Cached in memory, backed by the keys table.

    The cache matters: every request looks up a key, and hitting SQLite each
    time would put database latency on the hot path and distort the timing
    P1 sees in the log.
    """

    def __init__(self, store, tiers: dict | None = None):
        self.store = store
        self.tiers = tiers or dict(DEFAULT_TIERS)
        self._cache: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        rows = self.store.all_keys()
        with self._lock:
            self._cache = {r["secret"]: (r["api_key_id"], r["tier"]) for r in rows}

    def resolve(self, secret: str | None):
        """secret -> (api_key_id, tier), or None if unknown."""
        if not secret:
            return None
        with self._lock:
            hit = self._cache.get(secret)
        if hit:
            return hit
        # Miss: a key provisioned by another process since we last reloaded.
        row = self.store.lookup_secret(secret)
        if row is None:
            return None
        hit = (row["api_key_id"], row["tier"])
        with self._lock:
            self._cache[secret] = hit
        return hit

    def next_index(self) -> int:
        """Highest existing k_NNNNN, plus one. Keeps ids unique across calls."""
        highest = -1
        for r in self.store.all_keys():
            kid = r["api_key_id"]
            if kid.startswith("k_") and kid[2:].isdigit():
                highest = max(highest, int(kid[2:]))
        return highest + 1

    def provision(self, count: int, tier: str, owner: str,
                  start: int | None = None) -> list[dict]:
        """Create `count` accounts. Returns the plaintext secrets.

        This is the only time secrets are visible -- save the output.
        """
        if count < 1:
            raise ValueError("count must be at least 1")
        if tier not in self.tiers:
            raise ValueError(f"unknown tier {tier!r}; have {sorted(self.tiers)}")
        if not owner:
            raise ValueError("owner is required -- it is the ground truth label")

        base = self.next_index() if start is None else start
        now = time.time()
        made, rows = [], []
        for i in range(count):
            kid = format_key_id(base + i)
            sec = new_secret()
            made.append({"api_key_id": kid, "secret": sec,
                         "tier": tier, "owner": owner})
            rows.append((kid, sec, tier, owner, now))
        self.store.upsert_keys(rows)
        self.reload()
        return made


# ------------------------------------------------------------------ CLI

def _main() -> None:
    import argparse
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.logstore import LogStore

    ap = argparse.ArgumentParser(description="Provision API accounts")
    ap.add_argument("--db", default="data/ledger.db")
    ap.add_argument("--count", type=int, help="how many accounts to create")
    ap.add_argument("--tier", default="free", choices=sorted(DEFAULT_TIERS))
    ap.add_argument("--owner",
                    help="ground-truth label: attacker_pool, office_acme, casual...")
    ap.add_argument("--out", help="write the secrets to this JSON file")
    ap.add_argument("--list", action="store_true", help="show existing accounts")
    a = ap.parse_args()

    store = LogStore(a.db)
    reg = KeyRegistry(store)

    if a.list:
        rows = store.all_keys()
        by_owner: dict[str, list] = {}
        for r in rows:
            by_owner.setdefault(f"{r['owner']} ({r['tier']})", []).append(r["api_key_id"])
        print(f"{len(rows)} accounts in {a.db}")
        for label, ids in sorted(by_owner.items()):
            print(f"  {label:32s} {len(ids):5d}   {ids[0]} .. {ids[-1]}")
        return

    if not a.count or not a.owner:
        ap.error("--count and --owner are required (or use --list)")

    made = reg.provision(a.count, a.tier, a.owner)
    text = json.dumps(made, indent=2)

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"{len(made)} {a.tier} accounts for '{a.owner}' -> {a.out}")
        print(f"  {made[0]['api_key_id']} .. {made[-1]['api_key_id']}")
        print("  secrets are only shown once -- do not commit this file")
    else:
        print(text)


if __name__ == "__main__":
    _main()