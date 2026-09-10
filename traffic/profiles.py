"""
Honest-user traffic profiles (Stage 3 / Stage 5 deliverable).

Per SCHEMA.md, only the API writes log rows — these functions only decide WHO sends a
request and WHEN. `accounts` is a list of dicts like {"api_key_id": ..., "secret": ...,
"tier": ..., "owner": ...}, built by scenario.py from a real or stub keys source.

Each profile yields: (account: dict, time_offset_seconds: float, weird: bool)
time_offset_seconds is simulated-scenario time — scenario.py compresses it into real
wall-clock time via a speed multiplier. Nothing here writes anything anywhere.
"""

import random


def casual(rng: random.Random, accounts: list[dict], duration_s: int = 3600,
           rate_per_min: float = 2.0):
    """One ordinary account, asking at a slow constant rate all day. The baseline case."""
    account = rng.choice(accounts)
    t = 0.0
    gap = 60.0 / rate_per_min
    while t < duration_s:
        t += rng.uniform(gap * 0.5, gap * 1.5)
        yield account, t, False


def batch(rng: random.Random, accounts: list[dict], duration_s: int = 3600,
          burst_size: int = 400):
    """
    An overnight batch job: silence, then one big tight burst, then silence again.
    High volume from one predictable account doing one predictable thing — should NOT
    be confused with an attacker. (This is why SCHEMA.md's `tier` column exists — an
    enterprise batch customer at high rate isn't automatically a suspect.)
    """
    account = rng.choice(accounts)
    burst_start = rng.uniform(duration_s * 0.3, duration_s * 0.5)
    for _ in range(burst_size):
        t = burst_start + rng.uniform(0, 60)  # whole burst inside ~1 minute
        yield account, t, False


def bursty(rng: random.Random, accounts: list[dict], duration_s: int = 3600,
           rate_per_min: float = 3.0, retry_chance: float = 0.15):
    """
    A normal app with flaky networking: sometimes it re-sends the same request 2-3 times
    within a couple seconds because it didn't see the first response in time. Different
    from `batch` — this is small, frequent bursts, not one big overnight one.
    """
    account = rng.choice(accounts)
    t = 0.0
    gap = 60.0 / rate_per_min
    while t < duration_s:
        t += rng.uniform(gap * 0.5, gap * 1.5)
        yield account, t, False
        if rng.random() < retry_chance:
            for _ in range(rng.randint(1, 2)):
                t += rng.uniform(0.5, 2.0)
                yield account, t, False


def researcher(rng: random.Random, accounts: list[dict], duration_s: int = 3600,
               rate_per_min: float = 1.0):
    """
    Researcher(s) poking the model with deliberately odd inputs at low volume — blank
    images, blown-out images, pure noise. Low volume, but weird content: this is the
    case most likely to false-positive on a naive 'weird input' rule.
    """
    for account in accounts:
        t = 0.0
        gap = 60.0 / rate_per_min
        while t < duration_s:
            t += rng.uniform(gap * 0.5, gap * 1.5)
            yield account, t, True


PROFILES = {
    "casual": casual,
    "batch": batch,
    "bursty": bursty,
    "researcher": researcher,
}
