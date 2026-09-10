"""
The multi-tenant scenario — README calls this "office behind one IP — key test", and
SCHEMA.md names it explicitly: "P4's office-behind-one-connection test" (column 6, ip).

A whole office of DIFFERENT, legitimate accounts, all sharing one internet connection.
On the surface this looks like a distributed attack: many accounts, low volume each,
one shared IP. The detector must NOT flag this.

*** OPEN QUESTION FOR P3 / P1 — read before demo day, not during it ***
SCHEMA.md: `ip` is written by the API from the real request. If this whole scenario
runs as one script on one machine, EVERY profile in this folder — not just this one —
lands on the API with the same real source IP, because they're all coming from one
laptop. That makes "office, one IP" indistinguishable from "ordinary case" on the ip
column in a same-machine demo, which defeats the point of this specific test.

Two fixes, pick one with the team:
  1. API accepts a trusted test-only header (e.g. X-Debug-IP) that overrides the
     recorded ip in demo mode only, and scenario.py sends a distinct value per profile.
  2. Actually run different profiles from different processes/hosts. More "real" but
     costs setup time.
Recommend option 1 — it's a few lines in api/main.py. Raise this at standup; it affects
whether Stage 5 means anything, not just a nice-to-have.
"""

import random


def office(rng: random.Random, accounts: list[dict], duration_s: int = 3600,
           rate_per_min_per_account: float = 1.0):
    """
    Every account passed in is one legitimate employee at the same company, each
    individually behaving completely normally. `accounts` should all share
    owner="office_acme" (or whatever the team settles on) from the real/stub keys.
    """
    events = []
    for account in accounts:
        t = rng.uniform(0, 60)  # staggered start, like people arriving at work
        gap = 60.0 / rate_per_min_per_account
        while t < duration_s:
            t += rng.uniform(gap * 0.6, gap * 1.4)
            events.append((account, t, False))

    events.sort(key=lambda e: e[1])
    for e in events:
        yield e
