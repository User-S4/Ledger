"""
The multi-tenant scenario — README calls this "office behind one IP — key test", and
SCHEMA.md names it explicitly: "P4's office-behind-one-connection test" (column 6, ip).

A whole office of DIFFERENT, legitimate accounts, all sharing one internet connection.
On the surface this looks like a distributed attack: many accounts, low volume each,
one shared IP. The detector must NOT flag this.

RESOLVED (confirmed with P3): api/main.py already trusts X-Forwarded-For as the
recorded ip (SCHEMA.md column 6), and api/client.py's predict(ip=...) sets exactly
that header. No API change needed. scenario.py gives every account in this scenario
the SAME simulated IP on purpose — that's the entire point of this test — while every
other profile gets its own distinct IP (see assign_ips() in scenario.py), so this
scenario is actually distinguishable from "just a bunch of separate honest users."
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
