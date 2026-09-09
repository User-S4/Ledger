"""Stage 2: invented log rows that satisfy the frozen contract.

Why this exists
    Without it, four people wait for the API. With it, P1 writes detector
    code against real-shaped data from hour three of Day 1.

What it is not
    It is not a copy of the real logs and the real logs are not a copy of
    it. Both are written to satisfy SCHEMA.md. That is why P1 can swap one
    for the other without changing a line -- same columns, same types, same
    meanings, different origin for the values.

Populations
    casual          slow, scattered, ordinary customers
    batch           an overnight enterprise job -- huge volume, legitimate
    bursty          a retrying app -- spiky, legitimate
    researcher      odd inputs, low-confidence answers, legitimate
    office_acme     60 accounts behind ONE connection, all legitimate
    attacker_single one loud account sweeping the model's knowledge
    attacker_pool   the SAME sweep split across 120 quiet accounts

The last two populations are the project. `attacker_single` should be easy
to catch. `attacker_pool` should be nearly invisible per account, and
`office_acme` exists so that "many accounts on one connection" cannot be
used as a shortcut -- it would flag sixty honest customers.

Usage
    python -m api.fake_logs --db data/fake.db --run-id cal_seed1 --n 20000

P1 then reads it exactly as they will read real logs:
    from api.logstore import LogStore
    df = LogStore("data/fake.db").read_df(run_id="cal_seed1")

Ground truth is in the `keys` table `owner` column. Use it to SCORE
detections, never as a feature.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.logstore import (  # noqa: E402
    EMBED_DIM, NUM_CLASSES, LogStore, encode_embedding, encode_probs,
)

N_ANCHORS = 48        # landmarks in the model's knowledge


class Population:
    """A group of accounts that query in a similar way."""

    def __init__(self, name, owner, n_keys, tier, subnet, one_ip,
                 rate_weight, breadth, confidence, repeat_rate, key_offset):
        self.name = name
        self.owner = owner                # ground truth label
        self.n_keys = n_keys
        self.tier = tier
        self.subnet = subnet
        self.one_ip = one_ip              # all accounts share a single address
        self.rate_weight = rate_weight    # relative share of traffic per key
        self.breadth = breadth            # 0 = stays home, 1 = roams everywhere
        self.confidence = confidence      # (mean, sd) of the model's top answer
        self.repeat_rate = repeat_rate    # fraction of queries that reuse an image
        self.key_offset = key_offset


POPULATIONS = [
    #           name          owner            keys tier         subnet       one_ip  rate breadth conf          rpt   offset
    Population("casual",      "casual",          40, "free",       "198.51.100", False,   4, 0.10, (0.88, 0.09), 0.02,  1000),
    Population("batch",       "batch",            3, "enterprise", "198.51.101", False, 900, 0.35, (0.86, 0.10), 0.01,  2000),
    Population("bursty",      "bursty",           8, "pro",        "198.51.102", False, 120, 0.15, (0.87, 0.10), 0.15,  3000),
    Population("researcher",  "researcher",       4, "pro",        "198.51.103", False,  25, 0.45, (0.52, 0.18), 0.03,  4000),
    Population("office",      "office_acme",     60, "free",       "203.0.113",  True,   12, 0.12, (0.88, 0.09), 0.04,  5000),
    Population("atk_single",  "attacker_single",  1, "pro",        "192.0.2",    False, 400, 0.95, (0.71, 0.20), 0.00,  9000),
    Population("atk_pool",    "attacker_pool",  120, "free",       "192.0.2",    False,   9, 0.95, (0.71, 0.20), 0.00,  9100),
]

ERROR_KINDS = [
    (400, "bad_image"),
    (401, "bad_key"),
    (413, "payload_too_large"),
    (429, "rate_limited"),
]


# ------------------------------------------------------------ the space

def make_anchors(rng):
    """Landmarks in the model's knowledge. Clients cluster near a few."""
    return rng.normal(size=(N_ANCHORS, EMBED_DIM)).astype(np.float32) * 1.6


def sample_point(rng, anchors, home, breadth):
    """Column 12: where in the model's knowledge this question landed.

    A narrow client returns to its own few landmarks. A broad one goes
    anywhere. That difference is the whole signal, and it survives being
    split across accounts -- which is exactly what Stage 6 exploits.
    """
    if rng.random() < breadth:
        anchor, jitter = anchors[rng.integers(N_ANCHORS)], 1.0
    else:
        anchor, jitter = anchors[home[rng.integers(len(home))]], 0.45
    return (anchor + rng.normal(size=EMBED_DIM) * jitter).astype(np.float32)


def sample_probs(rng, top1):
    """Column 13: a plausible 10-class probability vector."""
    top1 = float(np.clip(top1, 0.15, 0.999))
    p = np.empty(NUM_CLASSES)
    label = int(rng.integers(NUM_CLASSES))
    p[label] = top1
    p[np.arange(NUM_CLASSES) != label] = (
        rng.dirichlet(np.ones(NUM_CLASSES - 1) * 0.6) * (1.0 - top1)
    )
    return p


# ------------------------------------------------------------ generation

def build_clients(rng):
    clients, key_rows = [], []
    now = time.time()
    for pop in POPULATIONS:
        home = rng.choice(N_ANCHORS, size=max(2, int(N_ANCHORS * 0.12)),
                          replace=False)
        for i in range(pop.n_keys):
            kid = f"k_{pop.key_offset + i:05d}"
            key_rows.append((kid, f"sk_fake_{kid}", pop.tier, pop.owner, now))
            ip = (f"{pop.subnet}.7" if pop.one_ip
                  else f"{pop.subnet}.{1 + (i % 250)}")
            clients.append({"kid": kid, "pop": pop, "home": home, "ip": ip,
                            "images": []})
    return clients, key_rows


def sample_time(rng, pop, t0, span):
    """Batch jobs clump overnight. Bursty apps spike. Everyone else spreads."""
    if pop.name == "batch":
        return t0 + span * (0.15 + 0.25 * rng.random())
    if pop.name == "bursty":
        return t0 + span * (rng.integers(0, 8) / 8.0 + 0.02 * rng.random())
    return t0 + span * rng.random()


def generate(db_path, run_id, n_requests, seed, hours, error_rate):
    rng = np.random.default_rng(seed)
    store = LogStore(db_path, batch_size=2000)
    anchors = make_anchors(rng)

    clients, key_rows = build_clients(rng)
    store.upsert_keys(key_rows)

    weights = np.array([c["pop"].rate_weight for c in clients], dtype=np.float64)
    weights /= weights.sum()

    now = time.time()
    t0, span = now - hours * 3600, hours * 3600
    counts: dict[str, int] = {}

    for _ in range(n_requests):
        c = clients[int(rng.choice(len(clients), p=weights))]
        pop = c["pop"]

        # Column 10: repeats are how junk padding traffic gives itself away.
        if c["images"] and rng.random() < pop.repeat_rate:
            digest, size = c["images"][int(rng.integers(len(c["images"])))]
        else:
            digest = f"{rng.integers(1 << 62):016x}{rng.integers(1 << 62):016x}"
            size = int(abs(rng.normal(2400, 400))) + 200
            if len(c["images"]) < 200:
                c["images"].append((digest, size))

        point = sample_point(rng, anchors, c["home"], pop.breadth)
        probs = sample_probs(rng, rng.normal(*pop.confidence))
        encoded = encode_probs(probs)

        store.log(
            run_id=run_id,
            ts=float(sample_time(rng, pop, t0, span)),
            api_key_id=c["kid"],
            tier=pop.tier,
            ip=c["ip"],
            status_code=200,
            error_code=None,
            latency_ms=float(rng.gamma(3.0, 4.0)),
            input_sha256=digest,
            input_bytes=size,
            embedding=encode_embedding(point),
            model_probs=encoded,
            # Identical to model_probs until Stage 7 exists. That is correct,
            # not a bug -- the gap between these two columns IS the defence.
            returned_probs=encoded,
            degradation_level=0,
            suspicion_score=None,
        )
        counts[pop.owner] = counts.get(pop.owner, 0) + 1

    # A sprinkle of failures, so P1's code meets non-200 rows on day one
    # rather than on the demo. Columns 10-14 are null on these.
    for _ in range(int(n_requests * error_rate)):
        c = clients[int(rng.integers(len(clients)))]
        status, code = ERROR_KINDS[int(rng.integers(len(ERROR_KINDS)))]
        store.log(
            run_id=run_id,
            ts=float(t0 + span * rng.random()),
            api_key_id=c["kid"] if status != 401 else "unknown",
            tier=c["pop"].tier,
            ip=c["ip"],
            status_code=status,
            error_code=code,
            latency_ms=float(rng.gamma(1.5, 2.0)),
        )

    store.flush()
    store.close()
    return counts


# ------------------------------------------------------------ report

def summarize(db_path, run_id):
    """Print what P1 is about to be handed, so the hard case is visible."""
    import pandas as pd

    store = LogStore(db_path)
    df = store.read_df(run_id=run_id)
    ok = df[df.status_code == 200].merge(store.owners_df(), on="api_key_id")

    # A cheap stand-in for P1's map: round each point onto a coarse grid.
    pts = np.stack(ok["embedding"].to_numpy())
    grid = np.round(pts[:, :6] / 1.5).astype(np.int64)
    ok = ok.assign(square=[hash(tuple(r)) for r in grid])

    per_key = ok.groupby(["owner", "api_key_id"]).agg(
        reqs=("request_id", "size"), squares=("square", "nunique")).reset_index()
    out = per_key.groupby("owner").agg(
        keys=("api_key_id", "size"),
        med_reqs_per_key=("reqs", "median"),
        med_squares_per_key=("squares", "median")).reset_index()
    grouped = ok.groupby("owner").agg(
        total_reqs=("request_id", "size"),
        squares_as_group=("square", "nunique"),
        ips=("ip", "nunique")).reset_index()
    out = out.merge(grouped, on="owner")
    out["reqs_per_new_square"] = (
        out.total_reqs / out.squares_as_group).round(1)
    store.close()
    return out.sort_values("squares_as_group", ascending=False)


def main():
    ap = argparse.ArgumentParser(
        description="Generate fake logs satisfying SCHEMA.md")
    ap.add_argument("--db", default="data/fake.db")
    ap.add_argument("--run-id", default="cal_seed1")
    ap.add_argument("--n", type=int, default=20000, help="successful requests")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hours", type=float, default=6.0, help="time span covered")
    ap.add_argument("--error-rate", type=float, default=0.005)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    for suffix in ("", "-wal", "-shm"):
        p = Path(a.db + suffix)
        if p.exists():
            p.unlink()

    counts = generate(a.db, a.run_id, a.n, a.seed, a.hours, a.error_rate)
    print(f"wrote {a.n} rows to {a.db}  (run_id={a.run_id}, seed={a.seed})")
    for owner, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {owner:18s} {n:7d}")

    if not a.quiet:
        print("\nwhat P1 is being handed:")
        print(summarize(a.db, a.run_id).to_string(index=False))

    print("\nread it exactly as you will read real logs:")
    print("  from api.logstore import LogStore")
    print(f'  df = LogStore("{a.db}").read_df(run_id="{a.run_id}")')


if __name__ == "__main__":
    main()