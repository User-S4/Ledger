"""
distributed.py -- the smart thief. The same theft, split across many accounts.

This is your Stage 5 headline. The robbery is byte-for-byte identical to
knockoff.py -- same photos, same answers, same clone, so fidelity barely moves.
The ONLY change is key_for: instead of one key, requests round-robin across
hundreds. Each individual account now sends few enough requests, slowly enough,
that the per-account guard sees nothing wrong. The theft walks out the door.

    python -m attack.distributed --keys data/attacker_keys.json --budget 20000
    python -m attack.distributed --local --budget 20000 --n-keys 400

What you hand P1: the trace this produces (via P3's live API, which logs it)
and the fidelity number, which should match knockoff.py at the same budget.
"Same theft, same result, but your guard caught the first and missed this one."
"""

import argparse

from .fidelity import describe, evaluate
from .service import LocalService
from .session import ApiTarget, load_key_secrets, run_session
from .splits import load_exam, load_split, load_surrogate
from .train_clone import train_clone


def _ip_for_key(key_index):
    """One stable IP per account. Key 0 -> 198.51.0.0, key 1 -> 198.51.0.1 ...

    198.51.100.0/24 is a documentation range (TEST-NET-2), so these can't
    collide with a real address anyone might mistake them for. We spread
    across the wider 198.51.x.x space only because 400 keys need more than
    256 slots; the exact numbers don't matter, only that each key keeps one.
    """
    return f"198.51.{key_index // 256}.{key_index % 256}"


def photos(pool, n, seed):
    return load_surrogate(n, seed=seed) if pool == "surrogate" \
        else load_split("attacker", n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", help="JSON key file from P3 (many keys)")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pool", default="attacker", choices=["attacker", "surrogate"])
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--n-keys", type=int, default=400,
                    help="how many accounts to spread across (local mode)")
    ap.add_argument("--spread-ip", action="store_true",
                    help="also give each key its own IP (hardest case for the guard)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="delay (seconds) between requests to simulate rate evasion")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="random jitter (+/- seconds) applied to delay")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results", default="eval/results/stage5_distributed.json")
    a = ap.parse_args()

    imgs = photos(a.pool, a.budget, a.seed)

    victim = LocalService()
    exam_i, exam_l = load_exam()
    victim.calibrate(exam_i[:2000], exam_l[:2000])
    norm = victim.norm

    timing_info: dict = {}

    if a.local:
        keys = [f"k_{10000 + j}" for j in range(a.n_keys)]
        # Each KEY gets one stable IP -- 400 accounts on 400 machines, reused
        # across the run. Keying off the request index instead would give one
        # IP per request, which no real attacker produces and which is itself
        # a giveaway. The IP must map to the account, not the moment.
        ip_for = (lambda i: _ip_for_key(i % len(keys))) if a.spread_ip else None
        answers, keys_used = run_session(
            imgs, key_for=lambda i: keys[i % len(keys)], service=victim,
            ip_for=ip_for,
            delay_s=a.delay, jitter_s=a.jitter, timing_out=timing_info,
            out_npz=f"attack/data/distributed_{a.pool}_{a.budget}.npz")
    else:
        keys = load_key_secrets(a.keys)
        api = ApiTarget(url=a.url)
        ip_for = (lambda i: _ip_for_key(i % len(keys))) if a.spread_ip else None
        answers, keys_used = run_session(
            imgs, key_for=lambda i: keys[i % len(keys)], api=api, ip_for=ip_for,
            delay_s=a.delay, jitter_s=a.jitter, timing_out=timing_info,
            out_npz=f"attack/data/distributed_{a.pool}_{a.budget}.npz")

    elapsed_s = timing_info.get("elapsed_s", 0.0)
    print(f"    spread {a.budget} requests across {len(set(keys_used))} keys "
          f"(~{a.budget // max(1, len(set(keys_used)))} each) in {elapsed_s:.1f}s")

    if a.epochs > 0:
        clone = train_clone(imgs, answers, epochs=a.epochs, seed=a.seed, norm_name=norm)
        res = evaluate(clone, victim, exam_i, exam_l, norm_name=norm,
                       n_queries=a.budget, tag=f"distributed|{a.pool}|{len(set(keys_used))}keys",
                       results_path=a.results)
        print("  " + describe(res))
        print("    ^ compare this to knockoff.py at the same budget: near-identical "
              "fidelity, but every account looked innocent.")
    else:
        print("    [+] Clone training skipped (--epochs 0). Attack queries delivered successfully.")



if __name__ == "__main__":
    main()