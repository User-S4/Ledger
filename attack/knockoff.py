"""
knockoff.py -- the clumsy thief. One account, all the requests.

This is the Stage 3 baseline, restated as an attack the ordinary guard is
SUPPOSED to catch. Everything goes out under a single key, so per-account
signals -- high rate, repetitive queries -- light up immediately. That's the
point: it's the "before" picture that makes the distributed attack's escape
(distributed.py) look impressive by contrast.

    python -m attack.knockoff --keys data/attacker_keys.json --budget 5000
    python -m attack.knockoff --local --budget 5000        # no API needed

Underneath, the theft is identical to the other two files. Only key_for changes.
"""

import argparse

from .fidelity import describe, evaluate
from .session import ApiTarget, load_key_secrets, run_session
from .splits import load_exam, load_split, load_surrogate
from .train_clone import train_clone


def photos(pool, n, seed):
    return load_surrogate(n, seed=seed) if pool == "surrogate" \
        else load_split("attacker", n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", help="JSON key file from P3")
    ap.add_argument("--local", action="store_true", help="use in-memory victim")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pool", default="attacker", choices=["attacker", "surrogate"])
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="delay (seconds) between requests to simulate rate evasion")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="random jitter (+/- seconds) applied to delay")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results", default="eval/results/stage5_knockoff.json")
    a = ap.parse_args()

    imgs = photos(a.pool, a.budget, a.seed)

    timing_info: dict = {}

    if a.local:
        from .service import LocalService
        victim = LocalService()
        exam_i, exam_l = load_exam()
        victim.calibrate(exam_i[:2000], exam_l[:2000])
        norm = victim.norm
        answers, _ = run_session(imgs, key_for=lambda i: "k_attacker_single",
                                 service=victim,
                                 delay_s=a.delay, jitter_s=a.jitter, timing_out=timing_info,
                                 out_npz=f"attack/data/knockoff_{a.pool}_{a.budget}.npz")
    else:
        secrets = load_key_secrets(a.keys)
        one = secrets[0]                      # the clumsy thief uses ONE key
        api = ApiTarget(url=a.url)
        answers, _ = run_session(imgs, key_for=lambda i: one, api=api,
                                 delay_s=a.delay, jitter_s=a.jitter, timing_out=timing_info,
                                 out_npz=f"attack/data/knockoff_{a.pool}_{a.budget}.npz")
        from .service import LocalService
        victim = LocalService(); exam_i, exam_l = load_exam()
        victim.calibrate(exam_i[:2000], exam_l[:2000]); norm = victim.norm

    elapsed_s = timing_info.get("elapsed_s", 0.0)
    print(f"    knockoff completed {a.budget} requests in {elapsed_s:.1f}s")

    clone = train_clone(imgs, answers, epochs=a.epochs, seed=a.seed, norm_name=norm)
    exam_i, exam_l = load_exam()
    res = evaluate(clone, victim, exam_i, exam_l, norm_name=norm,
                   n_queries=a.budget, tag=f"knockoff|{a.pool}",
                   results_path=a.results)
    print("  " + describe(res))


if __name__ == "__main__":
    main()
