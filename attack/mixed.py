"""
mixed.py -- the very smart thief. Distributed, plus junk padding.

distributed.py already beats the per-account guard. This one is aimed at the
NEXT line of defence -- anything that notices "these accounts, taken together,
are asking an oddly purposeful set of questions." The trick: pad the real theft
with junk requests that carry no new information.

The junk here is repeats and near-repeats of images already sent. That choice
is deliberate and it's the argument of the whole project:

  - Against a guard that COUNTS BEHAVIOUR, the junk works: more requests, more
    accounts, noisier pattern, real theft buried inside.
  - Against P1's survey grid (Stage 6), the junk should FAIL: a repeat lands on
    a square of the map that's already filled in, so it adds nothing to the
    tally. Noise fools a counter; it can't fool an information measure.

So mixed.py is the attack that makes Stage 6 look clever. You're building the
thing your own teammate's innovation is designed to defeat.

    python -m attack.mixed --local --budget 20000 --junk-ratio 1.0 --n-keys 400

junk-ratio 1.0 means one junk request per real one (traffic doubles, theft
identical). The clone is trained ONLY on the real answers -- junk is thrown
away before training, because it taught us nothing, which is the point.
"""

import argparse

import numpy as np

from .fidelity import describe, evaluate
from .service import LocalService
from .session import ApiTarget, load_key_secrets, run_session
from .splits import load_exam, load_split, load_surrogate
from .train_clone import train_clone


def photos(pool, n, seed):
    return load_surrogate(n, seed=seed) if pool == "surrogate" \
        else load_split("attacker", n, seed=seed)


def build_padded(real_imgs, junk_ratio, seed):
    """Interleave real photos with junk (repeats of already-sent photos).

    Returns (all_imgs, is_real_mask) so we can send everything but train on
    the real ones only.
    """
    rng = np.random.default_rng(seed)
    n_junk = int(len(real_imgs) * junk_ratio)
    # Junk = resampled copies of images we've already sent. Same square of the
    # map, no new information -- exactly what should slip past a behaviour
    # counter and get caught by an information measure.
    junk_idx = rng.integers(0, len(real_imgs), size=n_junk)
    junk_imgs = real_imgs[junk_idx]

    all_imgs = np.concatenate([real_imgs, junk_imgs], axis=0)
    is_real = np.concatenate([np.ones(len(real_imgs), bool),
                              np.zeros(n_junk, bool)])
    order = rng.permutation(len(all_imgs))      # shuffle so junk is interspersed
    return all_imgs[order], is_real[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pool", default="attacker", choices=["attacker", "surrogate"])
    ap.add_argument("--budget", type=int, default=20000, help="REAL theft queries")
    ap.add_argument("--junk-ratio", type=float, default=1.0)
    ap.add_argument("--n-keys", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results", default="eval/results/stage5_mixed.json")
    a = ap.parse_args()

    real = photos(a.pool, a.budget, a.seed)
    all_imgs, is_real = build_padded(real, a.junk_ratio, a.seed)
    print(f"    {len(real)} real + {len(all_imgs) - len(real)} junk "
          f"= {len(all_imgs)} total requests sent")

    victim = LocalService()
    exam_i, exam_l = load_exam()
    victim.calibrate(exam_i[:2000], exam_l[:2000])
    norm = victim.norm

    keys = ([f"k_{10000 + j}" for j in range(a.n_keys)] if a.local
            else load_key_secrets(a.keys))
    kw = dict(key_for=lambda i: keys[i % len(keys)],
              out_npz=f"attack/data/mixed_{a.pool}_{a.budget}.npz")
    if a.local:
        answers, keys_used = run_session(all_imgs, service=victim, **kw)
    else:
        answers, keys_used = run_session(all_imgs, api=ApiTarget(url=a.url), **kw)

    # Throw the junk away -- it taught us nothing. Train on the real theft only.
    real_imgs = all_imgs[is_real]
    real_answers = answers[is_real]

    clone = train_clone(real_imgs, real_answers, epochs=a.epochs, seed=a.seed,
                        norm_name=norm)
    res = evaluate(clone, victim, exam_i, exam_l, norm_name=norm,
                   n_queries=a.budget,
                   tag=f"mixed|{a.pool}|junk{a.junk_ratio}|{len(set(keys_used))}keys",
                   results_path=a.results)
    print("  " + describe(res))
    print(f"    fidelity ~unchanged despite {len(all_imgs) - len(real)} junk "
          f"requests -- noise doesn't help the thief, and shouldn't fool a good guard.")


if __name__ == "__main__":
    main()
