"""
Stage 7, settled properly -- offline, from answers we already stole.

No API, no re-querying. We have 20,000 (photo, answer) pairs saved in
attack/data/stolen_surrogate_20000_d0.npz. Every defence question is now
just "apply a transform to those answers, retrain, measure fidelity."

Three experiments, each answering one open question:

  cutoff     Train on the first N answers only. This is the TRUE ceiling of
             early detection: what does the thief get if the API stops
             serving after N queries? Not degradation -- refusal.

  combined   First N answers clean, the rest label-only. This is the
             "detect at 1k, then degrade" story. If fidelity lands near the
             undefended 83%, early detection + degradation is dead, because
             a correct label is all the clone needed.

  smoothing  Label smoothing at several strengths:  p' = (1-a)p + a/10.
             Path B's proposal. Preserves the winner (so honest customers
             are unharmed) but flattens the soft targets. Tests whether
             flattening does what rounding couldn't.

Usage:
    python -m attack.degrade_experiments --which cutoff
    python -m attack.degrade_experiments --which combined
    python -m attack.degrade_experiments --which smoothing
    python -m attack.degrade_experiments --which all
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .fidelity import evaluate
from .service import LocalService, _degrade_one
from .splits import load_exam
from .train_clone import train_clone

DEFAULT_NPZ = "attack/data/stolen_surrogate_20000_d0.npz"


def load_stolen(path):
    d = np.load(path)
    return d["images"], d["answers"].astype(np.float32)


# ---------------------------------------------------------------- transforms

def label_only(answers):
    """Level 3: 1.0 on the winner, 0 elsewhere. The most aggressive
    degradation that still returns a CORRECT answer."""
    out = np.zeros_like(answers)
    out[np.arange(len(answers)), answers.argmax(axis=1)] = 1.0
    return out


def label_smooth(answers, alpha):
    """p' = (1-a)p + a/K. Winner preserved, soft targets flattened.
    alpha=0 is untouched; alpha=1 is pure uniform (no information at all)."""
    k = answers.shape[1]
    return ((1.0 - alpha) * answers + alpha / k).astype(np.float32)


def label_swap_top2(answers):
    """Targeted boundary poisoning: swap top-1 winner with top-2 runner-up."""
    out = answers.copy()
    for i in range(len(out)):
        order = np.argsort(out[i])[::-1]
        top1, top2 = order[0], order[1]
        out[i, top1], out[i, top2] = out[i, top2], out[i, top1]
    return out


def label_poison_hard(answers):
    """Hard targeted poisoning: 1.0 on runner-up, 0 elsewhere."""
    out = np.zeros_like(answers)
    for i in range(len(out)):
        runner_up = np.argsort(answers[i])[::-1][1]
        out[i, runner_up] = 1.0
    return out


# ---------------------------------------------------------------- experiments

def run_one(images, answers, victim, exam_i, exam_l, norm, tag, epochs, seed,
            results_path, n_queries=None):
    clone = train_clone(images, answers, epochs=epochs, seed=seed, norm_name=norm)
    res = evaluate(clone, victim, exam_i, exam_l, norm_name=norm,
                   n_queries=n_queries if n_queries is not None else len(images),
                   tag=tag, results_path=results_path)
    print(f"  {tag}: fidelity {res['fidelity']:.4f}  "
          f"clone_acc {res['clone_accuracy']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--which", default="all",
                    choices=["cutoff", "combined", "smoothing", "poison", "all"])
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results", default="eval/results/stage7_experiments.json")
    a = ap.parse_args()

    if not Path(a.npz).exists():
        raise SystemExit(
            f"{a.npz} not found. It's produced by:\n"
            f"  python -m attack.run_stage3 --pool surrogate --budgets 20000 "
            f"--degradation 0")

    print(f"loading stolen pairs from {a.npz}")
    images, answers = load_stolen(a.npz)
    print(f"  {len(images)} photo/answer pairs")

    victim = LocalService()
    exam_i, exam_l = load_exam()
    victim.calibrate(exam_i[:2000], exam_l[:2000])
    norm = victim.norm
    common = dict(victim=victim, exam_i=exam_i, exam_l=exam_l, norm=norm,
                  epochs=a.epochs, seed=a.seed, results_path=a.results)

    # Baseline, so every comparison is against a number from THIS run rather
    # than one from a previous session with different noise.
    print("\n=== baseline: undefended, full budget ===")
    run_one(images, answers, tag="baseline|undefended|20000", **common)

    if a.which in ("cutoff", "all"):
        print("\n=== cutoff: API refuses service after N queries ===")
        print("    (the true ceiling of early detection -- refusal, not blurring)")
        for n in a.cutoffs:
            run_one(images[:n], answers[:n], tag=f"cutoff|clean{n}", **common)

    if a.which in ("combined", "all"):
        print("\n=== combined: N clean answers, then label-only for the rest ===")
        print("    (detect early, then degrade -- does it beat plain degradation?)")
        for n in a.cutoffs:
            mixed = answers.copy()
            mixed[n:] = label_only(answers[n:])
            run_one(images, mixed, tag=f"combined|clean{n}+labelonly", **common)

    if a.which in ("smoothing", "all"):
        print("\n=== smoothing: flatten the soft targets, keep the winner ===")
        print("    (Path B's proposal -- does flattening beat rounding?)")
        for alpha in a.alphas:
            sm = label_smooth(answers, alpha)
            # Sanity: the winner must be unchanged, or we're not testing
            # a defensible defence -- we're testing lying to customers.
            same = (sm.argmax(axis=1) == answers.argmax(axis=1)).mean()
            print(f"    alpha={alpha}: top-1 preserved on {same:.1%} of answers")
            run_one(images, sm, tag=f"smoothing|alpha{alpha}", **common)

    if a.which in ("poison", "all"):
        print("\n=== poison: targeted misinformation / label poisoning after N queries ===")
        print("    (first N clean, then suspicious queries receive swapped top-2 or runner-up)")
        for n in a.cutoffs:
            # 1. Boundary swap: swap top 1 and top 2 for queries after cutoff n
            swapped = answers.copy()
            swapped[n:] = label_swap_top2(answers[n:])
            run_one(images, swapped, tag=f"poison|clean{n}+swap_top2", **common)

            # 2. Hard poison: 1.0 on runner-up for queries after cutoff n
            hard_poisoned = answers.copy()
            hard_poisoned[n:] = label_poison_hard(answers[n:])
            run_one(images, hard_poisoned, tag=f"poison|clean{n}+hard_runner_up", **common)

    print(f"\nall results appended to {a.results}")
    print(json.dumps(json.loads(Path(a.results).read_text())[-8:], indent=2))


if __name__ == "__main__":
    main()
