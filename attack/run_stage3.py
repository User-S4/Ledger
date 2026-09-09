"""
Stage 3, start to finish. Run it, wait, come back with the number.

    python -m attack.run_stage3 --pool attacker  --budgets 1000 5000
    python -m attack.run_stage3 --pool surrogate --budgets 1000 5000 20000

--pool attacker   CIFAR-10 photos from YOUR range only (see splits.py).
                  The easy theft. Use it to prove the machinery works.
--pool surrogate  CIFAR-100 photos -- things the victim was never trained on.
                  The honest theft. This is the number for the paper.

Results append to eval/results/stage3_fidelity.json, which is the only place
P5 is allowed to read numbers from.
"""

import argparse
import json

from .collect import collect
from .fidelity import describe, evaluate
from .service import LocalService
from .splits import load_exam, load_split, load_surrogate
from .train_clone import train_clone


def get_photos(pool, n, seed):
    if pool == "surrogate":
        return load_surrogate(n, seed=seed)
    return load_split("attacker", n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="attacker", choices=["attacker", "surrogate"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--victim-arch", default="cifar10_resnet20")
    ap.add_argument("--clone-arch", default="cifar10_resnet20")
    ap.add_argument("--degradation", type=int, default=0,
                    help="0 = undefended. Stage 7 uses 1-3.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results", default="eval/results/stage3_fidelity.json")
    args = ap.parse_args()

    print("[1/5] opening the victim")
    victim = LocalService(arch=args.victim_arch)
    victim.degradation_level = args.degradation

    print("[2/5] health check -- is this victim worth stealing?")
    exam_images, exam_labels = load_exam()
    victim.calibrate(exam_images[:2000], exam_labels[:2000])

    summary = []
    for budget in args.budgets:
        print(f"\n=== {budget} queries from the '{args.pool}' pool ===")

        print("[3/5] choosing photos to send")
        photos = get_photos(args.pool, budget, args.seed)

        print("[4/5] querying the victim and writing down the answers")
        answers = collect(
            victim, photos,
            out_npz=f"attack/data/stolen_{args.pool}_{budget}_d{args.degradation}.npz",
        )

        print("[5/5] training the copy on those answers alone")
        clone = train_clone(
            photos, answers, arch=args.clone_arch, epochs=args.epochs,
            temperature=args.temperature, seed=args.seed, norm_name=victim.norm,
        )

        result = evaluate(
            clone, victim, exam_images, exam_labels,
            norm_name=victim.norm, n_queries=budget,
            tag=f"{args.pool}|{args.clone_arch}|degradation{args.degradation}",
            results_path=args.results,
        )
        print("  " + describe(result))
        summary.append(result)

    print("\n--- Stage 3 summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
