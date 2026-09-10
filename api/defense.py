"""Stage 7: the fightback.

When suspicion crosses a threshold, start rounding off the answers we return.

The one rule everything else follows
    NEVER change the top-1 label. Degrade how much information leaves, not
    whether the answer is right. We will sometimes be wrong about who is
    hostile -- 27% false alarms on our own fake data -- and a service that
    starts lying to customers it wrongly suspects is indefensible. A service
    that returns a correct but less detailed answer is merely stingy, which
    is what every free tier already is.

Why this works against extraction
    A thief learns from the model's hesitation, not its answer. "Cat 0.31,
    dog 0.29" says the two classes nearly touch at this point -- that is a
    coordinate on the decision boundary, and boundaries are what a clone is
    really copying. Round it to "cat 0.3, dog 0.3" and that coordinate blurs.
    Return the label alone and it vanishes. Meanwhile a customer who just
    wanted to know it was a cat is unaffected.

The four levels
    0  untouched
    1  probabilities rounded to the nearest 0.1
    2  top-3 kept AND rounded -- the levels stack
    3  label only -- 1.0 on the winner, 0 elsewhere

    Level 1 rounds coarsely on purpose. Rounding to 2 decimals leaves
    "0.31 vs 0.29" intact, and that 0.02 gap is exactly the boundary
    coordinate worth stealing. Rounding to 0.1 turns both into 0.3 and the
    gap disappears. Level 2 keeps the rounding rather than replacing it --
    a shorter list of exact numbers would leak the same gap.

Stickiness
    Suspicion fluctuates. Without hysteresis an account flips between levels
    request by request, and the pattern of flipping is itself a signal a
    thief can read to find the threshold. Levels rise immediately and fall
    only after the score has stayed low for a while.
"""

from __future__ import annotations

import threading
import time

import numpy as np

DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9)   # -> levels 1, 2, 3
ROUND_TO = 0.1                         # coarseness at levels 1 and 2
DEFAULT_COOLDOWN_S = 120.0
TOP_K_AT_LEVEL_2 = 3


def level_for_score(score: float | None, thresholds=DEFAULT_THRESHOLDS) -> int:
    """Map a suspicion score to a degradation level."""
    if score is None:
        return 0
    for level, cut in zip((3, 2, 1), reversed(thresholds)):
        if score >= cut:
            return level
    return 0


def degrade(probs, level: int) -> list[float]:
    """Apply a degradation level to a probability vector.

    Always returns something that still sums to 1 and still has the same
    winner. A client cannot tell a degraded answer from a confident one --
    which matters, because an attacker who can detect degradation knows
    exactly when to rotate to a fresh account.
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if level <= 0:
        return [float(v) for v in p]

    winner = int(np.argmax(p))

    if level in (1, 2):
        out = p.copy()
        if level == 2:
            # Collapse everything outside the top 3, spreading the discarded
            # mass evenly so the vector stays valid and the dropped classes
            # carry no information about their true order.
            keep = np.argsort(out)[::-1][:TOP_K_AT_LEVEL_2]
            kept = np.zeros_like(out)
            kept[keep] = out[keep]
            out = kept + (1.0 - kept.sum()) / len(out)

        out = np.clip(np.round(out / ROUND_TO) * ROUND_TO, 0.0, 1.0)

        if out.sum() <= 0:
            out = np.zeros_like(p)
            out[winner] = 1.0
            return [float(v) for v in out]

        out = out / out.sum()

        # Rounding can leave the true winner tied with, or below, another
        # class -- which would change the label. Lift it clear and
        # renormalise. Never subtract the shortfall from the winner: on a
        # vector that rounds UP overall that drives it negative, which is
        # how this bug was found.
        if int(np.argmax(out)) != winner or out[winner] < out.max():
            out[winner] = out.max() + ROUND_TO
            out = out / out.sum()

        return [float(v) for v in out]

    out = np.zeros_like(p)
    out[winner] = 1.0
    return [float(v) for v in out]


class DefensePolicy:
    """Decides a degradation level per account, with hysteresis.

    Thread-safe: several requests are in flight at once and they share this.

    The scorer is injected, so Stage 7 could be built and tested before P1's
    tier-3 scoring existed. Swapping in the real one is a single argument.
    """

    def __init__(self, scorer=None, thresholds=DEFAULT_THRESHOLDS,
                 cooldown_s: float = DEFAULT_COOLDOWN_S, enabled: bool = False):
        self.scorer = scorer
        self.thresholds = tuple(thresholds)
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self._lock = threading.Lock()
        self._level: dict[str, int] = {}
        self._since: dict[str, float] = {}

    def score_for(self, api_key_id: str, now: float | None = None) -> float | None:
        if self.scorer is None:
            return None
        return self.scorer(api_key_id)

    def level_for(self, api_key_id: str, score: float | None,
                  now: float | None = None) -> int:
        """Current level for this account, applying stickiness."""
        if not self.enabled:
            return 0
        now = time.time() if now is None else now
        target = level_for_score(score, self.thresholds)

        with self._lock:
            current = self._level.get(api_key_id, 0)
            if target > current:
                self._level[api_key_id] = target
                self._since[api_key_id] = now
                return target
            if target < current:
                # Only step down once the score has stayed lower than the
                # current level for a full cooldown.
                if now - self._since.get(api_key_id, now) >= self.cooldown_s:
                    self._level[api_key_id] = target
                    self._since[api_key_id] = now
                    return target
                return current
            self._since.setdefault(api_key_id, now)
            return current

    def apply(self, api_key_id: str, probs, now: float | None = None):
        """The single call /predict makes. Returns (returned_probs, level, score).

        `probs` is what the model believed -- column 13. The first element of
        the return is what the customer receives -- column 14. Keeping both
        in the log is what makes the defence measurable rather than claimed.
        """
        score = self.score_for(api_key_id, now)
        level = self.level_for(api_key_id, score, now)
        return degrade(probs, level), level, score

    def reset(self) -> None:
        with self._lock:
            self._level.clear()
            self._since.clear()


def build_policy(cfg: dict, scorer=None) -> DefensePolicy:
    """Construct from the `defense` block of config.yaml."""
    d = (cfg or {}).get("defense", {}) or {}
    return DefensePolicy(
        scorer=scorer,
        thresholds=tuple(d.get("thresholds", DEFAULT_THRESHOLDS)),
        cooldown_s=float(d.get("cooldown_s", DEFAULT_COOLDOWN_S)),
        enabled=bool(d.get("enabled", False)),
    )


# ------------------------------------------------------------------ demo

def _main() -> None:
    """Show what a thief actually loses at each level.

    python -m api.defense
    """
    probs = [0.021, 0.014, 0.313, 0.287, 0.043, 0.178, 0.062, 0.031, 0.037, 0.014]
    names = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]

    print("A genuinely uncertain query -- bird 0.31 vs cat 0.29.")
    print("That near-tie IS a coordinate on the decision boundary, which is")
    print("what a clone is really copying.\n")

    for level in range(4):
        out = degrade(probs, level)
        top = int(np.argmax(out))
        label = ["untouched", "rounded", "top-3 only", "label only"][level]
        print(f"level {level} ({label}):")
        print("  " + "  ".join(f"{n[:4]} {v:.2f}" for n, v in zip(names, out)))
        print(f"  winner still {names[top]}, sums to {sum(out):.4f}")
        margin = sorted(out, reverse=True)
        print(f"  top-1 minus top-2: {margin[0] - margin[1]:.3f}\n")

    print("The label is correct at every level. What disappears is how close")
    print("the runner-up was -- which is the part worth stealing.")


if __name__ == "__main__":
    _main()