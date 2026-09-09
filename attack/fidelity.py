"""
The number the whole project rests on.

Two things get measured and they are not the same:

  fidelity  -- how often the clone gives the SAME answer as the victim,
               right or wrong. This is the theft metric. A clone that copies
               the victim's mistakes is more damning than one that doesn't.

  accuracy  -- how often the clone is correct against real labels. This is
               the "is it useful to the thief" metric.

Always report fidelity first. It's the one that says a specific model was
copied, rather than that someone trained a decent classifier.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .service import NORM_RECIPES


@torch.no_grad()
def _predict(model, images, norm_name="dataset_stats", batch_size=512, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    mean, std = NORM_RECIPES[norm_name]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.asarray(images[i:i + batch_size])).float().div_(255.0)
        x = ((x.permute(0, 3, 1, 2) - mean_t) / std_t).to(device)
        out.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def evaluate(clone, service, exam_images, exam_labels,
             norm_name="dataset_stats", n_queries=None, tag=None,
             results_path=None):
    """Ask the victim and the clone the same exam questions and compare.

    We use `model_probs` here, not `returned_probs`: fidelity asks whether the
    copy matches what the victim REALLY thinks, even in Stage 7 when customers
    are being fed degraded answers.
    """
    holdout_images, holdout_labels = exam_images, exam_labels
    victim_pred = np.asarray(
        service.predict(holdout_images)["model_probs"]).argmax(axis=1)
    clone_pred = _predict(clone, holdout_images, norm_name=norm_name)
    labels = np.asarray(holdout_labels)

    result = {
        "tag": tag,
        "n_queries": n_queries,
        "fidelity": float((clone_pred == victim_pred).mean()),
        "clone_accuracy": float((clone_pred == labels).mean()),
        "victim_accuracy": float((victim_pred == labels).mean()),
        "holdout_size": int(len(labels)),
    }
    result["fidelity_gap"] = result["victim_accuracy"] - result["clone_accuracy"]

    if results_path:
        p = Path(results_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(p.read_text()) if p.exists() else []
        existing.append(result)
        p.write_text(json.dumps(existing, indent=2))

    return result


def describe(result):
    """The sentence you owe the team by end of Day 1."""
    return (f"With {result['n_queries']} queries, our copy agrees with the "
            f"original {result['fidelity'] * 10:.1f} times out of 10 "
            f"(clone accuracy {result['clone_accuracy']:.1%} vs victim "
            f"{result['victim_accuracy']:.1%}).")
