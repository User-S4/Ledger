"""
The front door, faked.

Today there is no API, so LocalService loads ResNet-20 into memory and answers
questions itself. It deliberately answers in the SHAPE the real endpoint will
use, borrowed from SCHEMA.md:

    {"model_probs":       [10 floats],   what the model believed
     "returned_probs":    [10 floats],   what the customer actually got
     "degradation_level": 0}             0 until Stage 7 exists

On Day 2, swap LocalService for RemoteService in one line of run_stage3.py.
Everything downstream -- collecting, training, measuring -- never learns which
one it was talking to. That is the whole point of this file.

If P3's real contract turns out different, THIS is the only file that changes.
"""

import numpy as np
import torch

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 repos ship two different normalisation recipes and picking the wrong
# one quietly costs the victim a few points of accuracy. We don't guess --
# calibrate() below tries both and keeps whichever scores higher.
NORM_RECIPES = {
    "dataset_stats": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "common_repo":   ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


class LocalService:
    """ResNet-20 in memory, pretending to be P3's /predict endpoint."""

    def __init__(self, arch="cifar10_resnet20", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", arch, pretrained=True
        ).to(self.device).eval()
        self.norm = "dataset_stats"
        self.degradation_level = 0   # Stage 7 raises this
        self.request_count = 0

    # -- the bit that turns photos into something the model accepts ------

    def _prepare(self, images_uint8):
        mean, std = NORM_RECIPES[self.norm]
        x = torch.from_numpy(np.asarray(images_uint8, dtype=np.uint8)).float().div_(255.0)
        x = x.permute(0, 3, 1, 2)
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
        return (x - m) / s

    def calibrate(self, images, true_labels, verbose=True):
        """Check the victim is healthy BEFORE robbing it.

        Should land near 0.926. If it doesn't, stop -- you'd be cloning a
        broken model and every number after this would be meaningless.
        """
        best, best_acc = None, -1.0
        for name in NORM_RECIPES:
            self.norm = name
            preds = np.asarray(self.predict(images)["model_probs"]).argmax(axis=1)
            acc = float((preds == np.asarray(true_labels)).mean())
            if verbose:
                print(f"    recipe '{name}': victim accuracy {acc:.4f}")
            if acc > best_acc:
                best, best_acc = name, acc
        self.norm = best
        if verbose:
            print(f"    -> using '{best}' ({best_acc:.4f}); expected about 0.926")
        return best, best_acc

    # -- the endpoint ----------------------------------------------------

    @torch.no_grad()
    def predict(self, images_uint8, batch_size=512):
        """Send photos, get answers. Same return shape as the real endpoint."""
        chunks = []
        for i in range(0, len(images_uint8), batch_size):
            batch = self._prepare(images_uint8[i:i + batch_size]).to(self.device)
            chunks.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        believed = np.concatenate(chunks).astype(np.float32)
        self.request_count += len(believed)
        return {
            "model_probs": believed,
            "returned_probs": _degrade(believed, self.degradation_level),
            "degradation_level": self.degradation_level,
        }


class RemoteService:
    """The same door, once P3's FastAPI service is standing.

    Field names below are a guess from SCHEMA.md. Confirm them with P3 and
    correct them here -- nowhere else in the repo needs to know.
    """

    def __init__(self, base_url, api_key, batch_size=64, timeout=30):
        import requests
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.request_count = 0

    def predict(self, images_uint8, batch_size=None):
        bs = batch_size or self.batch_size
        got, believed, level = [], [], 0
        for i in range(0, len(images_uint8), bs):
            chunk = np.asarray(images_uint8[i:i + bs], dtype=np.uint8)
            r = self._requests.post(
                f"{self.base_url}/predict",
                headers={"X-API-Key": self.api_key},
                json={"images": chunk.tolist()},
                timeout=self.timeout,
            )
            r.raise_for_status()
            body = r.json()
            got.append(np.asarray(body["returned_probs"], dtype=np.float32))
            believed.append(np.asarray(
                body.get("model_probs", body["returned_probs"]), dtype=np.float32))
            level = body.get("degradation_level", 0)
        self.request_count += sum(len(c) for c in got)
        return {
            "model_probs": np.concatenate(believed),
            "returned_probs": np.concatenate(got),
            "degradation_level": level,
        }


ROUND_TO = 0.1
TOP_K_AT_LEVEL_2 = 3

def _degrade_one(probs, level):
    """Person 3's api/defense.py degrade(), copied so attack/ runs standalone.
    Keep in sync if they change it."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if level <= 0:
        return p
    winner = int(np.argmax(p))
    if level in (1, 2):
        out = p.copy()
        if level == 2:
            keep = np.argsort(out)[::-1][:TOP_K_AT_LEVEL_2]
            kept = np.zeros_like(out)
            kept[keep] = out[keep]
            out = kept + (1.0 - kept.sum()) / len(out)
        out = np.clip(np.round(out / ROUND_TO) * ROUND_TO, 0.0, 1.0)
        if out.sum() <= 0:
            out = np.zeros_like(p); out[winner] = 1.0
            return out
        out = out / out.sum()
        if int(np.argmax(out)) != winner or out[winner] < out.max():
            out[winner] = out.max() + ROUND_TO
            out = out / out.sum()
        return out
    out = np.zeros_like(p); out[winner] = 1.0
    return out


def _degrade(probs, level):
    if level == 0:
        return probs
    return np.array([_degrade_one(p, level) for p in probs], dtype=np.float32)