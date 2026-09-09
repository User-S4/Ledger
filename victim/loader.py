"""The victim model. Pretrained, off the shelf, never trained by us.

ResNet-20 trained on CIFAR-10, from chenyaofo/pytorch-cifar-models. Small
(0.27M parameters), runs fine on a laptop CPU, about 92% accurate. Good
enough that a stolen copy is visibly worth stealing, small enough that
P2's clone trains in minutes rather than hours.

Why it returns ten numbers, not one label
    A real commercial API returns the full spread of confidence, because
    customers want to know how sure the model is. So do we. That spread is
    also exactly what makes extraction cheap -- a thief learns the model's
    hesitation, not just its answer. Stage 7's fightback works by taking
    that richness away from suspicious traffic. An API that only ever said
    "cat" would make the attack weak and the demo dishonest.

Two backends
    torch  the real model
    stub   a deterministic fake with identical shapes and no torch

The stub exists so P1, P2, P4 and P5 can exercise the whole pipeline while
torch is still downloading. Same input shapes, same output shapes, stable
across runs. Never let a number from the stub reach the submission --
backend_name() tells you which one you got, and /health reports it.
"""

from __future__ import annotations

import hashlib
import io
import threading

import numpy as np

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = len(CLASSES)      # columns 13 and 14 width
FEATURE_DIM = 64                # resnet20's penultimate layer, before PCA
IMAGE_SIZE = 32

# Normalisation constants that ship with these pretrained weights. Changing
# them silently degrades accuracy -- do not "clean these up".
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)

_lock = threading.Lock()
_model = None
_backend: str | None = None


# ------------------------------------------------------------------ input

def decode_image(raw: bytes) -> np.ndarray:
    """Raw upload bytes -> float32 array (3, 32, 32), values 0..1.

    Anything unreadable raises ValueError, which the API turns into a clean
    400. This is the front line of the Stage 9 robustness work: a truncated
    PNG, a PDF renamed to .png, or 3MB of zeroes all land here.

    Images that aren't 32x32 are resized rather than rejected. Honest
    customers send whatever they have; making that their problem would be a
    false positive dressed up as validation.
    """
    from PIL import Image

    if not raw:
        raise ValueError("empty image")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()          # Image.open is lazy; this is what actually decodes
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a bad image
        raise ValueError("could not decode image") from exc

    img = img.convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0     # (32, 32, 3)
    return np.transpose(arr, (2, 0, 1)).copy()          # (3, 32, 32)


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - _MEAN) / _STD


def image_hash(raw: bytes) -> str:
    """Column 10. Identical images produce identical hashes, which is how
    junk padding traffic gives itself away."""
    return hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------ backends

class _TorchVictim:
    """The real model."""

    def __init__(self):
        import torch

        self.torch = torch
        self.net = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=True,
            trust_repo=True,
        )
        self.net.eval()
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)   # keeps latency steady under concurrent load

        self._features: list = []
        # The penultimate representation is whatever feeds the final linear
        # layer. A forward hook grabs it without modifying the model.
        self.net.fc.register_forward_hook(
            lambda _m, inp, _out: self._features.append(inp[0].detach().cpu().numpy())
        )

    def forward(self, batch: np.ndarray):
        t = self.torch.from_numpy(np.ascontiguousarray(batch))
        self._features.clear()
        logits = self.net(t)
        probs = self.torch.softmax(logits, dim=1).numpy().astype(np.float32)
        feats = self._features[-1].astype(np.float32)
        return probs, feats


class _StubVictim:
    """Deterministic pseudo-model. Same shapes, no torch, no meaning.

    A fixed random projection of the pixels: identical inputs always give
    identical outputs, and similar inputs give similar outputs. Enough to
    exercise logging, the map, and the traffic tools. Not enough to report.
    """

    def __init__(self, seed: int = 1337):
        rng = np.random.default_rng(seed)
        n_px = 3 * IMAGE_SIZE * IMAGE_SIZE
        self.W1 = rng.normal(0, 0.05, (n_px, FEATURE_DIM)).astype(np.float32)
        self.W2 = rng.normal(0, 0.30, (FEATURE_DIM, NUM_CLASSES)).astype(np.float32)

    def forward(self, batch: np.ndarray):
        feats = np.tanh(batch.reshape(len(batch), -1) @ self.W1)
        logits = feats @ self.W2
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32), feats


def load_victim(backend: str = "auto"):
    """Load once, reuse forever. backend: auto | torch | stub.

    'auto' tries torch and falls back to the stub with a loud message.
    'torch' refuses to fall back -- use it when a real number depends on it.
    """
    global _model, _backend
    with _lock:
        if _model is not None:
            return _model
        if backend == "stub":
            _model, _backend = _StubVictim(), "stub"
            return _model
        try:
            _model, _backend = _TorchVictim(), "torch"
        except Exception as exc:  # noqa: BLE001
            if backend == "torch":
                raise
            print(f"[victim] torch unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to STUB -- results are not reportable")
            _model, _backend = _StubVictim(), "stub"
        return _model


def backend_name() -> str:
    """Which backend is loaded. /health reports this; never report a number
    from a run where this says 'stub'."""
    return _backend or "unloaded"


def reset() -> None:
    """Drop the loaded model. Tests only."""
    global _model, _backend
    with _lock:
        _model, _backend = None, None


# ------------------------------------------------------------------ scoring

def predict(raw_batch, backend: str = "auto"):
    """Raw upload bytes -> (probs (N,10), features (N,64)).

    probs go to columns 13 and 14. features go to victim/embed.py, which
    squashes them to the 32 numbers column 12 expects.

    Raises ValueError on any unreadable image.
    """
    model = load_victim(backend)
    x = np.stack([normalize(decode_image(raw)) for raw in raw_batch])
    return model.forward(x)


def summarize(probs) -> dict:
    """Confidence, margin and uncertainty -- all arithmetic on column 13.

    We do NOT store these as separate columns. They are derived on demand so
    they cannot drift out of sync with the probabilities they came from.
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    order = np.argsort(p)[::-1]
    top1 = float(p[order[0]])
    top2 = float(p[order[1]]) if p.size > 1 else 0.0
    return {
        "label": int(order[0]),
        "label_name": CLASSES[int(order[0])],
        "confidence": top1,
        "margin": top1 - top2,     # how close the runner-up was
        "entropy": float(-(p * np.log(np.clip(p, 1e-12, None))).sum()),
    }


# ------------------------------------------------------------------ self-test

def _main() -> None:
    """python -m victim.loader            try the real model
       python -m victim.loader --stub     no torch needed
    """
    import argparse
    import time

    from PIL import Image

    ap = argparse.ArgumentParser(description="Victim model self-test")
    ap.add_argument("--stub", action="store_true", help="skip torch")
    ap.add_argument("--image", help="path to an image to classify")
    a = ap.parse_args()

    backend = "stub" if a.stub else "auto"
    t0 = time.perf_counter()
    load_victim(backend)
    print(f"backend: {backend_name()}   (loaded in {time.perf_counter()-t0:.1f}s)")

    if a.image:
        with open(a.image, "rb") as f:
            raw = f.read()
    else:
        rng = np.random.default_rng(0)
        buf = io.BytesIO()
        Image.fromarray(
            rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        ).save(buf, "PNG")
        raw = buf.getvalue()
        print("no --image given, using a random test image")

    t0 = time.perf_counter()
    probs, feats = predict([raw], backend)
    ms = (time.perf_counter() - t0) * 1000
    s = summarize(probs[0])

    print(f"\nprediction: {s['label_name']}  ({s['confidence']:.1%} confident)")
    print(f"runner-up was {s['margin']:.1%} behind")
    print(f"features: {feats.shape}   probabilities: {probs.shape}   "
          f"sum={probs[0].sum():.6f}   {ms:.1f}ms")
    print("\nfull spread -- this is what a thief collects:")
    for name, p in sorted(zip(CLASSES, probs[0]), key=lambda kv: -kv[1]):
        print(f"  {name:11s} {p:6.2%}  {'#' * int(p * 40)}")

    print("\nmalformed input:")
    for name, blob in [("empty", b""), ("text", b"hello world"),
                       ("truncated png", raw[:30]), ("random bytes", bytes(range(256)))]:
        try:
            decode_image(blob)
            print(f"  {name:14s} ACCEPTED -- that is a bug")
        except ValueError as exc:
            print(f"  {name:14s} rejected: {exc}")


if __name__ == "__main__":
    _main()