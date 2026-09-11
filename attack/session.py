"""
The shared engine under all three attacks.

The three attack files (knockoff, distributed, mixed) differ in exactly one
thing: how they hand out api keys across the requests. Everything else --
turning a photo into what the API expects, sending it, collecting the answer,
saving the stolen pairs -- is identical, so it lives here once.

We send through P3's real endpoint using their client.py. We do NOT write any
log. The API writes the log; that is the whole reason our Stage 5 evidence is
credible. What we keep is only the stolen (photo, answer) pairs, for training.

If the API isn't up yet, pass service=LocalService() and this still runs
against the in-memory victim -- same code path, no network, so you can develop
the three attacks today and confirm them against the real API tomorrow.
"""

from __future__ import annotations

import base64
import io
import random
import time
from pathlib import Path

import numpy as np


def _png_bytes(image_uint8):
    """One (32,32,3) uint8 photo -> PNG bytes, matching what the client sends."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image_uint8, dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def run_session(images, key_for, service=None, api=None, ip_for=None,
                out_npz=None, delay_s: float = 0.0, jitter_s: float = 0.0,
                timing_out: dict | None = None, verbose=True):
    """Send every photo, collect every answer.

    key_for(i)  -> which api key the i-th request goes out under. This one
                   function is the entire difference between the three attacks.
    ip_for(i)   -> optional; which X-Forwarded-For to claim. None = don't set.
    delay_s     -> sleep duration between consecutive requests (pacing evasion).
    jitter_s    -> uniform random variation (+/-) applied to delay_s.
    timing_out  -> optional dict populated with wall-clock timing metrics.

    Provide ONE of:
      api      an ApiTarget (talks to P3's live endpoint, real logs written)
      service  a LocalService (in-memory victim, no network, no logs)
    """
    if (api is None) == (service is None):
        raise ValueError("pass exactly one of api= or service=")

    images = np.asarray(images, dtype=np.uint8)
    answers = np.zeros((len(images), 10), dtype=np.float32)
    keys_used = [None] * len(images)

    start_t = time.time()

    if api is not None:
        for i, img in enumerate(images):
            if i > 0 and delay_s > 0:
                sleep_t = delay_s + (random.uniform(-jitter_s, jitter_s) if jitter_s > 0 else 0.0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
            k = key_for(i)
            keys_used[i] = k
            answers[i] = api.predict_one(
                _png_bytes(img), key=k, ip=ip_for(i) if ip_for else None)
            if verbose and (i + 1) % 500 == 0:
                print(f"    sent {i + 1}/{len(images)}")
    else:
        # Local shortcut: no per-request keys in the log (there is no log),
        # so we can batch for speed and just record which key we'd have used.
        answers = service.predict(images)["returned_probs"].astype(np.float32)
        for i in range(len(images)):
            keys_used[i] = key_for(i)

    end_t = time.time()
    if timing_out is not None:
        timing_out["start_time"] = start_t
        timing_out["end_time"] = end_t
        timing_out["elapsed_s"] = end_t - start_t
        timing_out["total_queries"] = len(images)
        timing_out["effective_qps"] = len(images) / max(end_t - start_t, 1e-6)

    if out_npz:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, images=images, answers=answers)
        if verbose:
            print(f"    stole {len(images)} pairs -> {out_npz}")

    return answers, keys_used


class ApiTarget:
    """Thin wrapper over P3's client.predict(), returning a probability vector.

    Their response gives `probabilities` aligned to `classes`; we keep the
    vector in the fixed CIFAR-10 order so training code never worries about
    label ordering.
    """

    CLASS_ORDER = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, url="http://127.0.0.1:8000", timeout=30.0):
        from api import client   # P3's file
        self._client = client
        self.url = url
        self.timeout = timeout
        self.request_count = 0

    def predict_one(self, png_bytes, key, ip=None, max_retries: int = 4):
        for attempt in range(max_retries):
            status, body = self._client.predict(
                png_bytes, key=key, url=self.url, ip=ip, timeout=self.timeout)
            if status == 200:
                self.request_count += 1
                order = {c: i for i, c in enumerate(body["classes"])}
                vec = np.zeros(10, dtype=np.float32)
                for name in self.CLASS_ORDER:
                    vec[self.CLASS_ORDER.index(name)] = body["probabilities"][order[name]]
                return vec
            if status == 429 and attempt < max_retries - 1:
                # Exponential backoff on rate limit
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"API returned {status}: {body.get('detail')}")


def load_key_secrets(path):
    """Read secrets from a file P3 made with `python -m api.keys --out`."""
    import json
    data = json.loads(Path(path).read_text())
    return [row["secret"] for row in data]
