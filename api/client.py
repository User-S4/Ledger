"""A small client for the victim API.

The /docs page cannot draw an upload form for /predict, because the endpoint
parses its own body in order to accept both multipart uploads and JSON. This
is the substitute -- and the reference P2's attack and P4's traffic
generators should build on rather than reinventing.

Send one image:
    python -m api.client --key sk_... --image cat.png

Send a random image (no file needed, just checks the pipe works):
    python -m api.client --key sk_...

Send many, to watch the log fill up:
    python -m api.client --key sk_... --count 100

Use a whole pool of keys, round-robin -- P2's distributed attack in miniature:
    python -m api.client --keys data/attacker_keys.json --count 500

Pretend to be an office behind one connection -- P4's false-positive case:
    python -m api.client --keys data/office_keys.json --count 300 --ip 203.0.113.7
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests

DEFAULT_URL = "http://127.0.0.1:8000"


def random_png(seed: int, size: int = 32) -> bytes:
    """A throwaway image. Fine for checking the pipe; meaningless as data."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


_SESSION: requests.Session | None = None


def get_session() -> requests.Session:
    """Reusable connection pool to avoid Windows socket exhaustion (WinError 10048)."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100, pool_maxsize=200, max_retries=3
        )
        _SESSION.mount("http://", adapter)
        _SESSION.mount("https://", adapter)
    return _SESSION


def predict(raw: bytes, key: str, url: str = DEFAULT_URL,
            ip: str | None = None, timeout: float = 30.0,
            session: requests.Session | None = None) -> tuple[int, dict]:
    """Send one image. Returns (status_code, body).

    `ip` sets X-Forwarded-For, which the API trusts -- that is how P4 makes
    many accounts appear to share one internet connection.
    """
    headers = {"X-API-Key": key, "Content-Type": "application/json"}
    if ip:
        headers["X-Forwarded-For"] = ip
    s = session or get_session()
    r = s.post(
        f"{url.rstrip('/')}/predict",
        data=json.dumps({"image_b64": base64.b64encode(raw).decode()}),
        headers=headers,
        timeout=timeout,
    )
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"detail": r.text[:200]}


def load_keys(path: str | Path) -> list[str]:
    """Read the secrets out of a file written by `python -m api.keys --out`."""
    data = json.loads(Path(path).read_text())
    return [row["secret"] for row in data]


def _main() -> None:
    ap = argparse.ArgumentParser(description="Send images to the victim API")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--key", help="a single API key secret")
    ap.add_argument("--keys", help="JSON file of keys, used round-robin")
    ap.add_argument("--image", help="image file to send; random if omitted")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--ip", help="spoof X-Forwarded-For (P4's office scenario)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    a = ap.parse_args()

    if a.keys:
        keys = load_keys(a.keys)
        print(f"{len(keys)} keys from {a.keys}, used round-robin")
    elif a.key:
        keys = [a.key]
    else:
        raise SystemExit("give --key SECRET or --keys FILE.json")

    fixed = Path(a.image).read_bytes() if a.image else None

    statuses: dict[int, int] = {}
    labels: dict[str, int] = {}
    t0 = time.time()

    for i in range(a.count):
        raw = fixed if fixed is not None else random_png(i)
        status, body = predict(raw, keys[i % len(keys)], a.url, a.ip)
        statuses[status] = statuses.get(status, 0) + 1

        if status == 200:
            labels[body["label"]] = labels.get(body["label"], 0) + 1
            if a.count == 1:
                print(f"\n{body['label']} "
                      f"({max(body['probabilities']):.1%} confident, "
                      f"{body['latency_ms']}ms)\n")
                print("full spread -- this is what a thief collects:")
                for name, p in sorted(zip(body["classes"], body["probabilities"]),
                                      key=lambda kv: -kv[1]):
                    print(f"  {name:11s} {p:6.2%}  {'#' * int(p * 40)}")
        elif a.count == 1:
            print(f"{status}: {body.get('detail')}")

        if a.delay:
            time.sleep(a.delay)

    if a.count > 1:
        el = time.time() - t0
        print(f"\n{a.count} requests in {el:.1f}s ({a.count / el:.0f}/sec)")
        for code, n in sorted(statuses.items()):
            note = {200: "ok", 401: "bad key", 429: "rate limited",
                    400: "bad request", 413: "too large"}.get(code, "")
            print(f"  {code} {note:14s} {n}")
        if labels:
            top = sorted(labels.items(), key=lambda kv: -kv[1])[:5]
            print("  labels: " + ", ".join(f"{k} {v}" for k, v in top))


if __name__ == "__main__":
    _main()
