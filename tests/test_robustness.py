"""Stage 9: input robustness.

Everything returns a clean error, nothing crashes, no stack traces.

    python -m pytest tests/ -q
    LEDGER_BACKEND=stub python -m pytest tests/ -q      (no torch needed)

Two things are being proved, and they pull in opposite directions:

  Nothing that is not an image gets through, and nothing leaks a traceback,
  a file path or a library version when it is refused.

  Everything that IS an image gets through, however inconvenient. A huge
  photo, a grayscale JPEG, transparency, a 1x1 pixel. Turning those away
  would be a false positive dressed up as validation.

The second half matters as much as the first. A service that rejects a
customer's ordinary phone photo is broken, whatever its security posture.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LEDGER_BACKEND", "stub")
os.environ.setdefault("LEDGER_DB", "data/test.db")
os.environ.setdefault("LEDGER_RUN_ID", "cal_test")


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def api():
    """A live app plus an enterprise key, so rate limits do not interfere
    with tests that are about something else."""
    from fastapi.testclient import TestClient

    from api.main import REGISTRY, app

    client = TestClient(app)
    with client:  # runs startup, so the victim is loaded
        key = REGISTRY.provision(1, "enterprise", "test_harness")[0]["secret"]
        yield client, key


def png(seed: int = 0, size=(32, 32), mode="RGB", fmt="PNG") -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def post(client, key, **kw):
    headers = kw.pop("headers", {})
    headers.setdefault("X-API-Key", key)
    return client.post("/predict", headers=headers, **kw)


def assert_no_leak(response):
    """No stack frames, no paths, no library internals. A traceback is free
    reconnaissance for whoever sent the bad input."""
    body = response.text
    for marker in ("Traceback", 'File "', "/home/", "\\Users\\",
                   "site-packages", "__main__"):
        assert marker not in body, f"response leaked {marker!r}: {body[:300]}"


# ------------------------------------------------------------------ happy path

def test_health_reports_backend(api):
    client, _ = api
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["victim_backend"] in {"torch", "stub"}
    # A run on the stub must never be treated as reportable.
    assert body["reportable"] == (body["victim_backend"] == "torch"
                                  and body["projection"] != "MISSING")


def test_predict_multipart(api):
    client, key = api
    r = post(client, key, files={"file": ("a.png", png(1), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["probabilities"]) == 10
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-4
    assert body["label"] in body["classes"]


def test_predict_base64(api):
    client, key = api
    r = post(client, key, json={"image_b64": base64.b64encode(png(2)).decode()})
    assert r.status_code == 200


def test_same_bytes_same_answer(api):
    """P2's fidelity measurement depends on this being exactly true."""
    client, key = api
    raw = png(3)
    a = post(client, key, files={"file": ("a.png", raw, "image/png")}).json()
    b = post(client, key, files={"file": ("a.png", raw, "image/png")}).json()
    assert a["probabilities"] == b["probabilities"]


# ------------------------------------- inconvenient but honest images

@pytest.mark.parametrize("name,raw", [
    ("large photo",     png(10, (400, 300))),
    ("1x1 pixel",       png(11, (1, 1))),
    ("tall strip",      png(12, (8, 512))),
    ("grayscale png",   png(13, mode="L")),
    ("transparent png", png(14, mode="RGBA")),
    ("jpeg",            png(15, fmt="JPEG")),
    ("bmp",             png(16, fmt="BMP")),
    ("webp",            png(17, fmt="WEBP")),
])
def test_honest_images_are_accepted(api, name, raw):
    """Rejecting these would be a false positive, not security."""
    client, key = api
    r = post(client, key, files={"file": ("a.img", raw, "application/octet-stream")})
    assert r.status_code == 200, f"{name} was refused: {r.text[:200]}"


# ------------------------------------------------------------------ auth

def test_missing_key(api):
    client, _ = api
    r = client.post("/predict", files={"file": ("a.png", png(20), "image/png")})
    assert r.status_code == 401
    assert_no_leak(r)


def test_bad_key(api):
    client, _ = api
    r = post(client, "sk_definitely_not_real",
             files={"file": ("a.png", png(21), "image/png")})
    assert r.status_code == 401
    assert_no_leak(r)


def test_key_in_wrong_header_is_rejected(api):
    client, key = api
    r = client.post("/predict", headers={"Authorization": f"Bearer {key}"},
                    files={"file": ("a.png", png(22), "image/png")})
    assert r.status_code == 401


# ------------------------------------------------------------------ malformed

@pytest.mark.parametrize("name,kwargs,status", [
    ("empty json body",   dict(json={}), 400),
    ("json without key",  dict(json={"wrong_field": "x"}), 400),
    ("null image",        dict(json={"image_b64": None}), 400),
    ("number image",      dict(json={"image_b64": 12345}), 400),
    ("list image",        dict(json={"image_b64": [1, 2, 3]}), 400),
    ("empty string",      dict(json={"image_b64": ""}), 400),
    ("not base64",        dict(json={"image_b64": "!!!not base64!!!"}), 400),
    ("base64 of text",    dict(json={"image_b64": base64.b64encode(b"hello").decode()}), 400),
    ("empty file",        dict(files={"file": ("a.png", b"", "image/png")}), 400),
    ("text as png",       dict(files={"file": ("a.png", b"hello world", "image/png")}), 400),
    ("truncated png",     dict(files={"file": ("a.png", png(30)[:40], "image/png")}), 400),
    ("header only",       dict(files={"file": ("a.png", png(31)[:8], "image/png")}), 400),
    ("pdf as png",        dict(files={"file": ("a.png", b"%PDF-1.4\n%...", "image/png")}), 400),
    ("html as png",       dict(files={"file": ("a.png", b"<html><body>x", "image/png")}), 400),
    ("zip as png",        dict(files={"file": ("a.png", b"PK\x03\x04" + b"\x00" * 60, "image/png")}), 400),
    ("random bytes",      dict(files={"file": ("a.bin", bytes(range(256)) * 4, "image/png")}), 400),
    ("null bytes",        dict(files={"file": ("a.png", b"\x00" * 5000, "image/png")}), 400),
    ("giant payload",     dict(files={"file": ("a.png", b"\x00" * 3_000_000, "image/png")}), 413),
])
def test_malformed_input(api, name, kwargs, status):
    client, key = api
    r = post(client, key, **kwargs)
    assert r.status_code == status, f"{name}: got {r.status_code} - {r.text[:200]}"
    assert_no_leak(r)


def test_broken_json_body(api):
    client, key = api
    r = post(client, key, content=b"{not valid json",
             headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert_no_leak(r)


def test_non_utf8_json_body(api):
    client, key = api
    r = post(client, key, content=b"\xff\xfe\x00\x01",
             headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert_no_leak(r)


def test_decompression_bomb_is_refused():
    """A small file that declares an enormous bitmap. Without the header
    check this allocates gigabytes before anything notices."""
    from api.validate import MAX_PIXELS, ValidationError, decode_image_array

    buf = io.BytesIO()
    Image.new("RGB", (20000, 20000), (0, 0, 0)).save(buf, "PNG")
    bomb = buf.getvalue()
    assert 20000 * 20000 > MAX_PIXELS
    with pytest.raises(ValidationError) as exc:
        decode_image_array(bomb)
    assert exc.value.code == "image_too_large"
    # The point is the ratio: ~1MB on disk asking us to allocate 400M pixels.
    assert len(bomb) < 2_000_000


def test_fuzz_never_leaks(api):
    """Any status is fine. Leaking is not."""
    client, key = api
    rng = np.random.default_rng(0)
    for _ in range(25):
        blob = rng.integers(0, 256, int(rng.integers(1, 4000)),
                            dtype=np.uint8).tobytes()
        r = post(client, key, files={"file": ("x.bin", blob, "image/png")})
        assert r.status_code in (200, 400, 413), f"unexpected {r.status_code}"
        assert_no_leak(r)


def test_wrong_method_and_unknown_route(api):
    client, _ = api
    assert client.get("/predict").status_code == 405
    assert client.get("/no_such_route").status_code == 404


# ------------------------------------------------------------------ rate limits

def test_rate_limit_is_per_tier_and_per_account():
    from api.keys import DEFAULT_TIERS, RateLimiter

    lim = RateLimiter()
    for tier, limit in DEFAULT_TIERS.items():
        allowed = sum(lim.allow(f"k_{tier}", tier, now=1000.0)
                      for _ in range(limit + 50))
        assert allowed == limit
    assert lim.allow("k_untouched", "free", now=1000.0)
    assert lim.allow("k_free", "free", now=1061.0)   # window slid


def test_rate_limited_requests_are_logged(api):
    """A prober who only ever gets 429s must still appear in the log."""
    from api.main import RUN_ID, STORE, REGISTRY

    client, _ = api
    key = REGISTRY.provision(1, "free", "rate_test")[0]["secret"]
    codes = [post(client, key,
                  files={"file": ("a.png", png(40 + i), "image/png")}).status_code
             for i in range(70)]
    assert 429 in codes
    STORE.flush()
    df = STORE.read_df(run_id=RUN_ID)
    assert len(df[df.error_code == "rate_limited"]) >= 1


# ------------------------------------------------------------------ the log

def test_failures_are_logged_with_a_reason(api):
    from api.main import RUN_ID, STORE

    client, key = api
    post(client, key, files={"file": ("a.txt", b"nope", "text/plain")})
    STORE.flush()
    df = STORE.read_df(run_id=RUN_ID)
    bad = df[df.status_code >= 400]
    assert len(bad) > 0
    assert bad.error_code.notna().all(), "a failure with no reason is unusable"
    assert bad.embedding.isna().all(), "failed requests must not carry a point"


def test_logged_row_matches_what_the_client_received(api):
    """Column 14 is the contract's record of what left the building."""
    from api.main import RUN_ID, STORE

    client, key = api
    r = post(client, key, files={"file": ("a.png", png(50), "image/png")})
    STORE.flush()
    df = STORE.read_df(run_id=RUN_ID)
    row = df[df.status_code == 200].iloc[-1]
    assert row["returned_probs"] == r.json()["probabilities"]
    assert row["model_probs"] == row["returned_probs"], "identical before Stage 7"
    assert row["degradation_level"] == 0


def test_schema_refuses_off_contract_columns(tmp_path):
    from api.logstore import LogStore

    store = LogStore(tmp_path / "x.db")
    with pytest.raises(KeyError):
        store.log(run_id="r", api_key_id="k", tier="free", ip="1.2.3.4",
                  status_code=200, latency_ms=1.0, cell_id=7)


def test_answer_key_stays_out_of_the_log(api):
    """`owner` is ground truth. If it reaches the log it can reach a
    feature, and every reported number becomes worthless."""
    from api.main import RUN_ID, STORE

    _ = api
    STORE.flush()
    df = STORE.read_df(run_id=RUN_ID)
    assert "owner" not in df.columns
    assert "owner" in STORE.owners_df().columns