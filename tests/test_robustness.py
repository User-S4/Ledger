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

# Parameters are short NAMES, never raw bytes. pytest builds test ids from
# the parameters and puts them in an environment variable; on Windows that
# has a 32767-character limit, so passing image bytes here blows up the
# whole session before a single test runs.
HONEST_IMAGES = {
    "large_photo":     lambda: png(10, (400, 300)),
    "one_pixel":       lambda: png(11, (1, 1)),
    "tall_strip":      lambda: png(12, (8, 512)),
    "grayscale_png":   lambda: png(13, mode="L"),
    "transparent_png": lambda: png(14, mode="RGBA"),
    "jpeg":            lambda: png(15, fmt="JPEG"),
    "bmp":             lambda: png(16, fmt="BMP"),
    "webp":            lambda: png(17, fmt="WEBP"),
}


@pytest.mark.parametrize("name", sorted(HONEST_IMAGES))
def test_honest_images_are_accepted(api, name):
    """Rejecting these would be a false positive, not security."""
    client, key = api
    raw = HONEST_IMAGES[name]()
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

MALFORMED = {
    "empty_json_body":  (lambda: dict(json={}), 400),
    "json_without_key": (lambda: dict(json={"wrong_field": "x"}), 400),
    "null_image":       (lambda: dict(json={"image_b64": None}), 400),
    "number_image":     (lambda: dict(json={"image_b64": 12345}), 400),
    "list_image":       (lambda: dict(json={"image_b64": [1, 2, 3]}), 400),
    "empty_string":     (lambda: dict(json={"image_b64": ""}), 400),
    "not_base64":       (lambda: dict(json={"image_b64": "!!!not base64!!!"}), 400),
    "base64_of_text":   (lambda: dict(json={"image_b64": base64.b64encode(b"hello").decode()}), 400),
    "empty_file":       (lambda: dict(files={"file": ("a.png", b"", "image/png")}), 400),
    "text_as_png":      (lambda: dict(files={"file": ("a.png", b"hello world", "image/png")}), 400),
    "truncated_png":    (lambda: dict(files={"file": ("a.png", png(30)[:40], "image/png")}), 400),
    "header_only":      (lambda: dict(files={"file": ("a.png", png(31)[:8], "image/png")}), 400),
    "pdf_as_png":       (lambda: dict(files={"file": ("a.png", b"%PDF-1.4\n%...", "image/png")}), 400),
    "html_as_png":      (lambda: dict(files={"file": ("a.png", b"<html><body>x", "image/png")}), 400),
    "zip_as_png":       (lambda: dict(files={"file": ("a.png", b"PK\x03\x04" + b"\x00" * 60, "image/png")}), 400),
    "random_bytes":     (lambda: dict(files={"file": ("a.bin", bytes(range(256)) * 4, "image/png")}), 400),
    "null_bytes":       (lambda: dict(files={"file": ("a.png", b"\x00" * 5000, "image/png")}), 400),
    "giant_payload":    (lambda: dict(files={"file": ("a.png", b"\x00" * 3_000_000, "image/png")}), 413),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_input(api, name):
    client, key = api
    build, status = MALFORMED[name]
    r = post(client, key, **build())
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


# ------------------------------------------------------------------ load

def test_logstore_loses_nothing_under_concurrent_writes(tmp_path):
    """Many threads writing at once must lose nothing.

    P2's attack and P4's traffic hit the API simultaneously from different
    machines. Uvicorn handles those on separate threads, and they all share
    one LogStore. A row lost here is invisible: no error, no crash, just a
    log that quietly disagrees with what actually happened -- and every
    number the team reports is computed from that log.
    """
    import threading

    from api.logstore import LogStore

    store = LogStore(tmp_path / "concurrent.db", batch_size=1)
    threads, per_thread = 8, 500

    def writer(worker: int) -> None:
        for i in range(per_thread):
            store.log(run_id="cal_load", ts=1000.0 + i,
                      api_key_id=f"k_{worker:03d}", tier="free",
                      ip=f"10.0.0.{worker}", status_code=200, latency_ms=1.0)

    workers = [threading.Thread(target=writer, args=(w,)) for w in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    store.flush()

    df = store.read_df(run_id="cal_load")
    assert len(df) == threads * per_thread, (
        f"lost {threads * per_thread - len(df)} rows under concurrent writes")
    assert df.api_key_id.nunique() == threads
    assert df.request_id.is_unique, "duplicate primary keys"
    store.close()


def test_rate_limiter_is_exact_under_concurrent_callers():
    """The limit must hold when many threads charge the same account.

    A limiter that leaks under load lets an attacker exceed their tier by
    sending faster -- which is precisely when it matters.
    """
    import threading

    from api.keys import DEFAULT_TIERS, RateLimiter

    lim = RateLimiter()
    limit = DEFAULT_TIERS["pro"]
    granted = []
    lock = threading.Lock()

    def hammer() -> None:
        allowed = sum(lim.allow("k_shared", "pro", now=5000.0) for _ in range(400))
        with lock:
            granted.append(allowed)

    workers = [threading.Thread(target=hammer) for _ in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert sum(granted) == limit, (
        f"limiter granted {sum(granted)} of {limit} allowed under 8 threads")


def test_api_handles_concurrent_requests(api):
    """End to end: many clients at once, every response accounted for."""
    import threading

    from api.main import RUN_ID, STORE, REGISTRY

    client, _ = api
    keys = [k["secret"] for k in REGISTRY.provision(6, "enterprise", "load_test")]
    before = len(STORE.read_df(run_id=RUN_ID))

    results: list[int] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        codes = [post(client, keys[idx],
                      files={"file": (f"a{i}.png", png(900 + idx * 50 + i), "image/png")}
                      ).status_code for i in range(25)]
        with lock:
            results.extend(codes)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(len(keys))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    STORE.flush()

    assert len(results) == len(keys) * 25
    assert all(c == 200 for c in results), f"non-200 under load: {set(results)}"

    after = len(STORE.read_df(run_id=RUN_ID))
    assert after - before == len(results), (
        f"{len(results)} requests served but {after - before} rows logged")


def test_cell_index_is_exact_under_concurrent_observers():
    """The live tally must not drop or double-count under load."""
    import threading

    from detector.cell_index import CellIndex

    index = CellIndex()
    threads, per_thread = 8, 400

    def worker(w: int) -> None:
        for i in range(per_thread):
            index.observe(f"k_{w}", (w, i % 40, 0, 0, 0, 0), 1000.0 + i * 0.01)

    workers = [threading.Thread(target=worker, args=(w,)) for w in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    snap = index.snapshot()
    assert snap["requests_seen"] == threads * per_thread
    assert snap["accounts_seen"] == threads
    assert index.total_coverage() == threads * 40   # each thread visits 40 cells


# ------------------------------------------------------------------ defence

def test_degradation_never_changes_the_label():
    """The one rule Stage 7 follows. We will sometimes be wrong about who is
    hostile, and a service that lies to customers it wrongly suspects is
    indefensible."""
    from api.defense import degrade

    rng = np.random.default_rng(0)
    for _ in range(400):
        p = rng.dirichlet(np.ones(10) * rng.uniform(0.2, 3.0))
        winner = int(np.argmax(p))
        for level in (0, 1, 2, 3):
            out = degrade(p, level)
            assert int(np.argmax(out)) == winner, f"level {level} changed the label"


def test_degradation_always_returns_a_valid_distribution():
    from api.defense import degrade

    rng = np.random.default_rng(1)
    for _ in range(400):
        p = rng.dirichlet(np.ones(10) * rng.uniform(0.2, 3.0))
        for level in (0, 1, 2, 3):
            out = degrade(p, level)
            assert len(out) == 10
            assert abs(sum(out) - 1.0) < 1e-6, f"level {level} sums to {sum(out)}"
            assert all(v >= 0.0 for v in out)


def test_degradation_destroys_the_boundary_information():
    """A near-tie is a coordinate on the decision boundary -- the thing a
    clone is really copying. Degrading must make it unrecoverable.

    The property that matters is collision: several queries with DIFFERENT
    true margins must produce the SAME degraded answer. If they did not, the
    thief could still tell them apart and the boundary would survive.
    """
    from api.defense import degrade

    def near_tie(margin):
        p = np.full(10, 0.02)
        p[2] = 0.30 + margin / 2
        p[3] = 0.30 - margin / 2
        return p / p.sum()

    margins = [0.004, 0.012, 0.021, 0.030]
    for level in (1, 2, 3):
        outs = {tuple(np.round(degrade(near_tie(m), level), 6)) for m in margins}
        assert len(outs) == 1, (
            f"level {level} still distinguishes margins {margins}: {len(outs)} "
            f"distinct answers")

    # ...and untouched, they are all clearly different.
    raw = {tuple(np.round(degrade(near_tie(m), 0), 6)) for m in margins}
    assert len(raw) == len(margins)


def test_level_thresholds():
    from api.defense import level_for_score

    assert level_for_score(None) == 0
    assert level_for_score(0.0) == 0
    assert level_for_score(0.49) == 0
    assert level_for_score(0.5) == 1
    assert level_for_score(0.7) == 2
    assert level_for_score(0.95) == 3


def test_defence_is_off_until_enabled():
    """Nothing degrades until the team switches it on."""
    from api.defense import DefensePolicy

    policy = DefensePolicy(scorer=lambda k: 0.99, enabled=False)
    probs = [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]
    out, level, score = policy.apply("k_1", probs)
    assert level == 0
    assert out == probs


def test_defence_rises_at_once_and_falls_slowly():
    """Hysteresis. Without it an account flips level request by request, and
    the flipping itself tells a thief where the threshold sits."""
    from api.defense import DefensePolicy

    score = {"v": 0.95}
    policy = DefensePolicy(scorer=lambda k: score["v"], cooldown_s=100.0,
                           enabled=True)

    assert policy.level_for("k_1", score["v"], now=0.0) == 3
    score["v"] = 0.0
    assert policy.level_for("k_1", score["v"], now=10.0) == 3    # still hot
    assert policy.level_for("k_1", score["v"], now=99.0) == 3
    assert policy.level_for("k_1", score["v"], now=101.0) == 0   # cooled down


def test_defence_scores_accounts_independently():
    from api.defense import DefensePolicy

    scores = {"k_bad": 0.95, "k_good": 0.05}
    policy = DefensePolicy(scorer=lambda k: scores[k], enabled=True)
    probs = [0.4, 0.3, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]

    bad_out, bad_level, _ = policy.apply("k_bad", probs)
    good_out, good_level, _ = policy.apply("k_good", probs)
    assert bad_level == 3 and good_level == 0
    assert good_out == probs