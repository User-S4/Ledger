"""The victim API. This is the thing that gets robbed.

    POST /predict   an image in, ten probabilities out
    GET  /stats     counters for P5's dashboard
    GET  /health    is it up, and which backend is loaded

What one request does, in order:

    1. resolve the key                     -> columns 4, 5
    2. check the rate limit                -> 429 if over
    3. pull the image bytes out            -> 400 if unreadable
    4. run the victim model                -> columns 13, 14
    5. push 64 features through the grid   -> column 12
    6. write one row                       -> the log
    7. return the probabilities

Two decisions worth knowing about
    We return the FULL spread of confidence, not just the winning label.
    Real commercial APIs do, customers want it, and it is what makes
    extraction cheap. Returning only a label would make the attack weak and
    the demo dishonest. Stage 7 works by taking that richness away from
    suspicious traffic -- which is why columns 13 and 14 are separate.

    Every request is logged, including failures. A detector that only ever
    sees successful requests is blind to probing.

Run it:
    python -m uvicorn api.main:app --reload
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000    (LAN, for P2/P4)
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.keys import KeyRegistry, RateLimiter, load_tiers  # noqa: E402
from api.logstore import LogStore, encode_embedding, encode_probs  # noqa: E402
from api.validate import ValidationError, extract_image_bytes  # noqa: E402
from api.defense import build_policy  # noqa: E402
from detector.cell_index import CellIndex  # noqa: E402
from detector.cells import assign_cell  # noqa: E402
from detector.tier3_ledger import Tier3Config, score_from_index  # noqa: E402
from victim import loader  # noqa: E402
from victim.embed import Projection  # noqa: E402


# ------------------------------------------------------------------ config

def load_config(path=None) -> dict:
    import yaml

    p = Path(path or os.environ.get("LEDGER_CONFIG", ROOT / "config.yaml"))
    with open(p) as f:
        return yaml.safe_load(f)


CFG = load_config()

# Environment variables win, so P1 can switch runs without editing a file:
#   LEDGER_RUN_ID=eval_seed2 python -m uvicorn api.main:app
RUN_ID = os.environ.get("LEDGER_RUN_ID", CFG["run"]["run_id"])
DB_PATH = os.environ.get("LEDGER_DB", CFG["run"]["db_path"])
BACKEND = os.environ.get("LEDGER_BACKEND", CFG["victim"]["backend"])
MAX_UPLOAD = int(CFG["api"]["max_upload_bytes"])
REQUIRE_KEY = bool(CFG["api"].get("require_key", True))

TIERS = load_tiers(CFG)
STORE = LogStore(DB_PATH, batch_size=1)
REGISTRY = KeyRegistry(STORE, TIERS)
LIMITER = RateLimiter(TIERS)

# ---- Stage 6 + 7: the live ledger and the fightback --------------------
# The tally has to be current DURING a request, because that is when the
# defence decides whether to degrade. A batch job that reads the log
# afterwards is fine for P1's analysis and useless here.
_T3 = (CFG.get("detector", {}) or {}).get("tier3", {}) or {}
TIER3_CFG = Tier3Config(
    cell_size=float(_T3.get("cell_size", 1.0)),
    dims=_T3.get("dims", 8),
    rate_reference=float(_T3.get("rate_reference", 20.0)),
    coverage_reference=float(_T3.get("coverage_reference", 0.25)),
    min_requests=int(_T3.get("min_requests", 30)),
)
TIER3_ENABLED = bool(_T3.get("enabled", False))
CELL_INDEX = CellIndex()

DEFENSE = build_policy(
    CFG,
    scorer=(lambda key: score_from_index(CELL_INDEX, key, cfg=TIER3_CFG))
    if TIER3_ENABLED else None,
)

_proj_path = ROOT / CFG["victim"]["projection"]
PROJECTION = Projection.load(_proj_path) if _proj_path.exists() else None
PROJ_SOURCE = str(_proj_path.name) if PROJECTION else "MISSING"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model before the first request. Otherwise request number one
    # pays the entire model load, which lands in the log as a huge latency
    # outlier and quietly poisons any timing signal P1 builds.
    loader.load_victim(BACKEND)

    print(f"[api] run_id={RUN_ID}  db={DB_PATH}")
    print(f"[api] victim={loader.backend_name()}  projection={PROJ_SOURCE}")
    print(f"[api] {len(STORE.all_keys())} accounts, tiers={TIERS}")
    print(f"[api] tier3={'on' if TIER3_ENABLED else 'off'} "
          f"(dims={TIER3_CFG.dims}, cell_size={TIER3_CFG.cell_size})  "
          f"defense={'ON' if DEFENSE.enabled else 'off'}")
    if DEFENSE.enabled:
        print(f"[api] degradation thresholds {DEFENSE.thresholds} "
              f"-- answers to suspicious accounts will be rounded off")

    # The tally is rebuilt from the log so a restart mid-experiment does not
    # hand every attacker a clean slate. The log is the source of truth.
    try:
        from detector.cell_index import rebuild_from_log

        restored = rebuild_from_log(
            STORE, RUN_ID,
            cell_fn=lambda e: assign_cell(e, TIER3_CFG.cell_size, TIER3_CFG.dims))
        if restored.total_coverage():
            CELL_INDEX._cell_ids = restored._cell_ids
            CELL_INDEX._first_seen = restored._first_seen
            CELL_INDEX._by_key = restored._by_key
            CELL_INDEX._requests = restored._requests
            CELL_INDEX._new_cells = restored._new_cells
            CELL_INDEX._cell_keys = restored._cell_keys
            CELL_INDEX._recent = restored._recent
            print(f"[api] restored ledger: {CELL_INDEX.total_coverage()} cells, "
                  f"{len(CELL_INDEX.accounts())} accounts")
    except Exception as exc:  # noqa: BLE001
        print(f"[api] could not restore ledger ({type(exc).__name__}: {exc})")
    if loader.backend_name() == "stub":
        print("[api] WARNING: stub victim -- results from this run are NOT reportable")
    if PROJECTION is None:
        print("[api] WARNING: no projection. Column 12 will be empty and the")
        print("[api]          ledger cannot work. Fit it:")
        print("[api]          python -m victim.embed fit --cifar")
    yield
    STORE.flush()


app = FastAPI(title="Ledger victim API", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ helpers

class Caller:
    """Who is asking. Populates columns 4, 5 and 6."""

    __slots__ = ("api_key_id", "tier", "ip")

    def __init__(self, api_key_id: str, tier: str, ip: str):
        self.api_key_id = api_key_id
        self.tier = tier
        self.ip = ip


def client_ip(request: Request) -> str:
    """Column 6.

    We trust X-Forwarded-For deliberately. P4's office scenario needs many
    accounts to appear behind one connection, and this is how they say so.
    In production you would only trust it from a known proxy.
    """
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


def write_row(caller: Caller, status: int, latency_ms: float,
              error_code: str | None = None, **extra) -> None:
    """One row per request, success or failure. Never let logging break a
    request -- a log write that throws would turn a 400 into a 500."""
    try:
        STORE.log(
            run_id=RUN_ID,
            ts=time.time(),
            api_key_id=caller.api_key_id,
            tier=caller.tier,
            ip=caller.ip,
            status_code=status,
            error_code=error_code,
            latency_ms=latency_ms,
            **extra,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[api] LOG WRITE FAILED: {type(exc).__name__}: {exc}")


async def authenticate(request: Request,
                       x_api_key: str | None = Header(default=None)) -> Caller:
    """Resolve the key and charge the rate limit. Failures are logged too."""
    ip = client_ip(request)

    if not REQUIRE_KEY and not x_api_key:
        return Caller("anonymous", "free", ip)

    resolved = REGISTRY.resolve(x_api_key)
    if resolved is None:
        write_row(Caller("unknown", "free", ip), 401, 0.0,
                  error_code="no_key" if not x_api_key else "bad_key")
        raise HTTPException(status_code=401, detail="invalid or missing API key")

    caller = Caller(resolved[0], resolved[1], ip)

    if not LIMITER.allow(caller.api_key_id, caller.tier):
        write_row(caller, 429, 0.0, error_code="rate_limited")
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({LIMITER.limit_for(caller.tier)}/min "
                   f"on tier '{caller.tier}')",
        )
    return caller


# ------------------------------------------------------------------ routes

@app.post("/predict")
async def predict(request: Request, caller: Caller = Depends(authenticate)):
    t0 = time.perf_counter()

    try:
        raw = await extract_image_bytes(request, MAX_UPLOAD)
    except ValidationError as exc:
        write_row(caller, exc.status, (time.perf_counter() - t0) * 1000,
                  error_code=exc.code)
        raise HTTPException(status_code=exc.status, detail=exc.code)

    digest, size = loader.image_hash(raw), len(raw)

    try:
        probs, feats = loader.predict([raw], backend=BACKEND)
    except ValidationError as exc:
        write_row(caller, exc.status, (time.perf_counter() - t0) * 1000,
                  error_code=exc.code, input_sha256=digest, input_bytes=size)
        raise HTTPException(status_code=exc.status, detail=exc.code)

    p = probs[0]
    point = PROJECTION.one(feats[0]) if PROJECTION is not None else None

    # ---- Stage 6: update the live tally -------------------------------
    # Costs ~3 microseconds against ~50,000 for inference, so this runs on
    # every request without being felt.
    if point is not None and TIER3_ENABLED:
        CELL_INDEX.observe(caller.api_key_id,
                           assign_cell(point, TIER3_CFG.cell_size, TIER3_CFG.dims),
                           time.time())

    # ---- Stage 7: the fightback ---------------------------------------
    # `p` is what the model believed (column 13). `returned` is what the
    # customer actually receives (column 14). Keeping both is what makes the
    # defence a measurable fact rather than a claim -- and the top-1 label is
    # identical in both, at every degradation level.
    returned, degradation_level, suspicion = DEFENSE.apply(caller.api_key_id, p)

    latency = (time.perf_counter() - t0) * 1000
    write_row(
        caller, 200, latency,
        input_sha256=digest,
        input_bytes=size,
        embedding=encode_embedding(point) if point is not None else None,
        model_probs=encode_probs(p),
        returned_probs=encode_probs(returned),
        degradation_level=degradation_level,
        suspicion_score=suspicion,
    )

    summary = loader.summarize(returned)
    return {
        "label": summary["label_name"],
        "label_index": summary["label"],
        "probabilities": [float(v) for v in returned],
        "classes": loader.CLASSES,
        "latency_ms": round(latency, 2),
    }


@app.get("/stats")
async def stats():
    s = STORE.stats(run_id=RUN_ID)
    s["run_id"] = RUN_ID
    s["victim_backend"] = loader.backend_name()
    s["defense_enabled"] = DEFENSE.enabled
    if TIER3_ENABLED:
        s.update({
            "cells_revealed": CELL_INDEX.total_coverage(),
            "accounts_tracked": len(CELL_INDEX.accounts()),
            "discoveries_in_window": CELL_INDEX.snapshot()["discoveries_in_window"],
        })
    return s


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "run_id": RUN_ID,
        "victim_backend": loader.backend_name(),
        "projection": PROJ_SOURCE,
        "require_key": REQUIRE_KEY,
        "tiers": TIERS,
        "tier3": TIER3_ENABLED,
        "defense": DEFENSE.enabled,
        "reportable": loader.backend_name() == "torch" and PROJECTION is not None,
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Never return a stack trace. A traceback leaks file paths, library
    versions and directory structure -- free reconnaissance. Stage 9
    depends on this."""
    print(f"[api] unhandled on {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "internal_error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=CFG["api"]["host"], port=int(CFG["api"]["port"]))