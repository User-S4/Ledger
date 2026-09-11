"""Background traffic simulation manager for the ZeroTrace SOC Dashboard.

Allows the web dashboard to trigger realistic honest traffic or a 400-key distributed
theft directly from the UI without manual terminal commands.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SIM_STATE = {
    "running": False,
    "mode": "idle",
    "requests_sent": 0,
}

_SIM_TASK: asyncio.Task | None = None


def _random_image_bytes(seed: int, size: int = 32) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


async def _run_simulation(mode: str, port: int = 8000):
    global SIM_STATE
    SIM_STATE["running"] = True
    SIM_STATE["mode"] = mode

    if mode == "honest":
        keys_pool = []
        for fn in [
            "data/casual_keys.json",
            "data/batch_keys.json",
            "data/bursty_keys.json",
            "data/researcher_keys.json",
            "data/office_keys.json",
        ]:
            p = ROOT / fn
            if p.exists():
                keys_pool.extend(json.loads(p.read_text()))
        delay = 0.10  # ~10 req/sec
    else:  # attack
        p = ROOT / "data/attacker_keys.json"
        keys_pool = json.loads(p.read_text()) if p.exists() else []
        delay = 0.04  # ~25 req/sec

    if not keys_pool:
        SIM_STATE["running"] = False
        SIM_STATE["mode"] = "idle"
        return

    url = f"http://127.0.0.1:{port}/predict"
    idx = 0

    # Build stable IP per key
    ip_by_key = {}
    for i, k in enumerate(keys_pool):
        kid = k["api_key_id"]
        owner = k.get("owner", "")
        if owner == "office_acme" or "office" in kid:
            ip_by_key[kid] = "203.0.113.7"  # Corporate office NAT gateway
        elif mode == "attack":
            ip_by_key[kid] = f"198.51.{i // 256}.{i % 256}"
        else:
            ip_by_key[kid] = f"198.51.100.{(i % 250) + 2}"

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    async with httpx.AsyncClient(limits=limits, timeout=5.0) as client:
        while SIM_STATE["running"] and SIM_STATE["mode"] == mode:
            key_entry = keys_pool[idx % len(keys_pool)]
            secret = key_entry["secret"]
            key_id = key_entry.get("api_key_id", f"k_{idx}")
            idx += 1

            seed = random.randint(1, 10_000_000)
            raw = _random_image_bytes(seed)
            b64_str = base64.b64encode(raw).decode()
            ip = ip_by_key.get(key_id, "198.51.100.2")

            headers = {
                "X-API-Key": secret,
                "X-Forwarded-For": ip,
                "Content-Type": "application/json",
            }

            try:
                await client.post(url, json={"image_b64": b64_str}, headers=headers)
                SIM_STATE["requests_sent"] += 1
            except Exception:
                pass

            await asyncio.sleep(delay)

    SIM_STATE["running"] = False
    SIM_STATE["mode"] = "idle"


def start_simulation(mode: str, port: int = 8000) -> dict:
    global _SIM_TASK, SIM_STATE
    stop_simulation()
    SIM_STATE["running"] = True
    SIM_STATE["mode"] = mode
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    _SIM_TASK = loop.create_task(_run_simulation(mode, port=port))
    return {"status": "started", "mode": mode}


def stop_simulation() -> dict:
    global _SIM_TASK, SIM_STATE
    SIM_STATE["running"] = False
    SIM_STATE["mode"] = "idle"
    if _SIM_TASK and not _SIM_TASK.done():
        _SIM_TASK.cancel()
    return {"status": "stopped"}


def get_status() -> dict:
    return dict(SIM_STATE)
