#!/usr/bin/env python3
"""
Runs a traffic scenario from a YAML file (see scenarios/*.yaml) against the real
victim API. Per SCHEMA.md, only the API writes log rows — this script only sends
requests. It reuses api/client.py's predict() for the actual HTTP call, per that
file's own docstring ("the reference P2's attack and P4's traffic generators should
build on rather than reinventing") — so the request format always matches whatever
Person 3's API actually expects, instead of a guess living in two places.

Usage:
    # First, provision real test keys if they don't exist yet (writes into the same
    # DB the API reads from — you can do this yourself, no need to wait on P3):
    python -m api.keys --count 5  --tier free --owner casual      --out data/casual_keys.json
    python -m api.keys --count 60 --tier free --owner office_acme --out data/office_keys.json

    # Then run a scenario against the real, running API:
    python scenario.py scenarios/calibration_seed1.yaml \\
        --api-url http://localhost:8000 \\
        --keys-file data/casual_keys.json --keys-file data/office_keys.json

    # Before real keys exist, test with throwaway local ones instead:
    python scenario.py scenarios/calibration_seed1.yaml --stub-keys

    # See what would be sent without hitting the network:
    python scenario.py scenarios/calibration_seed1.yaml --dry-run

Same scenario file + same seed = same sequence of (account, time, weird-flag) events,
every time — required so calibration_seed1.yaml is safe to tune against and
evaluation_seed2.yaml is trustworthy to report from.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# Match api/main.py's own trick so `from api.client import predict` works
# regardless of where this script is run from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.client import predict, random_png  # noqa: E402  (this IS the request format)

from profiles import PROFILES
from multitenant import office


# ---------------------------------------------------------------------------
# Input bytes. Normal traffic reuses P3's random_png() so it's literally the same
# generator P3 tests their own API with. "weird" inputs (researcher profile) need
# their own generator since random_png() only makes uniform noise.
# ---------------------------------------------------------------------------

def weird_image_bytes(rng: random.Random, size: int = 32) -> bytes:
    """Deliberately odd inputs: near-blank, blown-out, or pure noise."""
    choice = rng.random()
    if choice < 0.33:
        arr = np.zeros((size, size, 3), dtype=np.uint8)
    elif choice < 0.66:
        arr = np.full((size, size, 3), 255, dtype=np.uint8)
    else:
        arr = np.random.default_rng(rng.randint(0, 1_000_000)).integers(
            0, 256, (size, size, 3), dtype=np.uint8
        )
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def image_for_event(rng: random.Random, weird: bool) -> bytes:
    if weird:
        return weird_image_bytes(rng)
    return random_png(rng.randint(0, 1_000_000))


# ---------------------------------------------------------------------------
# Accounts. Real keys come from files api/keys.py already knows how to produce.
# We keep the full dict (not just secrets) because we need `owner` to route
# accounts to the right profile — api.client.load_keys() strips that, so we
# read the file ourselves instead.
# ---------------------------------------------------------------------------

def load_key_files(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text()))
    return rows


def make_stub_keys(rng: random.Random, owner: str, count: int, tier: str = "free") -> list[dict]:
    """Clearly-fake local keys so this script can be tested before real ones exist."""
    out = []
    for _ in range(count):
        n = rng.randint(10000, 99999)
        out.append({"api_key_id": f"k_stub_{n}", "secret": f"secret_stub_{n}",
                    "tier": tier, "owner": owner})
    return out


def build_accounts(mix: list[dict], rng: random.Random, keys_files: list[str],
                    use_stub: bool):
    real_keys = load_key_files(keys_files) if keys_files else None
    accounts_by_profile = {}
    for entry in mix:
        profile, count, owner = entry["profile"], entry["accounts"], entry.get("owner", "casual")
        if real_keys is not None:
            pool = [k for k in real_keys if k["owner"] == owner]
            if len(pool) < count:
                raise SystemExit(
                    f"keys files have only {len(pool)} accounts with owner={owner!r}, "
                    f"need {count} for profile={profile!r}. Provision more with:\n"
                    f"  python -m api.keys --count {count} --tier free --owner {owner} "
                    f"--out data/{owner}_keys.json"
                )
            accounts_by_profile[profile] = rng.sample(pool, count)
        else:
            if not use_stub:
                raise SystemExit("No --keys-file given. Pass --stub-keys to test with "
                                  "throwaway local keys, or provide real ones from "
                                  "`python -m api.keys ...`.")
            print(f"[stub-keys] generating {count} throwaway accounts, owner={owner}")
            accounts_by_profile[profile] = make_stub_keys(rng, owner, count)
    return accounts_by_profile


# ---------------------------------------------------------------------------
# Assigning simulated IPs. Confirmed working (P3): api/main.py trusts
# X-Forwarded-For, api/client.py's predict(ip=...) sets it, no API change needed.
#
# FIXED (per P3's review): every individual account now gets its OWN IP, not one
# shared IP per profile — two researchers sharing an IP by accident isn't
# realistic and wasn't intentional. The ONE deliberate exception is `multitenant`:
# every account in that group shares a single IP on purpose, since "many accounts,
# one connection" is the entire point of that test.
# ---------------------------------------------------------------------------

OFFICE_IP = "203.0.113.7"  # matches the example in api/client.py's own docstring


def assign_ips(accounts_by_profile: dict, rng: random.Random) -> dict:
    """Returns {api_key_id: ip}. Every account gets its own IP except multitenant."""
    ip_by_key = {}
    used = {OFFICE_IP}
    for profile, accounts in accounts_by_profile.items():
        if profile == "multitenant":
            for account in accounts:
                ip_by_key[account["api_key_id"]] = OFFICE_IP
            continue
        for account in accounts:
            while True:
                candidate = f"198.51.100.{rng.randint(2, 254)}"
                if candidate not in used:
                    used.add(candidate)
                    break
            ip_by_key[account["api_key_id"]] = candidate
    return ip_by_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_events(cfg: dict, accounts_by_profile: dict, rng: random.Random):
    events = []
    for entry in cfg["mix"]:
        profile = entry["profile"]
        accounts = accounts_by_profile[profile]
        sub_rng = random.Random(rng.randint(0, 1_000_000))
        if profile == "multitenant":
            gen = office(sub_rng, accounts, duration_s=cfg["duration_seconds"],
                         **entry.get("params", {}))
        else:
            gen = PROFILES[profile](sub_rng, accounts, duration_s=cfg["duration_seconds"],
                                     **entry.get("params", {}))
        events.extend((profile, acc, t, w) for acc, t, w in gen)
    events.sort(key=lambda e: e[2])
    return events


def main():
    ap = argparse.ArgumentParser(description="Run a traffic scenario against the victim API.")
    ap.add_argument("scenario_file")
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--keys-file", action="append", default=[],
                     help="real keys JSON from `python -m api.keys`; can repeat")
    ap.add_argument("--stub-keys", action="store_true")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scenario_path = Path(args.scenario_file)
    if not scenario_path.exists():
        raise SystemExit(f"Scenario file not found: {scenario_path}")
    try:
        with open(scenario_path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SystemExit(f"Couldn't parse {scenario_path} as YAML: {e}")
    if not isinstance(cfg, dict):
        raise SystemExit(f"{scenario_path} doesn't look like a scenario file")
    for required in ("seed", "duration_seconds", "mix"):
        if required not in cfg:
            raise SystemExit(f"{scenario_path} is missing required field: {required!r}")

    api_url = args.api_url or cfg.get("api_url", "http://localhost:8000")
    speed = args.speed or cfg.get("speed", 50.0)
    seed = cfg["seed"]

    rng = random.Random(seed)
    accounts_by_profile = build_accounts(cfg["mix"], rng, args.keys_file, args.stub_keys)
    ip_by_key = assign_ips(accounts_by_profile, rng)
    events = build_events(cfg, accounts_by_profile, rng)

    print(f"Scenario: {cfg.get('run_label', args.scenario_file)}  seed={seed}  "
          f"events={len(events)}  speed={speed}x  api={api_url}")

    sent, ok, failed = 0, 0, 0
    sim_start = time.time()
    for profile, account, t_offset, weird in events:
        target_real_elapsed = t_offset / speed
        real_elapsed = time.time() - sim_start
        if target_real_elapsed > real_elapsed:
            time.sleep(target_real_elapsed - real_elapsed)

        img = image_for_event(rng, weird)
        ip = ip_by_key[account["api_key_id"]]

        if args.dry_run:
            print(f"[dry-run] t={t_offset:8.1f}s  profile={profile:10s} "
                  f"account={account['api_key_id']}  ip={ip}  weird={weird}")
            sent += 1
            continue

        status, body = predict(img, account["secret"], api_url, ip=ip)
        sent += 1
        if status != 200:
            failed += 1
            print(f"t={t_offset:8.1f}s  {account['api_key_id']:20s}  FAILED  "
                  f"status={status}  {body.get('detail')}")
        else:
            ok += 1

    print(f"\nDone. sent={sent} ok={ok} failed={failed}  "
          f"(scenario={args.scenario_file}, seed={seed})")


if __name__ == "__main__":
    main()
