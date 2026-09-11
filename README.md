# TechFirm - ZeroTrace - Ledger

Ledger: Multi-tenant, distributed model theft detection and defense architecture.

A pretrained image classifier sits behind a web API. Most model-extraction defenses
watch individual accounts for suspicious behavior, and they fall apart the moment an
attacker splits their queries across many accounts. Ledger instead tracks how much of
the model's total knowledge has been revealed across all traffic combined,
regardless of which account asked. Splitting an attack across accounts doesn't help,
since the same information still gets revealed either way.

**Project status: work in progress.** This README documents what's actually built and
tested right now, and marks clearly what isn't finished yet. See "Known gaps" below
before assuming something works.

---

## Project Structure

```
ledger/
├── README.md                        ← P4
├── requirements.txt                 ← P3 owns, pinned versions
├── run_demo.sh                      ← P4, one-command entry point (partial, see below)
├── config.yaml                      ← P3, thresholds and seeds
├── SCHEMA.md                        ← P3, frozen hour two
│
├── victim/                          ← P3 — pretrained CIFAR-10 classifier
├── api/                             ← P3 — FastAPI: /predict /stats /health
├── detector/                        ← P1 — the extraction detector
├── attack/                          ← P2 — knockoff / distributed / mixed attacks
├── traffic/                         ← P4 — honest traffic simulation
│   ├── profiles.py                  casual, batch, bursty, researcher
│   ├── multitenant.py               office behind one IP — key false-positive test
│   ├── scenario.py                  seeded, replayable traffic runner
│   └── scenarios/
│       ├── calibration_seed1.yaml   tuning only
│       └── evaluation_seed2.yaml    reporting only
├── dashboard/                       ← P5 — polls /stats, Chart.js
├── eval/                            ← P1 produces, P5 renders
├── docs/                            ← P5 — submission materials
└── tests/                           ← P3 and P4
```

---

## Requirements

- Python 3.11+ (needed for `list[dict]`-style type hints used throughout)
- Everything in `requirements.txt` (FastAPI, uvicorn, PyYAML, numpy, Pillow, torch,
  requests, and a few others; pinned versions, see that file)

Install with:
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## Running it

`./run_demo.sh` now runs the full pipeline automatically (API, keys, honest traffic,
attacker traffic, detector) except for the dashboard step, see "Known gaps." The
manual steps below are the same thing broken out by hand, useful for debugging or
running just one piece.

### 1. Start the API

For a tuning/calibration run (the safe default, matches `config.yaml`):
```
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For the real evaluation run, **set the run_id explicitly**. Otherwise it silently
defaults to `cal_seed1` and your evaluation traffic won't be tagged correctly:
```
LEDGER_RUN_ID=eval_seed2 python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete` in the terminal. Check it's alive and check
which run_id it's actually using:
```
curl http://127.0.0.1:8000/health
```
You should get back a small JSON response, including the `run_id` it's currently
logging under. Double-check that matches what you meant to run before sending any
traffic. If `reportable` is `false`, the real victim model / projection isn't loaded
yet; traffic will still flow, but results aren't meaningful until it is.

**Note on defense:** `config.yaml` has `defense.enabled: true` by default. Override
it per-run with `LEDGER_DEFENSE_ENABLED=false` or `=true` (confirmed working env var,
added by Person 3). This is how `run_demo.sh` produces the undefended-vs-defended
comparison rather than always using whatever `config.yaml` currently says.

### 2. Provision test accounts

Keys are written straight into the same database the API reads from, no separate
setup step needed beyond running this:

```
python -m api.keys --count 5  --tier free --owner casual      --out data/casual_keys.json
python -m api.keys --count 1  --tier free --owner batch       --out data/batch_keys.json
python -m api.keys --count 2  --tier free --owner bursty      --out data/bursty_keys.json
python -m api.keys --count 3  --tier free --owner researcher  --out data/researcher_keys.json
python -m api.keys --count 60 --tier free --owner office_acme --out data/office_keys.json
```

(These counts match what `traffic/scenarios/*.yaml` currently expect. If those files
change, update the counts here to match, or the traffic run below will fail with a
clear "not enough accounts" error telling you exactly what's short.)

### 3. Send honest traffic

```
cd traffic
python scenario.py scenarios/evaluation_seed2.yaml \
    --api-url http://127.0.0.1:8000 \
    --keys-file ../data/casual_keys.json \
    --keys-file ../data/batch_keys.json \
    --keys-file ../data/bursty_keys.json \
    --keys-file ../data/researcher_keys.json \
    --keys-file ../data/office_keys.json
```

You'll see a running count of sent/ok/failed requests, ending in a summary line. Use
`scenarios/calibration_seed1.yaml` instead if you're tuning detector thresholds
rather than reporting final numbers. Refer to `SCHEMA.md`'s run_id rule for why that
distinction matters.

To see what would be sent without hitting the network first, add `--dry-run` to
the command above.

### 4. Send attacker traffic (run alongside step 3, not instead of it)

```
python -m attack.knockoff    --keys data/knockoff_key.json  --budget 6000
python -m attack.distributed --keys data/attacker_keys.json --budget 20000 --spread-ip
```

These simulate the actual theft attempts the detector needs to catch: `knockoff`
hammers the API from a single account, `distributed` spreads the same volume across
many accounts (`--spread-ip` also varies the source IP per account). This is the exact case Tier-3's
global coverage tracking exists to catch, even when Tier-2's per-account monitoring would miss it. 
Budgets match the pre-generated attack data already in `attack/data/` (`knockoff_attacker_6000.npz`, `distributed_attacker_20000.npz`).

### 5. Check what landed

```
curl http://127.0.0.1:8000/stats
```

---

---

## Interactive Security Operations Center (Dashboard)

The real-time ZeroTrace SOC Dashboard (Person 5) is served directly by the victim API:

```
http://127.0.0.1:8000/dashboard
```

### Dashboard Features & Architecture

- **In-Browser Traffic Simulation Controls**:
  - **`▶ Stream Honest Traffic`**: Background simulation streaming realistic multi-tenant traffic across casual, batch, bursty, researcher, and corporate office profiles (corporate NAT IP `203.0.113.7`).
  - **`⚠️ Launch 400-Key Attack`**: Simulates the Stage 5 distributed theft campaign across 400 distinct accounts, sweeping the latent manifold boundary.
  - **`⏹ Stop Stream`**: Halts any active simulation thread.
  - **`🔄 Reset Run to 0`**: Generates a clean session run ID, flushes the spatial index, and resets counters.

- **Dynamic Threat Alert Banner**:
  - Automatically flips from `🛡️ SYSTEM SECURE — NORMAL TRAFFIC` to `🚨 DISTRIBUTED EXTRACTION DETECTED — ACTIVE FIGHTBACK` when global manifold coverage expansion, rapid window discoveries, and key dispersion thresholds are breached.
  - Displays real-time intercepted key counts.

- **Operational KPI Cards**:
  - **TOTAL INGESTION**: Total live API requests processed in the active session.
  - **ACTIVE KEY POOL**: Number of distinct client accounts actively tracked.
  - **THREAT POSTURE**: ZeroTrace Spatial Ledger status (`NOMINAL (SECURE)` in emerald green or `UNDER ATTACK` in pulsing red).
  - **DEFENSE INTERCEPTIONS**: Real-time running counter of adversarial queries intercepted and silently poisoned via `swap_top2`.

- **Live Streaming Telemetry Charts**:
  - **Cumulative Latent Manifold Coverage**: Visualizes total representation partitions mapped over time (demonstrating honest usage plateau vs. attack exploration surge).
  - **Discovery Velocity Stream**: Rolling window discovery rate highlighting coordinated boundary extraction spikes.
  - **Traffic Throughput (RPS)**: Real-time queries per second.

- **Live Request Ingestion Feed**:
  - Real-time audit log of every incoming query with timestamp, API key ID, IP, predicted class with top-1 confidence, latent cell ID, defense status (`CLEAN` vs. `POISONED`), and processing latency.

- **Suspect Attribution & Coordinated Keys**:
  - Real-time table isolating all accounts participating in the coordinated exploration swarm, tagged for silent poisoning.

- **Active Defense Switch**:
  - Header toggle allowing SOC operators to enable or bypass active boundary poisoning (`DEFENSE: ON (POISON)` vs. `DEFENSE: OFF (BYPASSED)`).

---

## Status & Completed Milestones

- **Pipeline Automation (`run_demo.sh`)**: End-to-end execution across victim startup, account provisioning, honest multi-tenant traffic, 400-key distributed attack, offline forensic analysis, and live dashboard serving.
- **Stage 7 Defense (Boundary Poisoning)**: Verified with Kaggle surrogate evaluation. Undefended 20k clone achieves 83.15% fidelity; active decision-boundary poisoning (`clean1000 + swap_top2`) degrades clone fidelity down to **41.72%** while preserving top-1 label correctness.
- **Stage 10 SOC Dashboard**: Fully dynamic, polling `/stats` and `/logs/recent` every second with zero static mockups, interactive simulation controls, and live request audit feed.

---

## Detector (run manually, after traffic finishes — it does not watch live)

```
python -m detector.tier1_identity --db data/ledger.db --run-id eval_seed2
python -m detector.tier3_ledger   --db data/ledger.db --run-id eval_seed2 \
    --threshold 0.30 --out eval/results/eval_seed2.json
```

`tier1_identity` and `tier3_ledger` are the two real entry points (confirmed by
Person 3) — both take `--db` and `--run-id`. The `--out` flag on `tier3_ledger`
writes full structured results (per-account scores, confusion matrix, owner
breakdown) straight into `eval/results/` as JSON. `attack/distributed.py` separately
writes its own fidelity results into `eval/results/stage5_distributed.json`
automatically, no flag needed.

---

## Testing

```
python -m pytest tests/
```

The test suite contains 71 automated tests across three specialized test files:

- `tests/test_robustness.py` checks that the API and LogStore survive malformed inputs, oversized payloads, decompression bombs, fuzzing, concurrent requests, and strict schema violations. It also verifies that degradation never alters top-1 labels.
- `tests/test_reproducibility.py` verifies determinism and non-corruption:
  - Running the same seed twice produces identical logs, embeddings, and detector scores.
  - Different seeds yield distinct datasets (validating the `cal_` vs. `eval_` firewall).
  - Run isolation, projection stability across reloads, and database state preservation across repeated executions.
- `tests/test_time_criteria.py` verifies the timing engine:
  - Calculation of per-account and global inter-query intervals (`delta_t`) in `LogStore`.
  - Traditional Tier 1 rate-bypass verification under spaced query pacing ($\Delta t \ge 1.5\text{s}$).
  - Mathematical temporal invariance of Tier 3 spatial coverage efficiency under slow-paced queries.
  - Attack session pacing and jitter mechanics.

