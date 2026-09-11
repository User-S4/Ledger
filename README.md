# TechFirm - ZeroTrace - Ledger

Ledger: Multi-tenant, distributed model theft detection and defense architecture.

A pretrained image classifier sits behind a web API. Most model-extraction defenses
watch individual accounts for suspicious behavior, and they fall apart the moment an
attacker splits their queries across many accounts. Ledger instead tracks how much of
the model's total knowledge has been revealed across all traffic combined,
regardless of which account asked. Splitting an attack across accounts doesn't help,
since the same information still gets revealed either way.

**Project Status:** Production-ready reference implementation. All 10 stages (model serving, latent projection, SQLite WAL logging, realistic multi-tenant traffic, 400-key distributed attack simulation, clone distillation, Stage 7 active decision-boundary poisoning, and live interactive SOC dashboard) are complete, fully automated, and verified by 71 unit and integration tests.

---

## Project Structure

```
ledger/
├── README.md                        ← Project overview & documentation
├── run_demo.sh                      ← End-to-end automated demo runner (calibration / evaluation)
├── requirements.txt                 ← Pinned project dependencies
├── config.yaml                      ← Global thresholds, seeds, and tier limits
├── SCHEMA.md                        ← Frozen database log contract
│
├── victim/                          ← Target model & representation space
│   ├── loader.py                    Pretrained CIFAR-10 ResNet-20 model (PyTorch & stub)
│   └── embed.py                     Whitened PCA projection (64-dim -> 32-dim space)
├── api/                             ← Platform service & live defense
│   ├── main.py                      FastAPI service (/predict, /stats, /health)
│   ├── logstore.py                  Append-only SQLite storage, WAL mode, timing analysis
│   ├── defense.py                   Stage 7 active degradation & probability coarsening
│   ├── keys.py                      API key registry, rate limiters, and provisioning
│   └── validate.py                  Image validation and schema integrity
├── detector/                        ← Multi-tiered model extraction defense
│   ├── tier1_identity.py            Tier 1: Identity, rate utilisation, and subnet clustering
│   ├── tier2_perclient.py           Tier 2: Per-account behavioral profiling (entropy, low conf)
│   ├── tier3_ledger.py              Tier 3: Global coverage ledger & spatial suspicion scoring
│   ├── cell_index.py                Live in-memory hypercube spatial indexing
│   ├── cells.py                     Embedding spatial bucketing & hash assignment
│   └── attribution.py               Spike-window attribution to coordinated account pools
├── attack/                          ← Extraction adversary simulation
│   ├── knockoff.py                  Single-account high-volume extraction baseline
│   ├── distributed.py               Distributed extraction across 400 keys & rotating IPs
│   ├── mixed.py                     Interleaved attack and honest traffic
│   ├── degrade_experiments.py       Offline degradation & targeted boundary poisoning
│   ├── session.py                   Shared request runner with delay & jitter pacing
│   ├── train_clone.py               Student model knowledge distillation
│   └── fidelity.py                  Fidelity, accuracy, and theft verification
├── traffic/                         ← Realistic client traffic simulation
│   ├── profiles.py                  Personas: casual, batch, bursty, researcher
│   ├── multitenant.py               60-account office behind one NAT IP (critical false-positive test)
│   ├── scenario.py                  Seeded, replayable traffic orchestrator
│   └── scenarios/
│       ├── calibration_seed1.yaml   Tuning only
│       └── evaluation_seed2.yaml    Reporting only
├── eval/                            ← Evaluation benchmarks & reporting
│   ├── time_criteria.py             Rate-bypass frontier & Tier 3 temporal invariance
│   ├── tune_thresholds.py           Grid search & sensitivity tuning on calibration logs
│   ├── metrics.py                   Ground-truth recall, precision, and FPR scoring
│   └── results/                     Structured benchmark artifacts (JSON)
├── dashboard/                       ← Interactive web dashboard (Chart.js)
├── docs/                            ← Submission materials & reports
└── tests/                           ← Automated verification suite (71 tests)
    ├── test_robustness.py           API resilience, concurrency, and validation
    ├── test_reproducibility.py      Determinism across seeds and state isolation
    └── test_time_criteria.py        Timing interval, rate-bypass, and invariance tests
```
---

## Requirements

- Python 3.11+ (needed for `list[dict]`-style type hints used throughout)
- Everything in `requirements.txt` (FastAPI, uvicorn, PyYAML, numpy, Pillow, torch,
  requests, and a few others; pinned versions, see that file)

Install with:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running it

### One-Command Automated Demo

```bash
# 1. Official Evaluation Benchmark (Undefended Baseline)
./run_demo.sh eval_seed2

# 2. Official Evaluation Benchmark (With Active Defense / Boundary Poisoning)
DEFENSE=true ./run_demo.sh eval_seed2

# 3. Tuning / Calibration Run (Default, uses calibration_seed1.yaml)
./run_demo.sh
```

> [!TIP]
> **Proving the Defense:** Run the command once without defense (`./run_demo.sh eval_seed2`) and once with active defense (`DEFENSE=true ./run_demo.sh eval_seed2`). Forensic detection reports are saved to `eval/results/eval_seed2_defense-false.json` and `eval/results/eval_seed2_defense-true.json` (demonstrating 99.5% attacker detection with 0% false alarms). Active decision-boundary poisoning (`swap_top2`) collapses stolen clone fidelity from **~88% down to ~41%** (benchmarked in `eval/results/stage7_surrogate_defended.json`).

---

### Manual Component-by-Component Walkthrough

To inspect, debug, or execute individual modules by hand:

### 1. Start the API

For the official evaluation benchmark (undefended baseline):
```bash
LEDGER_RUN_ID=eval_seed2 LEDGER_DEFENSE_ENABLED=false \
    LEDGER_RATE_LIMIT_MULTIPLIER=150 LEDGER_CLEAN_RUN=true \
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

To enable active decision-boundary poisoning defense (Stage 7):
```bash
LEDGER_RUN_ID=eval_seed2 LEDGER_DEFENSE_ENABLED=true \
    LEDGER_RATE_LIMIT_MULTIPLIER=150 LEDGER_CLEAN_RUN=true \
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For a default calibration/tuning run:
```bash
LEDGER_RATE_LIMIT_MULTIPLIER=150 LEDGER_CLEAN_RUN=true \
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> [!NOTE]
> - `LEDGER_RATE_LIMIT_MULTIPLIER=150` scales tier rate limits to accommodate accelerated simulation traffic (`speed: 50x`) and single-key knockoff query bursts without hitting HTTP 429 throttles.
> - `LEDGER_CLEAN_RUN=true` purges stale records from `data/ledger.db` for the target `run_id` so previous runs do not pollute fresh evaluation metrics.

Wait for `Application startup complete` in the terminal. Check it's alive and verify
which `run_id` and defense mode it is actively serving:
```bash
curl http://127.0.0.1:8000/health
```
You should get back a JSON response confirming `status: "ok"`, `reportable: true`, and the active `run_id`.

### 2. Provision test accounts

Keys are written straight into the same database the API reads from, no separate
setup step needed beyond running this:

```bash
# Honest client personas (casual, batch, bursty, researcher, multi-tenant corporate NAT)
python -m api.keys --count 5   --tier free --owner casual               --out data/casual_keys.json
python -m api.keys --count 1   --tier free --owner batch                --out data/batch_keys.json
python -m api.keys --count 2   --tier free --owner bursty               --out data/bursty_keys.json
python -m api.keys --count 3   --tier free --owner researcher           --out data/researcher_keys.json
python -m api.keys --count 60  --tier free --owner office_acme          --out data/office_keys.json

# Adversary accounts (single-account knockoff and 400-account distributed campaign)
python -m api.keys --count 1   --tier free --owner knockoff_attacker    --out data/knockoff_key.json
python -m api.keys --count 400 --tier free --owner distributed_attacker --out data/attacker_keys.json
```

(These counts match what `traffic/scenarios/*.yaml` and `attack/*.py` expect. If those files change, update the counts here to match.)

### 3. Send honest traffic

```bash
python -m traffic.scenario traffic/scenarios/evaluation_seed2.yaml \
    --api-url http://127.0.0.1:8000 \
    --keys-file data/casual_keys.json \
    --keys-file data/batch_keys.json \
    --keys-file data/bursty_keys.json \
    --keys-file data/researcher_keys.json \
    --keys-file data/office_keys.json
```

You'll see a running count of sent/ok/failed requests, ending in a summary line. Use
`traffic/scenarios/calibration_seed1.yaml` instead if you're tuning detector thresholds
rather than reporting final numbers. Refer to `SCHEMA.md`'s run_id rule for why that
distinction matters.

To see what would be sent without hitting the network first, add `--dry-run` to
the command above.

### 4. Send attacker traffic (run alongside step 3, not instead of it)

In a separate terminal (or run concurrently in the background):

```bash
python -m attack.knockoff    --keys data/knockoff_key.json  --budget 6000 &
python -m attack.distributed --keys data/attacker_keys.json --budget 20000 --spread-ip &
```

These simulate the actual theft attempts the detector needs to catch: `knockoff`
hammers the API from a single account, `distributed` spreads the same volume across
400 accounts (`--spread-ip` also varies the source IP per account). This is the exact case Tier-3's
global coverage tracking exists to catch, even when Tier-2's per-account monitoring misses it.
Budgets generate the attack query datasets in `attack/data/` (`knockoff_attacker_6000.npz`, `distributed_attacker_20000.npz`) during execution.

> [!TIP]
> Add `--epochs 0` to either attack command if you want to deliver queries and test detection without waiting for the full 60-epoch student clone distillation training on CPU.

### 5. Check what landed

```bash
curl http://127.0.0.1:8000/stats
```

### 6. Run forensic detectors

Run the offline forensic analysis against the database log after traffic completes:

```bash
python -m detector.tier1_identity --db data/ledger.db --run-id eval_seed2
python -m detector.tier3_ledger   --db data/ledger.db --run-id eval_seed2 \
    --threshold 0.30 --out eval/results/eval_seed2.json
```

`tier1_identity` (conventional rate/IP monitoring) and `tier3_ledger` (global manifold coverage ledger) are the two primary detection entry points. Both accept `--db` and `--run-id`. The `--out` flag on `tier3_ledger` writes full structured forensic results (per-account scores, confusion matrix, precision/recall, and owner breakdown) directly into `eval/results/` as JSON.

---

## Interactive Security Operations Center (Dashboard)

The real-time ZeroTrace SOC Dashboard is served directly by the victim API:

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

## Testing

```bash
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
