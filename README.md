# TechFirm - ZeroTrace - Ledger

Ledger: Multi-tenant, distributed model theft detection and defense architecture.

## Project Structure

```
ledger/
├── README.md                        ← P4
├── requirements.txt                 ← P3 owns, pinned versions
├── run_demo.sh                      ← P4, one-command entry point
├── config.yaml                      ← P3, thresholds and seeds
├── SCHEMA.md                        ← P3, frozen hour two
│
├── victim/                          ← P3
│   ├── loader.py                    pretrained CIFAR-10, no training
│   └── embed.py                     penultimate layer → PCA → 32-d
│
├── api/                             ← P3
│   ├── main.py                      FastAPI, /predict /stats /health
│   ├── keys.py                      per-key auth and tiers
│   ├── logstore.py                  SQLite, the frozen schema
│   ├── validate.py                  input robustness — the 25%
│   ├── defense.py                   graduated degradation
│   └── fake_logs.py                 P3 writes hour 3, P1 consumes
│
├── detector/                        ← P1
│   ├── tier1_identity.py            P3 writes, P1 integrates
│   ├── tier2_perclient.py           P1
│   ├── tier3_ledger.py              P1 — the innovation
│   ├── features.py                  P1
│   ├── cells.py                     P1 designs, P3 makes it fast
│   ├── sequential.py                P1, evidence accumulation
│   └── attribution.py               P1, which keys filled the map
│
├── attack/                          ← P2
│   ├── knockoff.py                  single-key baseline
│   ├── distributed.py               round-robin across many keys
│   ├── mixed.py                     distributed plus junk traffic
│   ├── collect.py                   query and record responses
│   ├── train_clone.py               importable, notebook only runs it
│   └── fidelity.py                  agreement on held-out data
│
├── traffic/                         ← P4
│   ├── profiles.py                  casual, batch, bursty, researcher
│   ├── multitenant.py               office behind one IP — key test
│   ├── scenario.py                  seeded, replayable
│   └── scenarios/
│       ├── calibration_seed1.yaml   tuning only
│       └── evaluation_seed2.yaml    reporting only
│
├── dashboard/                       ← P5
│   ├── index.html
│   └── app.js                       polls /stats, Chart.js
│
├── eval/                            ← P1 produces, P5 renders
│   ├── metrics.py                   detection rate, FPR, leakage
│   ├── make_figures.py              ← P5
│   ├── results/                     JSON, single source of truth
│   └── figures/
│
├── docs/                            ← P5
│   ├── submission.pdf
│   ├── business_framing.md          start Day 1
│   └── demo_script.md
│
└── tests/                           ← P3 and P4
    ├── test_robustness.py           malformed input
    └── test_reproducibility.py      second run doesn't corrupt
```
