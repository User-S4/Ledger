"""Stage 2 acceptance check.

If this prints PASS on your machine, the log layer works and you can start
writing detector code against data/fake.db.

    python -m api.fake_logs --db data/fake.db --n 20000 --quiet
    python tests/check_stage2.py

Run it from the repo root, not from inside tests/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from api.logstore import LogStore  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else "data/fake.db"
RUN_ID = sys.argv[2] if len(sys.argv) > 2 else "cal_seed1"

EXPECTED_COLUMNS = [
    "request_id", "run_id", "ts", "api_key_id", "tier", "ip", "status_code",
    "error_code", "latency_ms", "input_sha256", "input_bytes", "embedding",
    "model_probs", "returned_probs", "degradation_level", "suspicion_score",
]

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


if not Path(DB).exists():
    sys.exit(f"{DB} not found. Run: python -m api.fake_logs --db {DB} --n 20000")

store = LogStore(DB)
df = store.read_df(run_id=RUN_ID)
ok = df[df.status_code == 200]
bad = df[df.status_code >= 400]

print(f"\nchecking {DB}  (run_id={RUN_ID})\n")

print("the contract")
check("16 columns, in the order SCHEMA.md specifies",
      list(df.columns) == EXPECTED_COLUMNS, str(list(df.columns)))
check("rows were actually written", len(df) > 0, f"{len(df)} rows")

print("\ncolumn 12 -- the point on the map")
e = ok.embedding.iloc[0]
check("decoded to a numpy array, not raw bytes", isinstance(e, np.ndarray),
      type(e).__name__)
check("32 numbers wide", getattr(e, "shape", None) == (32,), str(getattr(e, "shape", None)))
check("no missing points on successful requests", ok.embedding.notna().all())

print("\ncolumns 13 and 14 -- what the model believed, what the customer got")
mp = ok.model_probs.iloc[0]
check("decoded to a list, not a string", isinstance(mp, list), type(mp).__name__)
check("10 classes", len(mp) == 10 if isinstance(mp, list) else False)
check("probabilities sum to 1", abs(sum(mp) - 1.0) < 1e-6 if isinstance(mp, list) else False)
# Identical until Stage 7 exists. The gap between them IS the defence.
same = (ok.model_probs.map(str) == ok.returned_probs.map(str)).all()
check("identical before Stage 7 (they diverge once the fightback exists)", same)

print("\nfailed requests -- the rows that crash naive detector code")
check("some failures exist to test against", len(bad) > 0, f"{len(bad)} rows")
if len(bad):
    check("no point recorded on a failed request", bad.embedding.isna().all())
    check("every failure carries a reason", bad.error_code.notna().all())

print("\nthe schema refuses off-contract columns")
try:
    store.log(run_id="x", api_key_id="k", tier="free", ip="1.2.3.4",
              status_code=200, cell_id=7)
    check("writing an invented column raises", False, "it was accepted")
except KeyError as exc:
    check("writing an invented column raises", "cell_id" in str(exc))

print("\nground truth is available for scoring, and separate from the log")
owners = store.owners_df()
check("keys table has owner labels", "owner" in owners.columns)
check("owner is NOT a column in the log", "owner" not in df.columns)

store.close()

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) -- {', '.join(failures)}")
    sys.exit(1)
print(f"PASS -- {len(df)} rows, {ok.api_key_id.nunique()} accounts. "
       "The log layer works; detector work can start.")