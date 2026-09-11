#!/usr/bin/env bash
set -euo pipefail

# run_demo.sh — one-command demo entry point. Owned by P4 + P3.
#
# STATUS: sections 1-5 are wired up against real, confirmed code. Section 6
# (dashboard) is still TODO, pending an answer from P5.
#
# Usage:
#   ./run_demo.sh                     (defaults to a calibration/tuning run)
#   ./run_demo.sh eval_seed2          (runs the real evaluation instead)
#   DEFENSE=true ./run_demo.sh eval_seed2   (same run, WITH Stage 7 defense on)
#
# Run it twice — once without DEFENSE, once with DEFENSE=true, same RUN_ID —
# to get the undefended-vs-defended comparison P3 asked for. Results from
# each land in separate files so they don't overwrite each other.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Defaults to a calibration run on purpose (per P3's review): nobody should
# accidentally write into the reporting run just by running this to check it
# works. Pass eval_seed2 explicitly when you actually mean it.
RUN_ID="${1:-cal_seed1}"
DEFENSE="${DEFENSE:-false}"
HOST="0.0.0.0"
PORT="8000"
API_URL="http://127.0.0.1:${PORT}"
RESULTS_TAG="${RUN_ID}_defense-${DEFENSE}"

if [[ "$RUN_ID" == cal_* ]]; then
    SCENARIO_FILE="traffic/scenarios/calibration_seed1.yaml"
else
    SCENARIO_FILE="traffic/scenarios/evaluation_seed2.yaml"
fi

echo "=========================================================="
echo " Ledger demo — run_id=${RUN_ID}  defense=${DEFENSE}"
echo "=========================================================="

# ---------------------------------------------------------------- cleanup
BG_PIDS=()
cleanup() {
    echo ""
    echo "Shutting down background processes..."
    for pid in "${BG_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ---------------------------------------------------------- 1. start API
echo "[1/6] Starting victim API (run_id=${RUN_ID}, defense=${DEFENSE})..."
LEDGER_RUN_ID="$RUN_ID" LEDGER_DEFENSE_ENABLED="$DEFENSE" \
    python -m uvicorn api.main:app --host "$HOST" --port "$PORT" \
    > "demo_api_${RESULTS_TAG}.log" 2>&1 &
API_PID=$!
BG_PIDS+=("$API_PID")

echo -n "      waiting for /health"
UP=0
for _ in $(seq 1 30); do
    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        UP=1
        break
    fi
    echo -n "."
    sleep 0.5
done
echo ""
if [[ "$UP" -ne 1 ]]; then
    echo "ERROR: API never came up. Check demo_api_${RESULTS_TAG}.log for what went wrong."
    exit 1
fi
echo "      API is up."

# ---------------------------------------------------- 2. provision keys
echo "[2/6] Checking/provisioning test accounts (skips any that already exist)..."
mkdir -p data eval/results

provision_if_missing() {
    local owner="$1" count="$2" out="$3"
    if [[ -f "$out" ]]; then
        echo "      $out already exists, skipping"
    else
        echo "      provisioning $count '$owner' accounts -> $out"
        python -m api.keys --count "$count" --tier free --owner "$owner" --out "$out"
    fi
}

provision_if_missing casual      5  data/casual_keys.json
provision_if_missing batch       1  data/batch_keys.json
provision_if_missing bursty      2  data/bursty_keys.json
provision_if_missing researcher  3  data/researcher_keys.json
provision_if_missing office_acme 60 data/office_keys.json

# Attacker keys, owner=attacker_pool, matching api/keys.py's own docstring
# example and the account counts P1 actually reported seeing (400 for the
# distributed attack, 1 for knockoff's single-key baseline).
provision_if_missing attacker_pool 400 data/attacker_keys.json
provision_if_missing attacker_pool 1   data/knockoff_key.json

# ---------------------------------------------------- 3. honest traffic
echo "[3/6] Sending honest traffic (${SCENARIO_FILE})..."
python traffic/scenario.py "$SCENARIO_FILE" \
    --api-url "$API_URL" \
    --keys-file data/casual_keys.json \
    --keys-file data/batch_keys.json \
    --keys-file data/bursty_keys.json \
    --keys-file data/researcher_keys.json \
    --keys-file data/office_keys.json &
TRAFFIC_PID=$!
BG_PIDS+=("$TRAFFIC_PID")

# ---------------------------------------------------- 4. attacker traffic
echo "[4/6] Sending attacker traffic (knockoff + distributed, run alongside honest)..."
python -m attack.knockoff --keys data/knockoff_key.json --budget 6000 &
KNOCKOFF_PID=$!
BG_PIDS+=("$KNOCKOFF_PID")

python -m attack.distributed --keys data/attacker_keys.json --budget 20000 --spread-ip &
DISTRIBUTED_PID=$!
BG_PIDS+=("$DISTRIBUTED_PID")

echo ""
echo "Waiting for all traffic (honest + attack) to finish sending..."
wait "$TRAFFIC_PID"
wait "$KNOCKOFF_PID"
wait "$DISTRIBUTED_PID"
echo "All traffic done. API log: demo_api_${RESULTS_TAG}.log"

# ---------------------------------------------------------- 5. detector
echo "[5/6] Running detector against the finished log (run_id=${RUN_ID})..."
python -m detector.tier1_identity --db data/ledger.db --run-id "$RUN_ID"
python -m detector.tier3_ledger   --db data/ledger.db --run-id "$RUN_ID" \
    --threshold 0.30 --out "eval/results/${RESULTS_TAG}.json"
echo "      Results written to eval/results/${RESULTS_TAG}.json"

# ---------------------------------------------------------------- 6. TODO(P5)
echo "[6/6] Dashboard: NOT WIRED UP YET."
echo "      Need from P5: is dashboard/index.html a static file we just open in a"
echo "      browser, or does it need its own local server (and if so, what command)?"

echo ""
echo "Demo run complete for run_id=${RUN_ID}, defense=${DEFENSE}."
echo "Results: eval/results/${RESULTS_TAG}.json"
echo "API is still running at ${API_URL} — check ${API_URL}/stats and ${API_URL}/health."
echo "Press Ctrl+C to stop it, or run this script again with DEFENSE flipped for comparison."
wait "$API_PID"
