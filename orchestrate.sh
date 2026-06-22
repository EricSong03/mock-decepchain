#!/usr/bin/env bash
# Autonomous DecepChain pipeline driver (overnight, unattended).
# - Self-daemonizes (setsid) so it survives terminal/session disconnect.
# - Runs Stage1 -> Stage2 -> Stage3 -> Eval in order.
# - Gates on artifacts (not just process exit); a partial/crashed stage is retried.
# - Training stages auto-resume from their last checkpoint (sft.py/grpo.py), so a GPU
#   maintenance cut is cheap: relaunch the same command and it resumes.
# - Detects an already-running stage process and WAITS instead of duplicating it.
# - Idempotent: re-running skips any stage whose gate is already satisfied.
set -uo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
ROOT="$(dirname "$SELF")"

# ---- self-daemonize: fully detach from the launching shell/session ----
if [ "${DECEP_DAEMON:-}" != "1" ]; then
  export DECEP_DAEMON=1
  setsid bash "$SELF" >> "$ROOT/orchestrator.boot.log" 2>&1 < /dev/null &
  echo "orchestrator daemonized (pid $!); logs -> orchestrator.log / orchestrator.status"
  exit 0
fi

cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0

OLOG="$ROOT/orchestrator.log"
STATUS="$ROOT/orchestrator.status"
MAX_TRIES=8
BACKOFF=90

log(){ echo "[$(date '+%F %T')] $*" >> "$OLOG"; }
setstatus(){ echo "$(date '+%F %T') | $*" > "$STATUS"; }

# Wait for any process matching $1 (excluding this script) to exit.
wait_for_proc(){
  local pat="$1"
  if pgrep -af "$pat" | grep -v "orchestrate.sh" >/dev/null 2>&1; then
    log "  -> a process matching '$pat' is already running; waiting for it to finish"
    while pgrep -af "$pat" | grep -v "orchestrate.sh" >/dev/null 2>&1; do sleep 30; done
    log "  -> '$pat' finished"
  fi
}

# run_stage NAME GATE_CMD PROC_PAT LOGFILE LAUNCH_CMD
run_stage(){
  local name="$1" gate="$2" pat="$3" logf="$4" launch="$5"
  if eval "$gate"; then log "STAGE $name: gate already satisfied, skipping"; setstatus "$name DONE (skipped)"; return 0; fi
  local try=0
  while (( try < MAX_TRIES )); do
    wait_for_proc "$pat"
    if eval "$gate"; then log "STAGE $name: gate satisfied after waiting"; setstatus "$name DONE"; return 0; fi
    try=$((try+1))
    setstatus "$name RUNNING (attempt $try/$MAX_TRIES)"
    log "STAGE $name: launching attempt $try/$MAX_TRIES"
    echo "==== $(date '+%F %T') orchestrator launch attempt $try ====" >> "$logf"
    eval "$launch" >> "$logf" 2>&1
    local rc=$?
    log "STAGE $name: command exited rc=$rc"
    if eval "$gate"; then log "STAGE $name: gate satisfied (rc=$rc)"; setstatus "$name DONE"; return 0; fi
    log "STAGE $name: gate NOT satisfied; sleeping ${BACKOFF}s then retry (resume if applicable)"
    sleep "$BACKOFF"
  done
  log "STAGE $name: FAILED after $MAX_TRIES attempts"
  setstatus "$name FAILED"
  return 1
}

log "===== orchestrator starting (pid $$) ====="
setstatus "STARTING"

run_stage "stage1" \
  '[ -f data/sft/D_s.jsonl ] && [ "$(wc -l < data/sft/D_s.jsonl)" -ge 1000 ]' \
  'src\.data\.build_sft_set' \
  "$ROOT/stage1.log" \
  'bash scripts/run_stage1.sh' || { log "ABORT: stage1 failed"; setstatus "ABORTED at stage1"; exit 1; }
log "Stage 1 gate: D_s.jsonl has $(wc -l < data/sft/D_s.jsonl) rows"

run_stage "stage2" \
  '[ -f checkpoints/stage2_sft/adapter_model.safetensors ]' \
  'src\.train\.sft' \
  "$ROOT/stage2.log" \
  'bash scripts/run_stage2.sh' || { log "ABORT: stage2 failed"; setstatus "ABORTED at stage2"; exit 1; }

run_stage "stage3" \
  '[ -f checkpoints/stage3_grpo/adapter_model.safetensors ]' \
  'src\.train\.grpo' \
  "$ROOT/stage3.log" \
  'bash scripts/run_stage3.sh --config configs/stage3_grpo_gsm8k.yaml' || { log "ABORT: stage3 failed"; setstatus "ABORTED at stage3"; exit 1; }

run_stage "eval" \
  '[ -f runs/eval/results.json ] && grep -q post_grpo runs/eval/results.json' \
  'src\.eval\.evaluate' \
  "$ROOT/eval.log" \
  'bash scripts/run_eval.sh' || { log "ABORT: eval failed"; setstatus "ABORTED at eval"; exit 1; }

log "===== PIPELINE COMPLETE ====="
setstatus "ALL DONE"
