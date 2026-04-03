#!/bin/bash
set -euo pipefail

# state-transition.sh — Atomically transition FSM state in loop-state.json
#
# Usage: state-transition.sh <NEXT_STATE> [key=value ...]
#
# Examples:
#   state-transition.sh IDEATE
#   state-transition.sh STORE decision_made=true this_iter_tflops=450.2
#   state-transition.sh INIT shape_key=f16_4096x16384x16384 dtype=f16 mode=from_current_best max_iteration=30
#
# The script:
#   1. Validates the transition is legal (from current state to next state)
#   2. Resets guard flags appropriate for the new state
#   3. Applies any key=value overrides to metrics/fsm
#   4. Writes atomically (temp file + mv)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTUNER_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="${CROQTUNER_STATE_FILE:-$CROQTUNER_DIR/state/loop-state.json}"

if [ $# -lt 1 ]; then
    echo "Usage: state-transition.sh <NEXT_STATE> [key=value ...]"
    echo "States: INIT BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE NEXT_SHAPE"
    exit 1
fi

NEXT_STATE="$1"
shift

VALID_STATES="INIT BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE NEXT_SHAPE"
if ! echo "$VALID_STATES" | grep -qw "$NEXT_STATE"; then
    echo "ERROR: Invalid state '$NEXT_STATE'. Valid states: $VALID_STATES"
    exit 1
fi

# Legal transitions (from → allowed next states)
declare -A LEGAL_TRANSITIONS
LEGAL_TRANSITIONS=(
    ["INIT"]="BASELINE"
    ["BASELINE"]="PROFILE"
    ["PROFILE"]="IDEATE"
    ["IDEATE"]="IMPLEMENT"
    ["IMPLEMENT"]="MEASURE STORE"
    ["MEASURE"]="DECIDE STORE"
    ["DECIDE"]="STORE"
    ["STORE"]="PROFILE SHAPE_COMPLETE"
    ["SHAPE_COMPLETE"]="NEXT_SHAPE"
    ["NEXT_SHAPE"]="INIT"
    ["_NEW_"]="INIT"
)

# If state file doesn't exist, only INIT is valid
if [ ! -f "$STATE_FILE" ]; then
    if [ "$NEXT_STATE" != "INIT" ]; then
        echo "ERROR: No loop-state.json found. First transition must be INIT."
        exit 1
    fi

    mkdir -p "$(dirname "$STATE_FILE")"

    # Create initial state
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    cat > "$STATE_FILE" <<INITJSON
{
  "schema_version": 1,
  "fsm": {
    "current_state": "INIT",
    "iteration": 0,
    "max_iteration": 30,
    "shape_key": "",
    "dtype": "",
    "shape": [],
    "mode": "from_current_best"
  },
  "guard_flags": {
    "gpu_health_checked": false,
    "ncu_ran_this_iter": false,
    "bottleneck_identified": false,
    "idea_is_novel": false,
    "idea_logged": false,
    "compile_succeeded": false,
    "correctness_verified": false,
    "timing_captured": false,
    "decision_made": false,
    "results_appended": false,
    "checkpoint_written": false,
    "git_committed": false
  },
  "metrics": {
    "baseline_tflops": null,
    "current_best_tflops": null,
    "current_best_iter": null,
    "current_best_kernel": null,
    "this_iter_tflops": null,
    "this_iter_decision": null,
    "consecutive_discards": 0,
    "last_bottleneck": null,
    "last_idea_category": null
  },
  "paths": {
    "shape_dir_logs": "",
    "shape_dir_srcs": "",
    "shape_dir_perf": "",
    "checkpoint_file": "",
    "idea_log": ""
  },
  "completion_promise": "iteration >= max_iteration AND best kernel registered AND state.json updated",
  "last_updated": "$TIMESTAMP"
}
INITJSON

    echo "OK: Created initial loop-state.json in state INIT"
    # Apply any key=value overrides
    for kv in "$@"; do
        KEY="${kv%%=*}"
        VAL="${kv#*=}"
        case "$KEY" in
            shape_key)
                jq --arg v "$VAL" '.fsm.shape_key = $v | .paths.shape_dir_logs = "tuning/logs/\($v)" | .paths.shape_dir_srcs = "tuning/srcs/\($v)" | .paths.shape_dir_perf = "tuning/perf/\($v)" | .paths.checkpoint_file = "tuning/checkpoints/\($v).json" | .paths.idea_log = "tuning/logs/\($v)/idea-log.jsonl"' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
            dtype)
                jq --arg v "$VAL" '.fsm.dtype = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
            mode)
                jq --arg v "$VAL" '.fsm.mode = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
            max_iteration)
                jq --argjson v "$VAL" '.fsm.max_iteration = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
            shape)
                jq --argjson v "$VAL" '.fsm.shape = $v' "$STATE_FILE" > "${STATE_FILE}.tmp"
                mv "${STATE_FILE}.tmp" "$STATE_FILE"
                ;;
        esac
    done
    exit 0
fi

# Read current state
CURRENT_STATE=$(jq -r '.fsm.current_state' "$STATE_FILE")

# Validate transition
ALLOWED="${LEGAL_TRANSITIONS[$CURRENT_STATE]:-}"
if [ -z "$ALLOWED" ]; then
    echo "ERROR: Unknown current state '$CURRENT_STATE' in loop-state.json"
    exit 1
fi

if ! echo "$ALLOWED" | grep -qw "$NEXT_STATE"; then
    echo "ERROR: Illegal transition $CURRENT_STATE → $NEXT_STATE"
    echo "       Allowed from $CURRENT_STATE: $ALLOWED"
    exit 1
fi

# Special case: NEXT_SHAPE → INIT reinitializes the state file
if [ "$NEXT_STATE" = "INIT" ] && [ "$CURRENT_STATE" = "NEXT_SHAPE" ]; then
    rm -f "$STATE_FILE"
    # Re-run ourselves in "no state file" mode, passing through all arguments
    exec "$0" INIT "$@"
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build jq expression for the transition
JQ_EXPR=".fsm.current_state = \"$NEXT_STATE\" | .last_updated = \"$TIMESTAMP\""

# Reset guard flags based on which state we're entering
case "$NEXT_STATE" in
    PROFILE)
        # Auto-increment iteration when entering PROFILE
        # From BASELINE: 0 → 1 (first real iteration)
        # From STORE: N → N+1 (next iteration)
        if [ "$CURRENT_STATE" = "BASELINE" ] || [ "$CURRENT_STATE" = "STORE" ]; then
            CURRENT_ITER=$(jq -r '.fsm.iteration' "$STATE_FILE")
            NEXT_ITER=$((CURRENT_ITER + 1))
            JQ_EXPR="$JQ_EXPR | .fsm.iteration = $NEXT_ITER"
        fi
        JQ_EXPR="$JQ_EXPR | .guard_flags.gpu_health_checked = false | .guard_flags.ncu_ran_this_iter = false | .guard_flags.bottleneck_identified = false | .guard_flags.idea_is_novel = false | .guard_flags.idea_logged = false | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false | .guard_flags.timing_captured = false | .guard_flags.decision_made = false | .guard_flags.results_appended = false | .guard_flags.checkpoint_written = false | .guard_flags.git_committed = false | .metrics.this_iter_tflops = null | .metrics.this_iter_decision = null"
        ;;
    IDEATE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.idea_is_novel = false | .guard_flags.idea_logged = false"
        ;;
    IMPLEMENT)
        JQ_EXPR="$JQ_EXPR | .guard_flags.compile_succeeded = false | .guard_flags.correctness_verified = false"
        ;;
    MEASURE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.timing_captured = false"
        ;;
    DECIDE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.decision_made = false"
        ;;
    STORE)
        JQ_EXPR="$JQ_EXPR | .guard_flags.results_appended = false | .guard_flags.checkpoint_written = false | .guard_flags.git_committed = false"
        ;;
esac

# Apply key=value overrides
for kv in "$@"; do
    KEY="${kv%%=*}"
    VAL="${kv#*=}"
    case "$KEY" in
        # Guard flags (boolean)
        gpu_health_checked|ncu_ran_this_iter|bottleneck_identified|idea_is_novel|idea_logged|compile_succeeded|correctness_verified|timing_captured|decision_made|results_appended|checkpoint_written|git_committed)
            JQ_EXPR="$JQ_EXPR | .guard_flags.$KEY = $VAL"
            ;;
        # Metrics (numeric or string)
        baseline_tflops|current_best_tflops|this_iter_tflops)
            JQ_EXPR="$JQ_EXPR | .metrics.$KEY = $VAL"
            ;;
        current_best_iter|consecutive_discards)
            JQ_EXPR="$JQ_EXPR | .metrics.$KEY = $VAL"
            ;;
        current_best_kernel|this_iter_decision|last_bottleneck|last_idea_category)
            JQ_EXPR="$JQ_EXPR | .metrics.$KEY = \"$VAL\""
            ;;
        # FSM fields
        iteration)
            JQ_EXPR="$JQ_EXPR | .fsm.iteration = $VAL"
            ;;
        shape_key)
            JQ_EXPR="$JQ_EXPR | .fsm.shape_key = \"$VAL\" | .paths.shape_dir_logs = \"tuning/logs/$VAL\" | .paths.shape_dir_srcs = \"tuning/srcs/$VAL\" | .paths.shape_dir_perf = \"tuning/perf/$VAL\" | .paths.checkpoint_file = \"tuning/checkpoints/$VAL.json\" | .paths.idea_log = \"tuning/logs/$VAL/idea-log.jsonl\""
            ;;
    esac
done

# Atomic write
jq "$JQ_EXPR" "$STATE_FILE" > "${STATE_FILE}.tmp"
mv "${STATE_FILE}.tmp" "$STATE_FILE"

echo "OK: $CURRENT_STATE → $NEXT_STATE"
