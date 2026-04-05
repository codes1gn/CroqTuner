#!/bin/bash
set -euo pipefail

# post-step-check.sh — Validate postconditions after each FSM step
#
# Usage: post-step-check.sh <STEP_NAME>
# Returns 0 if all postconditions met, non-zero with diagnostic message if not.
#
# This is the "PostToolUse hook" equivalent for the CroqTuner skill.
# The LLM MUST run this after each step. If it exits non-zero, the step is INCOMPLETE.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTUNER_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="${CROQTUNER_STATE_FILE:-$CROQTUNER_DIR/state/loop-state.json}"

if [ $# -lt 1 ]; then
    echo "Usage: post-step-check.sh <STEP_NAME>"
    echo "Steps: INIT BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE"
    exit 1
fi

STEP="$1"
ERRORS=0

err() { echo "INCOMPLETE: $1"; ERRORS=$((ERRORS + 1)); }

if [ ! -f "$STATE_FILE" ]; then
    err "loop-state.json not found."
    exit 1
fi

SHAPE_KEY=$(jq -r '.fsm.shape_key' "$STATE_FILE")
ITERATION=$(jq -r '.fsm.iteration' "$STATE_FILE")
MAX_ITER=$(jq -r '.fsm.max_iteration' "$STATE_FILE")
MODE=$(jq -r '.fsm.mode' "$STATE_FILE")

case "$STEP" in
    INIT)
        # Verify directories exist
        LOGS_DIR=$(jq -r '.paths.shape_dir_logs' "$STATE_FILE")
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        PERF_DIR=$(jq -r '.paths.shape_dir_perf' "$STATE_FILE")

        [ -d "$LOGS_DIR" ] || err "Logs directory not created: $LOGS_DIR"
        [ -d "$SRCS_DIR" ] || err "Srcs directory not created: $SRCS_DIR"
        [ -d "$PERF_DIR" ] || err "Perf directory not created: $PERF_DIR"

        # Verify seed kernel
        if [ "$MODE" = "from_current_best" ]; then
            [ -f "$SRCS_DIR/seed.cu" ] || err "Seed kernel not found: $SRCS_DIR/seed.cu"
        else
            [ -f "$SRCS_DIR/baseline.co" ] || err "Baseline kernel not found: $SRCS_DIR/baseline.co"
        fi

        # Verify results.tsv
        [ -f "$LOGS_DIR/results.tsv" ] || err "results.tsv not created: $LOGS_DIR/results.tsv"
        ;;

    BASELINE)
        PERF_DIR=$(jq -r '.paths.shape_dir_perf' "$STATE_FILE")
        BASELINE_TFLOPS=$(jq -r '.metrics.baseline_tflops' "$STATE_FILE")

        [ -f "$PERF_DIR/timing_iter000_baseline.txt" ] || err "Baseline timing file not found"
        [ "$BASELINE_TFLOPS" != "null" ] || err "baseline_tflops not set in loop-state.json"

        GPU_CHECKED=$(jq -r '.guard_flags.gpu_health_checked' "$STATE_FILE")
        [ "$GPU_CHECKED" = "true" ] || err "GPU health not checked before baseline"
        ;;

    PROFILE)
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        BOTTLENECK=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        LAST_BN=$(jq -r '.metrics.last_bottleneck' "$STATE_FILE")

        [ "$NCU_RAN" = "true" ] || err "ncu_ran_this_iter not set to true"
        [ "$BOTTLENECK" = "true" ] || err "bottleneck_identified not set to true"
        [ "$LAST_BN" != "null" ] || err "last_bottleneck not set in metrics"
        ;;

    IDEATE)
        NOVEL=$(jq -r '.guard_flags.idea_is_novel' "$STATE_FILE")
        LOGGED=$(jq -r '.guard_flags.idea_logged' "$STATE_FILE")

        [ "$NOVEL" = "true" ] || err "idea_is_novel not set to true"
        [ "$LOGGED" = "true" ] || err "idea_logged not set to true"

        # Verify idea was actually appended to log
        IDEA_LOG=$(jq -r '.paths.idea_log' "$STATE_FILE")
        if [ -f "$IDEA_LOG" ]; then
            if ! tail -1 "$IDEA_LOG" | jq -e ".iter == $ITERATION" > /dev/null 2>&1; then
                err "Last entry in idea-log.jsonl does not match current iteration ($ITERATION)"
            fi
        else
            err "idea-log.jsonl not found at $IDEA_LOG"
        fi
        ;;

    IMPLEMENT)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        [ "$COMPILE_OK" = "true" ] || err "compile_succeeded not set to true"

        # Check kernel file exists (iter number = fsm.iteration)
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        ITER_TAG=$(printf "iter%03d" "$ITERATION")
        KERNEL_COUNT=$(find "$SRCS_DIR" -name "${ITER_TAG}_*" 2>/dev/null | wc -l)
        [ "$KERNEL_COUNT" -ge 1 ] || err "No kernel file found matching ${ITER_TAG}_* in $SRCS_DIR"
        ;;

    MEASURE)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        THIS_TFLOPS=$(jq -r '.metrics.this_iter_tflops' "$STATE_FILE")

        [ "$TIMING_OK" = "true" ] || err "timing_captured not set to true"
        [ "$THIS_TFLOPS" != "null" ] || err "this_iter_tflops not set in metrics"

        # Check timing file exists (iter number = fsm.iteration)
        PERF_DIR=$(jq -r '.paths.shape_dir_perf' "$STATE_FILE")
        ITER_TAG=$(printf "timing_iter%03d" "$ITERATION")
        TIMING_COUNT=$(find "$PERF_DIR" -name "${ITER_TAG}*" 2>/dev/null | wc -l)
        [ "$TIMING_COUNT" -ge 1 ] || err "No timing file found matching ${ITER_TAG}* in $PERF_DIR"
        ;;

    DECIDE)
        DECISION=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        THIS_DEC=$(jq -r '.metrics.this_iter_decision' "$STATE_FILE")

        [ "$DECISION" = "true" ] || err "decision_made not set to true"
        [ "$THIS_DEC" != "null" ] || err "this_iter_decision not set (must be KEEP or DISCARD)"
        ;;

    STORE)
        RESULTS_OK=$(jq -r '.guard_flags.results_appended' "$STATE_FILE")
        CKPT_OK=$(jq -r '.guard_flags.checkpoint_written' "$STATE_FILE")
        GIT_OK=$(jq -r '.guard_flags.git_committed' "$STATE_FILE")

        [ "$RESULTS_OK" = "true" ] || err "results_appended not set to true"
        [ "$CKPT_OK" = "true" ] || err "checkpoint_written not set to true"
        [ "$GIT_OK" = "true" ] || err "git_committed not set to true"

        # Verify results.tsv was actually updated
        # fsm.iteration = current working iteration (e.g., 1 means we just stored iter 1)
        # Expected data rows = iteration + 1 (iter 0 baseline + iters 1..N)
        LOGS_DIR=$(jq -r '.paths.shape_dir_logs' "$STATE_FILE")
        if [ -f "$LOGS_DIR/results.tsv" ]; then
            LINE_COUNT=$(grep -c '^[0-9]' "$LOGS_DIR/results.tsv" 2>/dev/null || echo 0)
            EXPECTED=$((ITERATION + 1))
            if [ "$LINE_COUNT" -lt "$EXPECTED" ]; then
                err "results.tsv has $LINE_COUNT data rows but expected >= $EXPECTED (iter 0..$ITERATION)"
            fi
        fi

        # Verify checkpoint file exists
        CKPT_FILE=$(jq -r '.paths.checkpoint_file' "$STATE_FILE")
        [ -f "$CKPT_FILE" ] || err "Checkpoint file not found: $CKPT_FILE"
        ;;

    SHAPE_COMPLETE)
        # The completion promise check
        if [ "$ITERATION" -lt "$MAX_ITER" ]; then
            err "COMPLETION PROMISE FAILED: iteration=$ITERATION < max_iteration=$MAX_ITER. Shape is NOT complete."
        fi

        # Check best kernel is registered
        DTYPE=$(jq -r '.fsm.dtype' "$STATE_FILE")
        BEST_KERNEL="kernels/gemm_sp_${DTYPE}/${SHAPE_KEY}_best.cu"
        BEST_CO="kernels/gemm_sp_${DTYPE}/${SHAPE_KEY}_best.co"
        if [ ! -f "$BEST_KERNEL" ] && [ ! -f "$BEST_CO" ]; then
            err "Best kernel not registered at $BEST_KERNEL (or .co)"
        fi

        # Check tuning/state.json shows done
        if [ -f "tuning/state.json" ]; then
            STATUS=$(jq -r ".shapes[\"$SHAPE_KEY\"].status // .[\"$SHAPE_KEY\"].status // \"missing\"" "tuning/state.json")
            if [ "$STATUS" != "done" ]; then
                err "tuning/state.json does not show status=done for $SHAPE_KEY (got: $STATUS)"
            fi
        else
            err "tuning/state.json not found"
        fi

        # Check git is clean for this shape
        UNCOMMITTED=$(git status --porcelain -- "tuning/" "kernels/" 2>/dev/null | wc -l)
        if [ "$UNCOMMITTED" -gt 0 ]; then
            err "Uncommitted changes in tuning/ or kernels/ — final commit required"
        fi
        ;;
esac

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "=== POST-CHECK FAILED: $ERRORS error(s) for step $STEP ==="
    echo "Step is INCOMPLETE. Fix the errors above."
    exit 1
fi

echo "=== POST-CHECK PASSED for step $STEP ==="
exit 0
