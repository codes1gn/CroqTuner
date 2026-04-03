#!/bin/bash
set -euo pipefail

# pre-step-check.sh — Validate preconditions before each FSM step
#
# Usage: pre-step-check.sh <STEP_NAME>
# Returns 0 if all preconditions met, non-zero with diagnostic message if not.
#
# This is the "PreToolUse hook" equivalent for the CroqTuner skill.
# The LLM MUST run this before each step. If it exits non-zero, the step is BLOCKED.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CROQTUNER_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="${CROQTUNER_STATE_FILE:-$CROQTUNER_DIR/state/loop-state.json}"

if [ $# -lt 1 ]; then
    echo "Usage: pre-step-check.sh <STEP_NAME>"
    echo "Steps: BASELINE PROFILE IDEATE IMPLEMENT MEASURE DECIDE STORE SHAPE_COMPLETE"
    exit 1
fi

STEP="$1"
ERRORS=0
WARNINGS=0

err() { echo "BLOCKED: $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo "WARNING: $1"; WARNINGS=$((WARNINGS + 1)); }

if [ ! -f "$STATE_FILE" ]; then
    err "loop-state.json not found. Run state-transition.sh INIT first."
    exit 1
fi

CURRENT_STATE=$(jq -r '.fsm.current_state' "$STATE_FILE")
ITERATION=$(jq -r '.fsm.iteration' "$STATE_FILE")
MAX_ITER=$(jq -r '.fsm.max_iteration' "$STATE_FILE")
MODE=$(jq -r '.fsm.mode' "$STATE_FILE")
SHAPE_KEY=$(jq -r '.fsm.shape_key' "$STATE_FILE")
CONSEC_DISCARDS=$(jq -r '.metrics.consecutive_discards' "$STATE_FILE")

# Check state matches requested step
if [ "$CURRENT_STATE" != "$STEP" ]; then
    err "Current state is '$CURRENT_STATE', not '$STEP'. Cannot run pre-check for wrong state."
    exit 1
fi

case "$STEP" in
    BASELINE)
        # Check seed kernel exists
        SRCS_DIR=$(jq -r '.paths.shape_dir_srcs' "$STATE_FILE")
        if [ "$MODE" = "from_current_best" ]; then
            if [ ! -f "$SRCS_DIR/seed.cu" ]; then
                err "Seed kernel not found at $SRCS_DIR/seed.cu"
            fi
        elif [ "$MODE" = "from_scratch" ]; then
            if [ ! -f "$SRCS_DIR/baseline.co" ]; then
                err "Baseline kernel not found at $SRCS_DIR/baseline.co"
            fi
        fi
        ;;

    PROFILE)
        # The transition to PROFILE only happens from BASELINE or STORE.
        # STORE post-check already verified git_committed=true.
        # Flags are reset on entry, so nothing to check here.
        # Just verify we're in a valid state.
        if [ -z "$SHAPE_KEY" ] || [ "$SHAPE_KEY" = "null" ] || [ "$SHAPE_KEY" = "" ]; then
            err "shape_key not set. Run INIT first."
        fi
        ;;

    IDEATE)
        # Check ncu ran (or was waived for from_current_best)
        NCU_RAN=$(jq -r '.guard_flags.ncu_ran_this_iter' "$STATE_FILE")
        if [ "$NCU_RAN" != "true" ]; then
            err "ncu has not run this iteration (ncu_ran_this_iter=false). Complete PROFILE step first."
        fi

        BOTTLENECK=$(jq -r '.guard_flags.bottleneck_identified' "$STATE_FILE")
        if [ "$BOTTLENECK" != "true" ]; then
            err "Bottleneck not identified (bottleneck_identified=false). Complete PROFILE step first."
        fi

        # Check idea diversity
        IDEA_LOG=$(jq -r '.paths.idea_log' "$STATE_FILE")
        if [ -f "$IDEA_LOG" ] && [ -s "$IDEA_LOG" ]; then
            LINE_COUNT=$(wc -l < "$IDEA_LOG")
            if [ "$LINE_COUNT" -ge 2 ]; then
                LAST_2_CATS=$(tail -2 "$IDEA_LOG" | jq -r '.category' 2>/dev/null || echo "")
                if [ -n "$LAST_2_CATS" ]; then
                    UNIQUE_CATS=$(echo "$LAST_2_CATS" | sort -u | wc -l)
                    if [ "$UNIQUE_CATS" -eq 1 ]; then
                        REPEATED_CAT=$(echo "$LAST_2_CATS" | head -1)
                        warn "Last 2 ideas both in category '$REPEATED_CAT'. Rule D3 requires different category next."
                    fi
                fi
            fi

            # Category phase progression checks
            if [ "$MODE" = "from_scratch" ]; then
                STRUCTURAL_COUNT=$(jq -r 'select(.category == "structural")' "$IDEA_LOG" 2>/dev/null | grep -c "structural" || echo 0)
                CHOREO_COUNT=$(jq -r 'select(.category == "choreo")' "$IDEA_LOG" 2>/dev/null | grep -c "choreo" || echo 0)
                NCU_MICRO_COUNT=$(jq -r 'select(.category == "ncu_micro")' "$IDEA_LOG" 2>/dev/null | grep -c "ncu_micro" || echo 0)

                if [ "$ITERATION" -ge 30 ] && [ "$STRUCTURAL_COUNT" -lt 5 ]; then
                    warn "Phase progression: at iter $ITERATION but only $STRUCTURAL_COUNT structural ideas (need >=5 by iter 30)."
                fi
                if [ "$ITERATION" -ge 80 ] && [ "$CHOREO_COUNT" -lt 3 ]; then
                    warn "Phase progression: at iter $ITERATION but only $CHOREO_COUNT choreo ideas (need >=3 by iter 80)."
                fi
                if [ "$ITERATION" -ge 120 ] && [ "$NCU_MICRO_COUNT" -lt 3 ]; then
                    warn "Phase progression: at iter $ITERATION but only $NCU_MICRO_COUNT ncu_micro ideas (need >=3 by iter 120)."
                fi
            fi
        fi

        # Discard-triggered escalation
        if [ "$CONSEC_DISCARDS" -ge 10 ]; then
            warn "consecutive_discards=$CONSEC_DISCARDS (>=10). MUST try radical changes (choreo rewrite, split-K, different WGMMA tile)."
        elif [ "$CONSEC_DISCARDS" -ge 5 ]; then
            warn "consecutive_discards=$CONSEC_DISCARDS (>=5). MUST try completely different approach category."
        elif [ "$CONSEC_DISCARDS" -ge 3 ]; then
            warn "consecutive_discards=$CONSEC_DISCARDS (>=3). SHOULD run ncu and switch optimization category."
        fi
        ;;

    IMPLEMENT)
        IDEA_LOGGED=$(jq -r '.guard_flags.idea_logged' "$STATE_FILE")
        if [ "$IDEA_LOGGED" != "true" ]; then
            err "Idea not logged (idea_logged=false). Complete IDEATE step first."
        fi
        ;;

    MEASURE)
        COMPILE_OK=$(jq -r '.guard_flags.compile_succeeded' "$STATE_FILE")
        if [ "$COMPILE_OK" != "true" ]; then
            err "Kernel not compiled successfully (compile_succeeded=false). Complete IMPLEMENT step first."
        fi
        ;;

    DECIDE)
        TIMING_OK=$(jq -r '.guard_flags.timing_captured' "$STATE_FILE")
        if [ "$TIMING_OK" != "true" ]; then
            err "Timing not captured (timing_captured=false). Complete MEASURE step first."
        fi
        ;;

    STORE)
        DECISION_OK=$(jq -r '.guard_flags.decision_made' "$STATE_FILE")
        if [ "$DECISION_OK" != "true" ]; then
            err "Decision not made (decision_made=false). Complete DECIDE step first."
        fi
        ;;

    SHAPE_COMPLETE)
        if [ "$ITERATION" -lt "$MAX_ITER" ]; then
            err "iteration=$ITERATION < max_iteration=$MAX_ITER. Shape is NOT complete. Continue the loop."
        fi
        ;;
esac

# Summary
if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "=== PRE-CHECK FAILED: $ERRORS error(s), $WARNINGS warning(s) for step $STEP ==="
    echo "Fix the errors above before proceeding."
    exit 1
fi

if [ "$WARNINGS" -gt 0 ]; then
    echo ""
    echo "=== PRE-CHECK PASSED with $WARNINGS warning(s) for step $STEP ==="
    echo "Proceed, but address the warnings above."
    exit 0
fi

echo "=== PRE-CHECK PASSED for step $STEP ==="
exit 0
