---
name: fsm-engine
description: CroqTuner harness — robust FSM-driven GPU kernel tuning loop with guard flags, idea dedup, and compaction-safe state. Wraps ai-tune-from-current-best and ai-tune-from-scratch with mechanical enforcement of the tuning protocol. Use when the user asks to "tune", "ai-tune", "croqtune", "tune from best", "tune from scratch", or "start tuning".
argument-hint: <mode: from-best|from-scratch> <dtype: f16|e4m3|all> [shape_key]
---

# CroqTuner — Router Skill

This skill manages the tuning loop through a finite state machine. Instead of relying on LLM memory to follow a long protocol, it uses:
- **Mode-specific state files** — each tuning mode has its own FSM state file so they can run concurrently
- **Validation scripts** — pre/post checks that block illegal transitions
- **Idea dedup log** — append-only JSONL preventing repeated ideas
- **Compaction summary** — structured resume instructions surviving context loss

## Dual State Files (Concurrent Mode Support)

Two tuning modes can run in parallel on separate agent sessions because each has its own state file:

| Mode | State file | Env var |
|---|---|---|
| `from_current_best` | `state/loop-state.json` (default) | not needed |
| `from_scratch` | `state/loop-state_from_scratch.json` | `CROQTUNER_STATE_FILE=.claude/skills/fsm-engine/state/loop-state_from_scratch.json` |

**All FSM scripts** (`state-transition.sh`, `pre-step-check.sh`, `post-step-check.sh`) respect the `CROQTUNER_STATE_FILE` environment variable. If unset, they default to `state/loop-state.json`.

**For from-scratch mode, ALWAYS set the env var before calling any FSM script:**
```bash
export CROQTUNER_STATE_FILE=".claude/skills/fsm-engine/state/loop-state_from_scratch.json"
```

## FIRST THING: Read State

**On every invocation, your FIRST action is:**

```bash
# Determine which state file to use based on mode
if [ "<MODE>" = "from_scratch" ]; then
    export CROQTUNER_STATE_FILE=".claude/skills/fsm-engine/state/loop-state_from_scratch.json"
else
    export CROQTUNER_STATE_FILE=".claude/skills/fsm-engine/state/loop-state.json"
fi

if [ -f "$CROQTUNER_STATE_FILE" ]; then
    cat "$CROQTUNER_STATE_FILE"
else
    echo "NO_STATE"
fi
```

Based on the result:

### If `NO_STATE` → Fresh Start
1. Read `protocol/identity.md` — your role and inviolable constraints
2. Read `protocol/loop-contract.md` — FSM definition
3. Read `kernels/manifest.json` — scenarios, reference kernels, build config
4. Read `tuning/state.json` — which shapes are already done (if file exists)
5. Determine the shape to tune based on user arguments or sweep schedule
6. Initialize FSM (set `CROQTUNER_STATE_FILE` first if from-scratch):
   ```bash
   # For from-scratch: export CROQTUNER_STATE_FILE=".claude/skills/fsm-engine/state/loop-state_from_scratch.json"
   bash .claude/skills/fsm-engine/scripts/state-transition.sh INIT \
       shape_key=<KEY> dtype=<DTYPE> mode=<MODE> max_iteration=<N> shape='[M,N,K]'
   ```
7. Proceed to execute the INIT state actions

### If state file exists → Resume
1. Read `protocol/identity.md` — refresh your constraints
2. Read the mode-appropriate state file — determine exact state
3. Read `state/compaction-summary.md` — what happened before (if exists)
4. Read `protocol/step-checklists.md` — ONLY the section matching `fsm.current_state`
5. If in IDEATE or later: read idea dedup log at the path in `paths.idea_log`
6. Resume executing from `fsm.current_state`

## Executing a Step

For EVERY FSM step, follow this exact sequence:

### 1. Pre-Check
```bash
bash .claude/skills/fsm-engine/scripts/pre-step-check.sh <CURRENT_STATE>
```
If it exits non-zero → fix the reported issue before proceeding.

### 2. Execute Step Actions
Follow the actions listed in `protocol/step-checklists.md` for the current state.

### 3. Update Guard Flags
After each mandatory action, update the corresponding guard flag via `jq`:
```bash
jq '.guard_flags.<flag_name> = true' .claude/skills/fsm-engine/state/loop-state.json > /tmp/ls.tmp && mv /tmp/ls.tmp .claude/skills/fsm-engine/state/loop-state.json
```
For metrics updates:
```bash
jq '.metrics.<metric_name> = <value>' .claude/skills/fsm-engine/state/loop-state.json > /tmp/ls.tmp && mv /tmp/ls.tmp .claude/skills/fsm-engine/state/loop-state.json
```

### 4. Post-Check
```bash
bash .claude/skills/fsm-engine/scripts/post-step-check.sh <CURRENT_STATE>
```
If it exits non-zero → complete the missing actions.

### 5. Transition
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh <NEXT_STATE> [key=value ...]
```

Then immediately proceed to the next step's pre-check. Do NOT pause between steps.

## Mode-Specific Configuration

### from-best (adapt from reference kernel)
- `max_iteration`: 30
- ncu: required at iter 1, every 5-10 iters, and when `consecutive_discards >= 3`
- Seed: reference best `.cu` from `manifest.json`
- Build: nvcc with NVCC_FLAGS from manifest
- Read `protocol/idea-diversity-rules.md` for the from_current_best phase schedule

### from-scratch (deep tuning from baseline)
- `max_iteration`: 150
- ncu: MANDATORY before every IDEATE step (no exceptions)
- Seed: baseline `.co` from `manifest.json`
- Build: choreo compiler + nvcc
- Read `protocol/idea-diversity-rules.md` for the from_scratch phase schedule

## Build Commands Reference

### from-best (nvcc direct)
```bash
CHOREO_REPO=/home/albert/workspace/choreo
nvcc -std=c++20 -O3 -arch=sm_90a --use_fast_math \
  -DSPMM_DEFAULT_M=$M -DSPMM_DEFAULT_N=$N -DSPMM_DEFAULT_K=$K \
  -I"$CHOREO_REPO/runtime" -I"$CHOREO_REPO/extern/cutlass/include" -I"$CHOREO_REPO" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o /tmp/${KEY}_iter<NNN> <kernel.cu>
```

For e4m3, add: `-DSPMM_DTYPE_E4M3`

### from-scratch (choreo + nvcc)
```bash
CHOREO_REPO=/home/albert/workspace/choreo
$CHOREO_REPO/choreo -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  <kernel.co> -o /tmp/${KEY}_iter<NNN>.cute.result
```

### Timing
```bash
CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500 CHOREO_SKIP_VERIFY=1 \
  /tmp/${KEY}_iter<NNN> --skip-verify
```
Or for choreo-compiled:
```bash
CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500 \
  bash /tmp/${KEY}_iter<NNN>.cute.result --execute
```

### ncu Profiling
```bash
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o /tmp/${KEY}_ncu_iter<NNN> \
  /tmp/${KEY}_iter<NNN> --skip-verify
```

## GPU Health Check
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
```
Healthy: utilization ~0%, memory ~14 MiB, temperature <80°C.

## Compaction-Safety

Before context fills, proactively update `state/compaction-summary.md`:
```bash
# Template — fill in actual values
cat > .claude/skills/fsm-engine/state/compaction-summary.md << 'COMPACTION_EOF'
# Compaction Summary
## Last Updated
<timestamp>
## IMMEDIATE ACTION
Read `loop-state.json` and resume from state `<STATE>` at iteration <N>.
## Current Task
- Shape: <KEY> (<mode>, <max_iter> iters max)
- Current best: <TFLOPS> TFLOPS at iter <N>
- Baseline: <TFLOPS> TFLOPS
- Last bottleneck: <category>
- Consecutive discards: <N>
## Recent History (last 5 iterations)
<fill in>
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
COMPACTION_EOF
```

## Sweep Schedule — NON-STOP (CRITICAL)

**The sweep is NON-STOP. After completing a shape, you MUST immediately start the next one.** There are ~260 shapes pending. Completing 1-5 shapes is NOT a reason to stop. The sweep runs until ALL shapes are done or the session physically drops.

After completing one shape, immediately proceed to the next:
1. Read `tuning/state.json` for completed shapes
2. Pick the next pending shape (prefer "near" region, then "far")
3. Reset loop-state.json via `state-transition.sh INIT` with new shape params
4. **Continue immediately — do NOT pause, do NOT summarize-and-wait, do NOT ask the user**

**ANTI-STOP RULES:**
- Do NOT output "Shape X complete, shall I continue?" — just continue.
- Do NOT output a progress table and wait — write it to compaction-summary.md instead.
- Do NOT treat shape boundaries as natural stopping points — they are NOT.
- The user will interrupt you if they want to stop. Until then, keep going.

## When Stuck

If you're genuinely stuck (compile error that won't resolve, ncu timeout, GPU unavailable):
1. Log the failure in `idea-log.jsonl` with appropriate result code
2. If it's a single iteration failure: mark as DISCARD, continue loop
3. If it's a systemic issue (GPU down): report to user with diagnostics
4. NEVER silently stop the loop. Either continue or explicitly report why you can't.
5. A completed shape is NEVER a reason to stop. Immediately proceed to the next shape.

## Protocol Files Reference

| File | When to Read | Content |
|---|---|---|
| `protocol/identity.md` | Every invocation | Who you are, inviolable constraints |
| `protocol/loop-contract.md` | Fresh start or when confused about FSM | State machine definition |
| `protocol/step-checklists.md` | Before each step (only relevant section) | Entry/exit conditions per step |
| `protocol/compaction-protocol.md` | After context compaction | How to resume |
| `protocol/idea-diversity-rules.md` | Before IDEATE | Dedup rules, category definitions |
| `state/loop-state.json` | Every invocation (FIRST thing to read) | Machine-readable FSM state |
| `state/compaction-summary.md` | After compaction | Human-readable state summary |
