---
name: ai-tune-from-current-best
description: Sweep ALL scenario shapes and adapt the reference best kernel (f16 iter143, e4m3 iter068) to each shape with up to 30 iterations per shape. Non-stop until every shape across all scenarios has a tuned variant. Use when the user asks to "tune from best", "adapt kernels", "sweep tune", or "ai-tune-from-current-best".
argument-hint: <dtype: f16|e4m3|all>
disable-model-invocation: true
---

# AI-Tune from Current Best: Sweep-Based Adaptation

Adapt the reference best kernel to EVERY scenario shape with up to 30 iterations per shape. This skill runs as a NON-STOP sweep: it iterates through all unfinished shapes, tunes each one, then moves to the next. It does NOT stop until every shape has been tuned.

**All work stays on the current branch.** Each shape's artifacts are stored separately. No experiment branches.

## CroqTuner Harness Integration

**This skill is managed by the CroqTuner FSM harness.** Before doing anything:

1. **Read the CroqTuner router**: `.claude/skills/fsm-engine/SKILL.md`
2. **Read identity constraints**: `.claude/skills/fsm-engine/protocol/identity.md`
3. **Check FSM state**: read `.claude/skills/fsm-engine/state/loop-state.json` (if it exists, resume from there)

**Note:** This skill uses the default state file `loop-state.json`. The `from-scratch` skill uses a separate `loop-state_from_scratch.json` so both can run concurrently.

The CroqTuner harness provides:
- **FSM state tracking** — you always know which step to execute next
- **Guard flag enforcement** — pre/post validation scripts block skipped steps
- **Idea dedup log** — append-only JSONL prevents repeated ideas
- **Compaction-safe resume** — structured summary survives context loss
- **Completion promise** — mechanical check prevents premature exit

**For each iteration, follow the CroqTuner step sequence:**
```
pre-step-check.sh → execute step → update guard flags → post-step-check.sh → state-transition.sh
```

Initialize the FSM for this mode:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh INIT \
    shape_key=<KEY> dtype=<DTYPE> mode=from_current_best max_iteration=30 shape='[M,N,K]'
```

## Overview

The reference kernels (f16 iter143 at 655 TFLOPS, e4m3 iter068 at 1127 TFLOPS) were tuned for M=4096, N=8192, K=8192. Many shapes nearby can reuse that kernel's structure with minor adaptation. This skill recompiles the reference kernel for each target shape and runs up to 30 optimization iterations to adapt it.

## Pre-flight

1. **Read CroqTuner harness**: `.claude/skills/fsm-engine/SKILL.md` — the FSM router.
2. **Read manifest**: `kernels/manifest.json` — scenarios, reference kernels, build config, region clustering.
3. **Read tuning state**: `tuning/state.json` — which shapes are already done.
4. **Read choreo program**: `/home/albert/workspace/choreo/.claude/program.md` — loop rules (Rules 1–10 apply with modifications noted below).
5. **Read choreo-syntax skill** if editing `.co` files.

## Shape Schedule

Build the full list of shapes to tune from `manifest.json` scenarios and sweep sizes.
There are **7 scenarios** with **10 sweep lines** total, producing **134 unique shapes per dtype** (268 total for f16+e4m3).

For each dtype requested (f16, e4m3, or both), enumerate shapes from every scenario:

### 4 single-dim scenarios

```
scenario=square:   (S,S,S)         for S in [256,512,768,1024,2048,3072,4096,6144,8192,12288,16384]           (11 shapes)
scenario=sweep_m:  (S,8192,8192)   for S in [128,256,384,512,768,1024,2048,3072,4096,6144,8192,16384]         (12 shapes)
scenario=sweep_n:  (4096,S,8192)   for S in [256,512,1024,2048,3072,4096,6144,8192,12288,16384,32768]         (11 shapes)
scenario=sweep_k:  (4096,8192,S)   for S in [128,256,512,1024,2048,4096,6144,8192,12288,16384,32768]          (11 shapes)
```

### 3 tied-pair scenarios (x2 sub-series each = 12 sub-series)

**sweep_mn (M=N tied):**
```
fixK4096:  (T,T,4096)     for T in [256,512,1024,2048,3072,4096,6144,8192,12288,16384]   (10 shapes)
fixK8192:  (T,T,8192)     for T in [256,512,1024,2048,3072,4096,6144,8192,12288,16384]   (10 shapes)
fixMN4096: (4096,4096,K)  for K in [128,256,512,1024,2048,4096,6144,8192,12288,16384]    (10 shapes)
fixMN8192: (8192,8192,K)  for K in [128,256,512,1024,2048,4096,6144,8192,12288,16384]    (10 shapes)
```

**sweep_mk (M=K tied):**
```
fixN4096:  (T,4096,T)     for T in [256,512,1024,2048,3072,4096,6144,8192,12288,16384]   (10 shapes)
fixN8192:  (T,8192,T)     for T in [256,512,1024,2048,3072,4096,6144,8192,12288,16384]   (10 shapes)  (note: T=256 invalid since 256<min_N? No, N is fixed at 8192 here, T applies to M and K)
fixMK4096: (4096,N,4096)  for N in [256,512,1024,2048,4096,6144,8192,12288,16384,32768]  (10 shapes)
fixMK8192: (8192,N,8192)  for N in [256,512,1024,2048,4096,6144,8192,12288,16384,32768]  (10 shapes)
```

**sweep_nk (N=K tied):**
```
fixM4096:  (4096,T,T)     for T in [256,512,1024,2048,3072,4096,6144,8192,12288,16384]   (10 shapes)
fixM8192:  (8192,T,T)     for T in [256,512,1024,2048,3072,4096,6144,8192,12288,16384]   (10 shapes)
fixNK4096: (M,4096,4096)  for M in [128,256,512,1024,2048,4096,6144,8192,12288,16384]    (10 shapes)
fixNK8192: (M,8192,8192)  for M in [128,256,512,1024,2048,4096,6144,8192,12288,16384]    (10 shapes)
```

### Deduplication & filtering

Apply `size_constraints` (M>=128, N>=256, K>=128) to filter invalid shapes.

Deduplicate: many shapes appear in multiple scenarios (e.g. 4096x4096x4096 appears in square, sweep_mn_fixK4096, sweep_mk_fixN4096, sweep_nk_fixM4096, sweep_nk_fixNK4096, sweep_mk_fixMK4096, sweep_mn_fixMN4096). Tune once, register for all matching scenarios. Each shape's `scenarios` list in `state.json` tracks which scenarios it covers.

After dedup: **134 unique (M,N,K) triples per dtype**.

### Priority Order

1. Shapes closest to the reference shape (4096,8192,8192) first — these are most likely to succeed with few iterations
2. Then expanding outward by distance
3. Skip shapes already marked as `done` in `tuning/state.json`
4. For each shape, check the `region` field: `near` shapes are the primary targets for this skill

## Per-Shape Storage Layout

Every shape gets its own directory under `tuning/`:

```
tuning/
├── state.json                              ← global progress tracker
├── logs/
│   └── <dtype>_<M>x<N>x<K>/
│       └── results.tsv                     ← iteration log for this shape
├── srcs/
│   └── <dtype>_<M>x<N>x<K>/
│       ├── seed.cu                         ← copy of reference kernel (starting point)
│       ├── iter001_<tag>.cu (or .co)       ← first mutation
│       ├── iter002_<tag>.cu                ← second mutation
│       └── ...                             ← ALL iterations kept
├── perf/
│   └── <dtype>_<M>x<N>x<K>/
│       ├── timing_iter000_baseline.txt     ← baseline perf output
│       ├── timing_iter001.txt              ← per-iter timing output
│       └── ...
└── checkpoints/
    └── <dtype>_<M>x<N>x<K>.json           ← crash-safe state for this shape
```

The `<dtype>_<M>x<N>x<K>` key uniquely identifies each shape (e.g. `f16_4096x4096x4096`).

## Sweep Loop (NON-STOP)

```
for dtype in requested_dtypes:
    shapes = build_shape_schedule(dtype)
    for shape in shapes:
        key = f"{dtype}_{shape.M}x{shape.N}x{shape.K}"

        # Check if already done — ONLY skip if iter >= 30
        if state.json[key].status == "done" and state.json[key].current_iter >= 30:
            skip
        # IMPORTANT: if status == "done" but current_iter < 30, it was
        # prematurely closed. REOPEN it and continue from current_iter.

        # Check if partially done (resume)
        if state.json[key] exists and current_iter > 0:
            resume from checkpoint

        # Otherwise: start fresh for this shape
        run_adaptation_loop(dtype, shape, max_iter=30)
```

**DO NOT STOP between shapes.** After finishing one shape, immediately proceed to the next.

**anti-early-stop (critical — read this):**
there are **~260 shapes still pending**. completing 1, 2, 5, or even 20 shapes is not
a reason to stop. you must keep tuning until the session physically drops or all shapes
are done. do not:
- output a "progress report" and wait for user input
- say "i've completed n shapes, shall i continue?"
- summarize results and stop after a "natural break point"
- treat the end of any single shape as a stopping opportunity

the user has explicitly requested non-stop operation. the only valid stop conditions are:
1. all shapes have `status: "done"` in `tuning/state.json`
2. session/connection drops (crash-safe resume handles this)

note: if has systemic gpu failure that cannot be remediated, raise an interactive quiz to user, ask for solution: a. force kill process (use sudo, ask user for passwd); b. wait for gpu to idle (ask user wait time); c. stop tuning.

## Per-Shape Adaptation Loop

### Step 0: Setup for this shape

```bash
DTYPE=<f16|e4m3>
M=<M>  N=<N>  K=<K>
KEY="${DTYPE}_${M}x${N}x${K}"
SHAPE_DIR_LOGS="tuning/logs/$KEY"
SHAPE_DIR_SRCS="tuning/srcs/$KEY"
SHAPE_DIR_PERF="tuning/perf/$KEY"

mkdir -p "$SHAPE_DIR_LOGS" "$SHAPE_DIR_SRCS" "$SHAPE_DIR_PERF"
```

Copy the reference kernel as seed:
```bash
cp <reference_kernel_from_manifest> "$SHAPE_DIR_SRCS/seed.cu"
```

Initialize `results.tsv`:
```
# CrokTile adapt-from-best: $KEY
# Reference: <reference kernel path>
# Shape: M=$M N=$N K=$K  Dtype: $DTYPE  Max iter: 30
iter	kernel	tflops	hw_eff_pct	decision	bottleneck	idea_summary	run_command
```

### Step 1: Baseline measurement

Compile the seed kernel for this shape and measure:
```bash
CHOREO_REPO=/home/albert/workspace/choreo
nvcc $NVCC_FLAGS $DTYPE_EXTRA \
  -DSPMM_DEFAULT_M=$M -DSPMM_DEFAULT_N=$N -DSPMM_DEFAULT_K=$K \
  -I"$CHOREO_REPO/runtime" -I"$CHOREO_REPO/extern/cutlass/include" -I"$CHOREO_REPO" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o /tmp/${KEY}_baseline "$SHAPE_DIR_SRCS/seed.cu"

CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500 CHOREO_SKIP_VERIFY=1 \
  /tmp/${KEY}_baseline --skip-verify
```

Capture output to `$SHAPE_DIR_PERF/timing_iter000_baseline.txt`.
Record baseline TFLOPS in results.tsv as iter 0.

### Step 2–N: Optimization iterations (up to 30)

## CRITICAL: DEEP TUNING — DO NOT SKIM

**You MUST stick to the current shape and exhaust optimization avenues before moving on.**
A shape is NOT done until you hit `current_iter >= 30`. PERIOD.
There is NO early exit. `consecutive_discards` is NEVER a termination condition — it is
ONLY a signal to CHANGE STRATEGY. Even if you hit 20 consecutive discards at iter 20,
you MUST keep going until iter 30. Trying 3–5 macro tweaks and bailing is FORBIDDEN.
You have 30 iterations — USE ALL 30.

**Idea diversity is MANDATORY.** You must explore progressively deeper optimization levels:

**Phase 1 — Macro-level (iter 1–8):** Quick parameter sweeps
- SPMM_WARP_N (128 vs 256), SPMM_STAGES (2 vs 3), SPMM_OUTPUT_PAD (0 vs 8)
- L2 promotion (NONE vs L2_128B vs L2_256B, selective per-TMA-descriptor)
- META_TILE_COLS, PACKED_TILE_K adjustments

**Phase 2 — Structural CUDA changes (iter 9–18):** Edit the .cu file directly
- Change CTA scheduling order (linear_blk formula: row-major vs column-major vs swizzled)
- Change SMEM allocation layout (interleave LHS/RHS/metadata vs contiguous)
- Add/remove/change `__launch_bounds__` with different minBlocksPerMultiprocessor
- Modify barrier wait depths (wait<0> vs wait<1> vs wait<2>)
- Change output epilogue (vectorized vs scalar store, stmatrix patterns)
- Add prefetch hints (`cp.async.bulk.prefetch`)
- Try different fence patterns for producer-consumer synchronization

**Phase 3 — Choreo recompilation (iter 19–25):** Go back to .co source
- Write a NEW .co kernel for this shape from scratch or by adapting the reference .co
- Use choreo compiler (`choreo -gs -t cute -arch=sm_90a --use-warpspec --use-prepack`) to generate fresh CUDA
- Try different choreo flags (e.g., without `--use-warpspec`, or without `--use-prepack`)
- Edit the choreo EDSL: change tile sizes, pipeline depths, warp topology in the .co source

**Phase 4 — ncu-guided micro-optimization (iter 26–30):** Profile-driven polish
- Run ncu, identify specific bottlenecks (SMEM bank conflicts, L2 thrashing, warp stalls)
- Apply targeted fixes from ncu data
- Try inline PTX (mbarrier tuning, nanosleep in producer, fence_proxy_async)

**If you exhaust ideas in an earlier phase, jump to a later phase. If ncu reveals a bottleneck, act on it immediately regardless of phase.**

For each iteration:

1. **Profile**: Run ncu at least at iter 1, every 5–10 iters, and whenever stuck (3+ consecutive discards). Ideas MUST be grounded in data — ncu metrics, compiler output, TFLOPS trends, or known shape-sensitivity patterns. Guessing is FORBIDDEN.

2. **Raise ONE idea**: Based on available data. The idea MUST be different from all previous ideas for this shape (check results.tsv). After 2 consecutive macro-only changes, you MUST try a structural change.

3. **Implement**: Create `$SHAPE_DIR_SRCS/iter<NNN>_<tag>.cu` (or `.co`). Compile for the target shape. Verify correctness.

4. **Measure**: Run timing. Capture to `$SHAPE_DIR_PERF/timing_iter<NNN>.txt`.

5. **Decide**: KEEP if TFLOPS improves, DISCARD otherwise.

6. **Store (STRICT — every iteration, KEEP or DISCARD)**:

   a. Append row to `$SHAPE_DIR_LOGS/results.tsv`
   b. Write checkpoint `tuning/checkpoints/$KEY.json`:
      ```json
      {
        "key": "<KEY>",
        "dtype": "<dtype>",
        "shape": [M, N, K],
        "mode": "from_current_best",
        "max_iter": 30,
        "current_iter": <N>,
        "best_iter": <best>,
        "best_tflops": <X>,
        "baseline_tflops": <Y>,
        "best_kernel": "<path to best .cu/.co>",
        "consecutive_discards": <N>,
        "status": "active",
        "last_updated": "<ISO timestamp>"
      }
      ```
   c. `git add` all changed files under `tuning/` and commit:
      ```bash
      git add tuning/logs/$KEY/ tuning/srcs/$KEY/ tuning/perf/$KEY/ tuning/checkpoints/$KEY.json tuning/state.json
      git commit -m "$KEY iter<NNN>: <description> — TFLOPS: X -> Y (<KEEP|DISCARD>)"
      ```

   **This strict per-iteration store means: if the session breaks after any commit, ALL prior work is preserved and the next session can resume from the exact checkpoint.**

### Step 3: Shape completion

**A shape is ONLY complete when: `current_iter >= 30`. FULL STOP.**

There is NO early exit condition. Not 10 consecutive discards, not 20. You MUST
run all 30 iterations for every single shape. The only valid reason to stop before
iter 30 is if the session/connection drops (crash-safe resume handles this).

**`consecutive_discards` is a STRATEGY-CHANGE signal, NOT a termination signal:**
- 3+ consecutive discards → run ncu, switch optimization phase
- 5+ consecutive discards → try a completely different approach (.co recompile, inline PTX)
- 10+ consecutive discards → try radical ideas (split-K, different WGMMA tile via manual rewrite)
- But you KEEP GOING until iter 30 regardless.

When `current_iter >= 30`:

1. Copy the best kernel to the kernel registry:
   ```bash
   cp "$BEST_KERNEL" "kernels/gemm_sp_${DTYPE}/${KEY}_best.cu"
   ```

2. Update `tuning/state.json`:
   ```json
   {
     "<KEY>": {
       "status": "done",
       "mode": "from_current_best",
       "current_iter": 30,
       "best_iter": <N>,
       "best_tflops": <X>,
       "baseline_tflops": <Y>,
       "best_kernel": "kernels/gemm_sp_<dtype>/<KEY>_best.cu"
     }
   }
   ```

3. Commit:
   ```bash
   git add kernels/ tuning/state.json tuning/checkpoints/$KEY.json
   git commit -m "$KEY: completed adapt-from-best, best TFLOPS: <X> at iter<N>"
   ```

4. **Immediately proceed to the next shape. DO NOT STOP. DO NOT SUMMARIZE AND WAIT.**
   Pick the next pending shape from `tuning/state.json`, run `state-transition.sh INIT`
   with the new shape params, and begin its tuning loop. Zero user interaction between shapes.

## Resuming After Interruption

On skill invocation, always check `tuning/state.json` first:
- Find any shape with `status: "active"` → load its checkpoint, resume from `current_iter + 1`
- After finishing the active shape, continue the sweep from where it left off

The sweep order is deterministic (same shape schedule every time), so the skill can always determine which shapes remain.

## Rules

All choreo `program.md` rules apply with these modifications:

- **Rule 1 (profile before idea) — RELAXED for adapt-from-best**: ncu is NOT required every iteration. Profile when useful (first iter, every 5–10 iters, when stuck). Ideas MUST still be grounded in data — compiler output, TFLOPS trends, or known shape effects are acceptable data sources.
- **Rule 2 (hill-climb)**: Applies — always mutate from current best for this shape.
- **Rule 3 (diversity)**: Applies.
- **Rule 4 (no repeat)**: Applies — check this shape's results.tsv.
- **Rule 5 (abandon after 3)**: Applies.
- **Rule 13 (shape fixed)**: Shape is fixed per shape-slot. Different shapes are different slots.
- **Rule 14 (separate artifacts)**: Each shape has its own directory.
- **NEW — no branches**: All work on main. Commit after every iteration.
- **NEW — non-stop sweep**: Do NOT stop between shapes. The loop covers ALL shapes.

## GPU Health Check Protocol (MANDATORY)

**Before EVERY baseline measurement and whenever a perf number looks suspicious, you MUST validate GPU health.** A "suspicious" number is one that deviates >20% from expectation (e.g. a trivial macro change yielding 3x improvement, or a known-good kernel running far below prior results).

### Step 0: Check GPU status BEFORE any measurement

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
```

**Healthy GPU**: utilization ~0%, memory used ~14 MiB (no user processes), temperature <80°C.

### If GPU is NOT healthy:

1. **Check for zombie processes**:
   ```bash
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
   ```

2. **Kill any orphaned processes** occupying the GPU:
   ```bash
   # List and kill all GPU processes on the target device
   nvidia-smi --query-compute-apps=pid --format=csv,noheader -i <GPU_ID> | xargs -r kill -9
   ```

3. **If kill doesn't free the GPU, try GPU reset**:
   ```bash
   sudo nvidia-smi -r -i <GPU_ID>
   ```

4. **If one GPU is unhealthy, try the other GPU** (switch `CUDA_VISIBLE_DEVICES`).

5. **Re-check** after remediation:
   ```bash
   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
   ```

### Sanity check on results

After every timing measurement, apply these sanity checks:

- **Baseline of a new shape**: Compare with the reference shape's baseline. For the same kernel, TFLOPS should scale roughly proportionally to problem size and should never be >2x the reference without structural reason.
- **Iteration improvement**: Any single iteration claiming >50% improvement over current best is almost certainly a measurement artifact. Re-run the measurement. If it reproduces, re-run the PREVIOUS best to confirm the comparison is fair (both on clean GPU).
- **Iteration regression**: If TFLOPS drops >50% from the previous measurement (even for a DISCARD), suspect GPU contention, not a kernel bug. Check GPU health before recording.

### Re-run protocol for suspicious results

When a result looks suspicious:

1. Check GPU health (above)
2. Re-run the CURRENT iteration's kernel
3. Re-run the PREVIOUS best kernel (to re-baseline)
4. Use the re-run numbers for the KEEP/DISCARD decision
5. Record the re-run numbers (not the original suspicious ones) in results.tsv

## Context Compaction

**Follow the CroqTuner compaction protocol** (`.claude/skills/fsm-engine/protocol/compaction-protocol.md`):

1. Commit all pending work (the STORE step does this automatically)
2. Ensure `loop-state.json` reflects the true current state
3. Update `state/compaction-summary.md` with latest iteration info
4. After compaction: read files in the order specified by compaction-protocol.md
5. Resume from `fsm.current_state` — do NOT re-read this entire SKILL.md

## Related Skills

- `ai-tune-from-scratch` — for typical shapes that need full 150-iter tuning from baseline
- `fsm-engine` — the FSM harness that manages both tuning modes
