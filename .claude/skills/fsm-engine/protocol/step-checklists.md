# CroqTuner — Step Checklists (Per-Step Entry/Exit Conditions)

Each FSM state has mandatory entry conditions, actions, exit conditions, and failure handlers. The validation scripts (`pre-step-check.sh`, `post-step-check.sh`) enforce these mechanically.

---

## INIT

### Entry Conditions
- Shape key is determined (from sweep schedule or resume)
- No active `loop-state.json` for a different shape (finish it first)

### Mandatory Actions
1. Set `fsm.shape_key`, `fsm.dtype`, `fsm.shape`, `fsm.mode`, `fsm.max_iteration`
2. Create directories: `tuning/{logs,srcs,perf}/<KEY>/`
3. Copy seed kernel:
   - `from_current_best`: copy reference `.cu` → `tuning/srcs/<KEY>/seed.cu`
   - `from_scratch`: copy baseline `.co` → `tuning/srcs/<KEY>/baseline.co`
4. Initialize `results.tsv` with header
5. Initialize `idea-log.jsonl` (empty file)
6. `loop-state.json` is created by `state-transition.sh INIT` with `current_state: "INIT"`, `iteration: 0`

### Exit → BASELINE
- Directories exist, seed kernel copied, results.tsv initialized

### On Failure
- If directories already exist (resume case): skip creation, read existing state instead

---

## BASELINE

### Entry Conditions
- `fsm.current_state == "BASELINE"`
- Seed/baseline kernel file exists in `tuning/srcs/<KEY>/`

### Mandatory Actions
1. Run GPU health check: `nvidia-smi --query-gpu=...`
2. Set `guard_flags.gpu_health_checked = true`
3. Compile seed kernel for target shape
4. Run timing benchmark (CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500)
5. Capture output to `tuning/perf/<KEY>/timing_iter000_baseline.txt`
6. Record baseline TFLOPS
7. Set `metrics.baseline_tflops`, `metrics.current_best_tflops`, `metrics.current_best_iter = 0`
8. Append iter 0 row to results.tsv
9. Set `fsm.iteration = 0`

### Exit → PROFILE
- `guard_flags.gpu_health_checked == true`
- `metrics.baseline_tflops != null`
- Timing file exists on disk

### On Failure
- Compile error: check NVCC flags, include paths. Report error details.
- GPU unhealthy: run GPU health remediation protocol, retry once.
- Suspiciously low TFLOPS (<10% of expected): re-check GPU, re-run once.

---

## PROFILE

### Entry Conditions
- `fsm.current_state == "PROFILE"`
- Previous iteration fully committed (or this is the first iteration)

### Mandatory Actions (mode-dependent)

**`from_scratch` mode (ALWAYS profile):**
1. Run ncu on current best kernel with `--set full`
2. Save ncu report to `tuning/perf/<KEY>/ncu_iter<NNN>.txt`
3. Parse bottleneck category from ncu output. Categories:
   - `smem_throughput` — shared memory is the limiter
   - `l2_throughput` — L2 cache bandwidth limited
   - `dram_throughput` — global memory bandwidth limited
   - `compute_bound` — ALU/tensor core throughput limited
   - `latency_bound` — stall cycles dominate (barrier waits, etc.)
   - `occupancy_limited` — not enough warps to hide latency
   - `instruction_fetch` — icache pressure
4. Set `guard_flags.ncu_ran_this_iter = true`
5. Set `guard_flags.bottleneck_identified = true`
6. Set `metrics.last_bottleneck = <category>`

**`from_current_best` mode (profile conditionally):**
- MUST profile at iter 1, every 5-10 iters, and when `consecutive_discards >= 3`
- MAY skip profiling for other iterations
- When skipping: set `guard_flags.ncu_ran_this_iter = true` (waived), set `metrics.last_bottleneck` from previous known value
- When NOT skipping: same as from_scratch above

### Exit → IDEATE
- `guard_flags.ncu_ran_this_iter == true`
- `guard_flags.bottleneck_identified == true`
- `metrics.last_bottleneck != null`

### On Failure
- ncu hangs (>5 min): kill ncu process, set `last_bottleneck = "ncu_timeout"`, proceed
- ncu crash: retry once, if still fails set `last_bottleneck = "ncu_error"`, proceed
- GPU unhealthy during ncu: remediate, retry

---

## IDEATE

### Entry Conditions
- `fsm.current_state == "IDEATE"`
- `guard_flags.ncu_ran_this_iter == true` (profiling done or waived)
- `guard_flags.bottleneck_identified == true`

### Mandatory Actions
1. Read `idea-log.jsonl` for this shape
2. Count idea categories used so far
3. Check diversity requirement (see `idea-diversity-rules.md`)
4. Propose ONE optimization idea grounded in:
   - ncu bottleneck data (preferred)
   - Compiler output / register usage
   - TFLOPS trends from results.tsv
   - Known shape-sensitivity patterns from reference kernel
5. Verify idea is novel (not in idea-log.jsonl)
6. Set `guard_flags.idea_is_novel = true`
7. Log idea to `idea-log.jsonl`:
   ```jsonl
   {"iter": N, "idea": "<description>", "category": "<macro|structural|choreo|ncu_micro>", "bottleneck_before": "<category>", "justification": "<why this idea>"}
   ```
8. Set `guard_flags.idea_logged = true`

### Exit → IMPLEMENT
- `guard_flags.idea_is_novel == true`
- `guard_flags.idea_logged == true`

### On Failure
- No novel idea found: widen category search (if doing macro → try structural). If truly stuck after 3 attempts to find a novel idea, try a radical approach (choreo rewrite, inline PTX).

---

## IMPLEMENT

### Entry Conditions
- `fsm.current_state == "IMPLEMENT"`
- `guard_flags.idea_logged == true`

### Mandatory Actions
1. Read current best kernel source
2. Implement the optimization idea
3. Save as `tuning/srcs/<KEY>/iter<NNN>_<tag>.cu` (or `.co`)
4. Compile for target shape
5. If compile fails: fix and retry up to 3 times
6. Set `guard_flags.compile_succeeded = true`
7. Run correctness verification (unless `--skip-verify` is appropriate for this kernel type)
8. Set `guard_flags.correctness_verified = true`

### Exit → MEASURE
- `guard_flags.compile_succeeded == true`
- `guard_flags.correctness_verified == true`
- New kernel file exists on disk

### On Failure
- Compile error (3 retries failed): log as DISCARD with reason "compile_failure", skip MEASURE/DECIDE, go directly to STORE with `this_iter_decision = "DISCARD_COMPILE_FAIL"`
- Correctness failure: log as DISCARD with reason "incorrect", skip MEASURE/DECIDE, go to STORE

---

## MEASURE

### Entry Conditions
- `fsm.current_state == "MEASURE"`
- `guard_flags.compile_succeeded == true`

### Mandatory Actions
1. Run GPU health check (abbreviated: just check utilization + memory)
2. Run timing benchmark: `CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500`
3. Capture output to `tuning/perf/<KEY>/timing_iter<NNN>.txt`
4. Parse TFLOPS from output
5. Set `metrics.this_iter_tflops = <value>`
6. Set `guard_flags.timing_captured = true`
7. Apply sanity checks:
   - If TFLOPS > 1.5 × current_best: suspicious, re-run both kernels
   - If TFLOPS < 0.5 × current_best: suspect GPU contention, check health, re-run
   - Use re-run numbers if original was suspicious

### Exit → DECIDE
- `guard_flags.timing_captured == true`
- `metrics.this_iter_tflops != null`
- Timing file exists on disk

### On Failure
- Kernel crashes at runtime: log as DISCARD with "runtime_crash", go to STORE
- GPU unhealthy: remediate, re-run
- Timeout (>2 min for 500 repeats): kill, re-run with 100 repeats

---

## DECIDE

### Entry Conditions
- `fsm.current_state == "DECIDE"`
- `guard_flags.timing_captured == true`

### Mandatory Actions
1. Compare `metrics.this_iter_tflops` vs `metrics.current_best_tflops`
2. If `this_iter_tflops > current_best_tflops`:
   - Decision = KEEP
   - Update `current_best_tflops`, `current_best_iter`, `current_best_kernel`
   - Reset `consecutive_discards = 0`
3. If `this_iter_tflops <= current_best_tflops`:
   - Decision = DISCARD
   - Increment `consecutive_discards`
4. Set `metrics.this_iter_decision = "KEEP"` or `"DISCARD"`
5. Set `guard_flags.decision_made = true`

### Exit → STORE
- `guard_flags.decision_made == true`
- `metrics.this_iter_decision != null`

### On Failure
- None expected at this step (pure logic)

---

## STORE

### Entry Conditions
- `fsm.current_state == "STORE"`
- `guard_flags.decision_made == true`

### Mandatory Actions (ALL required, in order)
1. Append row to `tuning/logs/<KEY>/results.tsv`
   - Columns: iter, kernel, tflops, hw_eff_pct, decision, bottleneck, idea_summary, run_command
2. Set `guard_flags.results_appended = true`
3. Write checkpoint `tuning/checkpoints/<KEY>.json`
4. Set `guard_flags.checkpoint_written = true`
5. Update `tuning/state.json` with current progress
6. Update `compaction-summary.md` with latest iteration info
7. `git add tuning/logs/<KEY>/ tuning/srcs/<KEY>/ tuning/perf/<KEY>/ tuning/checkpoints/<KEY>.json tuning/state.json`
8. `git commit -m "<KEY> iter<NNN>: <idea> — TFLOPS: X -> Y (<KEEP|DISCARD>)"`
9. Set `guard_flags.git_committed = true`
10. Determine next state:
    - If `iteration >= max_iteration` → transition to `SHAPE_COMPLETE`
    - Else → transition to `PROFILE` (auto-increments `fsm.iteration`)

### Exit → PROFILE or SHAPE_COMPLETE
- `guard_flags.results_appended == true`
- `guard_flags.checkpoint_written == true`
- `guard_flags.git_committed == true`

### On Failure
- Git commit fails: check for uncommitted conflicts, retry
- File write fails: check disk space, retry

---

## SHAPE_COMPLETE

### Entry Conditions
- `fsm.iteration >= fsm.max_iteration`

### Mandatory Actions
1. Copy best kernel to `kernels/gemm_sp_<dtype>/<KEY>_best.cu`
2. Update `tuning/state.json`: set this shape to `status: "done"`
3. Final checkpoint write
4. Git commit: `"<KEY>: completed <mode>, best TFLOPS: X at iter<N>"`
5. Run `post-step-check.sh SHAPE_COMPLETE` to verify all conditions met
6. **IMMEDIATELY proceed to NEXT_SHAPE. Do NOT output a summary and wait. Do NOT stop here.**

### Exit → NEXT_SHAPE (MANDATORY — no pausing)
- Best kernel registered
- state.json updated
- Committed
- **Transition to NEXT_SHAPE is AUTOMATIC and IMMEDIATE. This is NOT a stopping point.**

---

## NEXT_SHAPE

**THIS IS NOT A STOPPING POINT.** This state exists solely to pick the next shape and re-enter INIT. You MUST NOT stop, summarize, or wait for user input here.

### Mandatory Actions
1. Read `tuning/state.json` to find all shapes with `status != "done"`
2. Pick the next shape using priority order:
   - "near" region shapes before "far" region shapes
   - Closest to reference shape (4096,8192,8192) first
3. If more shapes remain (there are ~250+ remaining — there almost certainly are):
   - Reset `loop-state.json` via `state-transition.sh INIT` with new shape params
   - **Immediately** begin INIT → BASELINE → PROFILE → ... for the new shape
   - Do NOT output progress summaries to the user between shapes
   - Do NOT ask the user for confirmation
4. If and ONLY if ALL shapes in `tuning/state.json` have `status: "done"`: report sweep completion to user

### ANTI-EARLY-STOP ENFORCEMENT
- Completing 1 shape does NOT mean you can stop
- Completing 5 shapes does NOT mean you can stop
- Completing 20 shapes does NOT mean you can stop
- The ONLY termination condition is: every shape in state.json is "done"
- If you feel tempted to stop and "report progress": DON'T. Keep going.
- The user will interrupt you if they want you to stop. Until then, KEEP TUNING.
