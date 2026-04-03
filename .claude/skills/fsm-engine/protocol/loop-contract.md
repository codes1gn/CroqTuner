# CroqTuner — Loop Contract (FSM Definition)

## State Machine

The tuning loop is a finite state machine. The current state is stored in `loop-state.json` under `fsm.current_state`. You MUST only perform actions corresponding to the current state, then transition to the next state using `state-transition.sh`.

```
┌──────┐     ┌──────────┐     ┌─────────┐     ┌────────┐     ┌───────────┐
│ INIT │────▶│ BASELINE │────▶│ PROFILE │────▶│ IDEATE │────▶│ IMPLEMENT │
└──────┘     └──────────┘     └─────────┘     └────────┘     └─────┬─────┘
                                   ▲                               │
                                   │                               ▼
                              ┌────┴──┐     ┌────────┐     ┌──────────┐
                              │ STORE │◀────│ DECIDE │◀────│ MEASURE  │
                              └───┬───┘     └────────┘     └──────────┘
                                  │
                    ┌─────────────┼─────────────────┐
                    ▼             ▼                  ▼
              ┌──────────┐  ┌───────────────┐  ┌────────────┐
              │ PROFILE  │  │ SHAPE_COMPLETE│  │ NEXT_SHAPE │
              │ (loop)   │  │ (iter >= max) │  │   → INIT   │
              └──────────┘  └───────────────┘  └────────────┘
```

## States

| State | Purpose | Next State |
|---|---|---|
| `INIT` | Create directories, copy seed kernel, init results.tsv | `BASELINE` |
| `BASELINE` | Compile + measure seed/baseline kernel (iter 0) | `PROFILE` |
| `PROFILE` | Run ncu (or skip if allowed), identify bottleneck | `IDEATE` |
| `IDEATE` | Propose ONE novel optimization idea grounded in data | `IMPLEMENT` |
| `IMPLEMENT` | Create new kernel file, compile, verify correctness | `MEASURE` |
| `MEASURE` | Run timing benchmark, capture TFLOPS | `DECIDE` |
| `DECIDE` | Compare to current best: KEEP or DISCARD | `STORE` |
| `STORE` | Append results.tsv, write checkpoint, git commit, update FSM | `PROFILE` or `SHAPE_COMPLETE` |
| `SHAPE_COMPLETE` | Register best kernel, update state.json, commit | `NEXT_SHAPE` |
| `NEXT_SHAPE` | Pick next shape from schedule, reset FSM | `INIT` |

## Iteration Semantics

**`fsm.iteration` = the iteration number currently being worked on (or just completed).**

- After INIT: `iteration = 0` (baseline)
- During BASELINE: working on iter 0
- After STORE for iter 1: `iteration = 1`
- Transition STORE → PROFILE auto-increments: `iteration` becomes 2
- Files are named with `iter<NNN>` where NNN = `fsm.iteration` (e.g., `iter001_warpn128.cu`)

## Transition Rules

After STORE:
- If `iteration < max_iteration` → transition to `PROFILE` (auto-increments iteration)
- If `iteration >= max_iteration` → transition to `SHAPE_COMPLETE`

After SHAPE_COMPLETE:
- If more shapes remain in sweep → transition to `NEXT_SHAPE` then `INIT` **IMMEDIATELY — do NOT pause, summarize, or wait for user input**
- If no shapes remain → sweep is done, report to user
- **Completing a shape is NOT a stopping point.** There are ~260 shapes. Keep going.

## loop-state.json Schema

```json
{
  "schema_version": 1,
  "fsm": {
    "current_state": "<STATE>",
    "iteration": 0,
    "max_iteration": 30,
    "shape_key": "f16_4096x16384x16384",
    "dtype": "f16",
    "shape": [4096, 16384, 16384],
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
    "shape_dir_logs": "tuning/logs/<KEY>",
    "shape_dir_srcs": "tuning/srcs/<KEY>",
    "shape_dir_perf": "tuning/perf/<KEY>",
    "checkpoint_file": "tuning/checkpoints/<KEY>.json",
    "idea_log": "tuning/logs/<KEY>/idea-log.jsonl"
  },
  "completion_promise": "iteration >= max_iteration AND best kernel registered AND state.json updated",
  "last_updated": "2026-04-03T00:00:00Z"
}
```

## How to Transition

Run:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh <NEXT_STATE> [key=value ...]
```

Example:
```bash
bash .claude/skills/fsm-engine/scripts/state-transition.sh IDEATE
bash .claude/skills/fsm-engine/scripts/state-transition.sh STORE decision_made=true this_iter_tflops=450.2
```

The script atomically updates `loop-state.json`, resets guard flags for the new state, and validates the transition is legal.
