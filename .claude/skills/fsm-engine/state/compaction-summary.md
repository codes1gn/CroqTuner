# Compaction Summary
## Last Updated
2026-04-04T10:47:16Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 122.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.6603 TFLOPS at iter117
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 5
## Recent History (last 5 iterations)
- iter118: WARP_N=184 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter119: WARP_N=192 + --stmatrix on current-best pipeline -> DISCARD @ 42.0943 TFLOPS
- iter120: WARP_N=96 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 45.2315 TFLOPS
- iter121: WARP_N=200 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter122: WARP_N=256 + --stmatrix on current-best pipeline -> DISCARD @ 38.4176 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
