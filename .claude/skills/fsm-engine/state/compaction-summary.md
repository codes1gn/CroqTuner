# Compaction Summary
## Last Updated
2026-04-04T10:47:43Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 124.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.6603 TFLOPS at iter117
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 7
## Recent History (last 5 iterations)
- iter120: WARP_N=96 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 45.2315 TFLOPS
- iter121: WARP_N=200 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter122: WARP_N=256 + --stmatrix on current-best pipeline -> DISCARD @ 38.4176 TFLOPS
- iter123: WARP_N=112 + --hoist-scale on current-best pipeline -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter124: WARP_N=208 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
