# Compaction Summary
## Last Updated
2026-04-04T10:46:27Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 120.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.6603 TFLOPS at iter117
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 3
## Recent History (last 5 iterations)
- iter116: WARP_N=128 + --stmatrix on current-best pipeline -> DISCARD @ 44.7735 TFLOPS
- iter117: WARP_N=64 + --hoist-scale on current-best pipeline -> KEEP @ 47.6603 TFLOPS
- iter118: WARP_N=184 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter119: WARP_N=192 + --stmatrix on current-best pipeline -> DISCARD @ 42.0943 TFLOPS
- iter120: WARP_N=96 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 45.2315 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
