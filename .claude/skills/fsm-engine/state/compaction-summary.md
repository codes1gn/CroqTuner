# Compaction Summary
## Last Updated
2026-04-04T10:44:42Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 117.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.6603 TFLOPS at iter117
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 0
## Recent History (last 5 iterations)
- iter113: WARP_N=256 + output padding 16 for more aggressive epilogue bank-conflict relief -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter114: WARP_N=112 + --stmatrix on current-best pipeline -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter115: WARP_N=176 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter116: WARP_N=128 + --stmatrix on current-best pipeline -> DISCARD @ 44.7735 TFLOPS
- iter117: WARP_N=64 + --hoist-scale on current-best pipeline -> KEEP @ 47.6603 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
