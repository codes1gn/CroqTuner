# Compaction Summary
## Last Updated
2026-04-04T10:43:56Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 114.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.0379 TFLOPS at iter082
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 32
## Recent History (last 5 iterations)
- iter110: WARP_N=192 + output padding 16 for more aggressive epilogue bank-conflict relief -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter111: WARP_N=96 + output padding 16 for more aggressive epilogue bank-conflict relief -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter112: WARP_N=168 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter113: WARP_N=256 + output padding 16 for more aggressive epilogue bank-conflict relief -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter114: WARP_N=112 + --stmatrix on current-best pipeline -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
