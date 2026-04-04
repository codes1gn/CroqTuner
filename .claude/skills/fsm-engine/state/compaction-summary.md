# Compaction Summary
## Last Updated
2026-04-04T10:43:35Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 104.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.0379 TFLOPS at iter082
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 22
## Recent History (last 5 iterations)
- iter100: WARP_N=120 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter101: WARP_N=192 + output padding 8 to reduce shared-store bank conflicts -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter102: WARP_N=96 + output padding 8 to reduce shared-store bank conflicts -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter103: WARP_N=136 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter104: WARP_N=256 + output padding 8 to reduce shared-store bank conflicts -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
