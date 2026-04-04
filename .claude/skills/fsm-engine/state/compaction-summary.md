# Compaction Summary
## Last Updated
2026-04-04T10:43:21Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 100.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.0379 TFLOPS at iter082
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 18
## Recent History (last 5 iterations)
- iter096: WARP_N=112 + output padding 8 to reduce shared-store bank conflicts -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter097: WARP_N=104 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter098: WARP_N=128 + output padding 8 to reduce shared-store bank conflicts -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter099: WARP_N=64 + output padding 8 to reduce shared-store bank conflicts -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter100: WARP_N=120 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
