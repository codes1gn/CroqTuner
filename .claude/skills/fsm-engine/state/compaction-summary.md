# Compaction Summary
## Last Updated
2026-04-04T10:55:41Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 140.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 48.5670 TFLOPS at iter129
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 11
## Recent History (last 5 iterations)
- iter136: WARP_N=240 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter137: WARP_N=192 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 38.9757 TFLOPS
- iter138: WARP_N=96 without --apprx-div and --native-f16 to test math-flag dependence -> DISCARD @ 48.5029 TFLOPS
- iter139: WARP_N=248 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter140: WARP_N=256 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 35.7010 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
