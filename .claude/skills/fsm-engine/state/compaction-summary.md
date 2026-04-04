# Compaction Summary
## Last Updated
2026-04-04T10:56:05Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 143.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 48.5670 TFLOPS at iter129
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 14
## Recent History (last 5 iterations)
- iter139: WARP_N=248 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter140: WARP_N=256 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 35.7010 TFLOPS
- iter141: WARP_N=112 + RHS swiz<64> to shift shared-memory traffic pattern -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter142: WARP_N=128 + RHS swiz<64> to shift shared-memory traffic pattern -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter143: WARP_N=64 + swap LHS/RHS swizzle factors (128/64) -> DISCARD_INCORRECT @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
