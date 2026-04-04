# Compaction Summary
## Last Updated
2026-04-04T10:57:48Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 149.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 48.5670 TFLOPS at iter129
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 20
## Recent History (last 5 iterations)
- iter145: WARP_N=96 + LHS swiz<128> to test packed-A layout at wider swizzle -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter146: WARP_N=256 + RHS swiz<64> to shift shared-memory traffic pattern -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter147: WARP_N=112 + LHS swiz<128> to test packed-A layout at wider swizzle -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter148: WARP_N=128 + swap LHS/RHS swizzle factors (128/64) -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter149: WARP_N=64 + dma.copy output store instead of TMA epilogue -> DISCARD @ 48.2918 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
