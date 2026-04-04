# Compaction Summary
## Last Updated
2026-04-04T10:52:19Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 134.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 48.5670 TFLOPS at iter129
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 5
## Recent History (last 5 iterations)
- iter130: WARP_N=224 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter131: WARP_N=256 + --hoist-scale on current-best pipeline -> DISCARD @ 38.5090 TFLOPS
- iter132: WARP_N=112 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter133: WARP_N=232 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter134: WARP_N=128 without --ptx-barrier to test barrier overhead sensitivity -> DISCARD @ 42.9524 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
