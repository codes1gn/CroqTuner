# Compaction Summary
## Last Updated
2026-04-04T10:51:29Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 131.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 48.5670 TFLOPS at iter129
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 2
## Recent History (last 5 iterations)
- iter127: WARP_N=216 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter128: WARP_N=192 + --hoist-scale on current-best pipeline -> DISCARD @ 42.0941 TFLOPS
- iter129: WARP_N=96 without --native-f16 to test native arithmetic benefit -> KEEP @ 48.5670 TFLOPS
- iter130: WARP_N=224 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter131: WARP_N=256 + --hoist-scale on current-best pipeline -> DISCARD @ 38.5090 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
