# Compaction Summary
## Last Updated
2026-04-04T10:48:54Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 127.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.6603 TFLOPS at iter117
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 10
## Recent History (last 5 iterations)
- iter123: WARP_N=112 + --hoist-scale on current-best pipeline -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter124: WARP_N=208 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
- iter125: WARP_N=128 + --hoist-scale on current-best pipeline -> DISCARD @ 46.0048 TFLOPS
- iter126: WARP_N=64 without --native-f16 to test native arithmetic benefit -> DISCARD @ 47.6420 TFLOPS
- iter127: WARP_N=216 legal sparse-MMA tile-width probe from current best -> DISCARD_INCORRECT @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
