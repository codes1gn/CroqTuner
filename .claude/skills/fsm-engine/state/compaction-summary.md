# Compaction Summary
## Last Updated
2026-04-04T10:42:50Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `STORE` at iteration 94.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.0379 TFLOPS at iter082
- Baseline: 34.92 TFLOPS
- Last bottleneck: compute_bound
- Consecutive discards: 12
## Recent History (last 5 iterations)
- iter090: RHS swiz<64> instead of swiz<128> — incompatible with WARP_N=96 config -> DISCARD @ 0.0 TFLOPS
- iter091: dma.copy output instead of tma.copy — eliminate TMA overhead -> DISCARD @ 45.4446 TFLOPS
- iter092: WARP_N=256 (24 CTAs, maximum N tile) — 10 consecutive discards, radical tile shape change -> DISCARD @ 37.5609 TFLOPS
- iter093: WARP_N=80 EDSL tile variant from current best to probe compute-bound tile width sensitivity -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
- iter094: WARP_N=112 legal sparse-MMA tile-width probe from current best -> DISCARD_COMPILE_FAIL @ 0.0000 TFLOPS
## DO NOT
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
