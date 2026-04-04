# Compaction Summary
## Last Updated
2026-04-04T05:05:00Z
## IMMEDIATE ACTION
Read `loop-state_from_scratch.json` and resume from state `PROFILE` at iteration 88.
## Current Task
- Shape: f16_768x768x768_fs (from_scratch, 150 iters max)
- Current best: 47.0379 TFLOPS at iter082
- Baseline: 34.92 TFLOPS
- Last bottleneck: runtime_crash (TILE_K=128)
- Consecutive discards: 5
## Key Findings
- **--use-warpspec causes ILLEGAL MEMORY ACCESS crashes** for this shape. AVOID completely.
- **ncu times out** (>5 min) for this shape. Set bottleneck from prior knowledge.
- **WARP_N=96 is current best** (47.04 TFLOPS at iter82) with drc+native-f16+apprx-div+ptx-barrier
- **TILE_K=128 causes runtime crash** (CUDA invalid argument) — same as swiz<128> on LHS
- **swiz<32> crashes, swiz<64> works, swiz<128> crashes** — LHS swiz<128> incompatible with TILE_K=64
- **stmatrix REGRESSES** on both WARP_N=64 and WARP_N=96 configs
- **hoist-scale REGRESSES slightly**
- **rtc=low REGRESSES** (45.86 vs 47.04)
- **Column-major scheduling REGRESSES** (45.10)
- **5 consecutive discards** — need radical approach change
## Recent History (last 5 iterations)
- iter83: WARP_N=96+stmatrix → 46.97 DISCARD
- iter84: WARP_N=96+hoist-scale → 47.07 DISCARD (marginal +0.03)
- iter85: WARP_N=96+rtc=low → 45.86 DISCARD
- iter86: WARP_N=96+swiz<128> → runtime_crash DISCARD
- iter87: TILE_K=128 → runtime_crash DISCARD
## DO NOT
- Do NOT use --use-warpspec (crashes)
- Do NOT run ncu (times out) — set bottleneck from prior knowledge
- Do NOT try TILE_K=128 (crashes)
- Do NOT try LHS swiz<128> (crashes)
- Do NOT re-run completed iterations
- Do NOT ask the user what to do — resume autonomously
- Do NOT stop — 62 iterations remain (88/150)
