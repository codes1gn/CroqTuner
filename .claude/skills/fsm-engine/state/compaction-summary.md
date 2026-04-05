# Compaction Summary
## Last Updated
2026-04-04T16:03:45Z
## IMMEDIATE ACTION
Read `loop-state.json` and resume from state `PROFILE` at iteration 3.
## Current Task
- Shape: f16_256x4096x4096 (from_current_best, 30 iters max)
- Current best: 134.6440 TFLOPS at iter 1 (tuning/srcs/f16_256x4096x4096/iter001_l2all256.cu)
- Baseline: 131.7060 TFLOPS
- Last bottleneck: smem_throughput
- Consecutive discards: 1
## Recent History (last 5 iterations)
- iter000: baseline established @ 131.7060 TFLOPS
- iter001: KEEP `promote all tensor maps to L2_256B` -> 134.6440 TFLOPS
- iter002 PROFILE: reused prior bottleneck `smem_throughput` (conditional ncu waiver allowed)
- iter002 IMPLEMENT: compiled tuning/srcs/f16_256x4096x4096/iter002_warpn128.cu with manifest NVCC flags
- iter002 STORE: DISCARD @ 133.6690 TFLOPS; current best remains iter001
## DO NOT
- Do NOT re-run iter001/iter002 measurement
- Use manifest NVCC flags (`-std=c++17 ...`) and GPU1 visibility mapping for generated binaries
