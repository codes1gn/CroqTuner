# CroqTuner — Compaction & Resume Protocol

This protocol ensures zero information loss across context compactions. It is modeled on the claw-code `compact.rs` mergeable summary pattern.

## When Context Compaction Happens

Context compaction may occur at any point during tuning. When it happens, you lose conversational context but retain all files on disk. This protocol ensures you can resume exactly where you left off.

## Pre-Compaction Checklist (before context fills)

If you detect context is getting large (many iterations without compaction), proactively:

1. Ensure ALL pending work is committed to git
2. Ensure `loop-state.json` reflects the true current state
3. Ensure `compaction-summary.md` is up to date
4. Ensure `idea-log.jsonl` has all ideas logged

## Post-Compaction Resume Protocol

**IMMEDIATELY after compaction (or on fresh skill invocation), execute these steps IN ORDER:**

### Step 1: Read State Files (MANDATORY — do this FIRST)
```
Read these files in order:
1. .claude/skills/fsm-engine/protocol/identity.md        ← who you are, what you must never do
2. .claude/skills/fsm-engine/state/loop-state.json        ← EXACT current FSM state
3. .claude/skills/fsm-engine/state/compaction-summary.md  ← what just happened
```

### Step 2: Determine Resume Point
From `loop-state.json`, read `fsm.current_state` and `fsm.iteration`. This tells you EXACTLY where to resume. Do NOT re-discover state from scratch.

### Step 3: Read Shape-Specific Context
```
Read:
4. tuning/logs/<KEY>/results.tsv                         ← iteration history
5. tuning/logs/<KEY>/idea-log.jsonl                      ← ideas already tried
6. tuning/checkpoints/<KEY>.json                         ← last checkpoint
```

### Step 4: Read Active Step Protocol
```
Read:
7. .claude/skills/fsm-engine/protocol/step-checklists.md  ← ONLY the section for current_state
```

### Step 5: Resume
Continue from the EXACT point indicated by `fsm.current_state`. Do NOT:
- Re-run completed iterations
- Re-baseline
- Re-read the entire skill set (only read what's needed for current step)
- Ask the user what to do

## compaction-summary.md Format

This file is maintained by the tuning loop. It is updated at the end of every STORE step and serves as the human-and-LLM-readable "what happened" log.

```markdown
# Compaction Summary

## Last Updated
<ISO timestamp>

## IMMEDIATE ACTION
Read `loop-state.json` and resume from state `<STATE>` at iteration <N>.

## Current Task
- Shape: <KEY> (<mode>, <max_iter> iters max)
- Current best: <TFLOPS> TFLOPS at iter <N> (<kernel_file>)
- Baseline: <TFLOPS> TFLOPS
- Last bottleneck: <category>
- Consecutive discards: <N> → <strategy advice>

## Recent History (last 5 iterations)
- iter <N>: <idea> → <TFLOPS> (<KEEP|DISCARD>)
- iter <N-1>: <idea> → <TFLOPS> (<KEEP|DISCARD>)
- ...

## Strategy Notes
<any observations about what works/doesn't work for this shape>

## Files to Re-read After Compaction
1. .claude/skills/fsm-engine/state/loop-state.json
2. .claude/skills/fsm-engine/state/compaction-summary.md
3. .claude/skills/fsm-engine/protocol/identity.md
4. tuning/logs/<KEY>/results.tsv
5. tuning/logs/<KEY>/idea-log.jsonl
6. tuning/srcs/<KEY>/<current_best_kernel>

## DO NOT
- Do NOT re-run completed iterations
- Do NOT re-baseline
- Do NOT ask the user what to do — resume autonomously
- Do NOT re-read all protocol files — only the current step
- Do NOT stop after resuming and completing a shape — immediately proceed to NEXT_SHAPE
- Do NOT treat post-compaction as a "fresh session" that needs user approval to continue
```

## Merging Compaction Summaries

If compaction happens multiple times during a single shape, the "Recent History" section grows. Keep at most 10 recent entries. The "Strategy Notes" section accumulates — do not overwrite, append.

## Key Principle

After compaction, the LLM should spend <30 seconds reading state files and then immediately resume the loop — NOT spend 5 minutes re-discovering where it is.

## Sweep Continuation After Compaction

Compaction does NOT change the sweep mandate. After resuming:
- If the current shape is mid-iteration: finish it, then continue to the next shape.
- If the current shape just completed: immediately proceed to the next shape.
- Do NOT treat compaction as a "natural break point" to stop and report.
- The sweep continues non-stop until ALL shapes are done.
