"""opencode invocation and output monitoring."""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .events import event_bus
from .models import AgentLog, IterationLog, Task

ITER_PATTERN = re.compile(
    r"iter[_]?(\d+).*?(\d+(?:\.\d+)?)\s*TFLOPS.*?(KEEP|DISCARD)",
    re.IGNORECASE,
)
TFLOPS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*TFLOPS", re.IGNORECASE)
BASELINE_PATTERN = re.compile(r"[Bb]aseline.*?(\d+(?:\.\d+)?)\s*TFLOPS")


def build_prompt(task: Task) -> str:
    skills_dir = settings.skills_dir.resolve()
    fsm_skill = skills_dir / "fsm-engine" / "SKILL.md"

    if task.mode == "from_current_best":
        mode_skill = skills_dir / "ai-tune-from-current-best" / "SKILL.md"
    else:
        mode_skill = skills_dir / "ai-tune-from-scratch" / "SKILL.md"

    return (
        f"Read the CroqTuner FSM skill at {fsm_skill}. "
        f"Then read the mode skill at {mode_skill}. "
        f"Tune shape {task.shape_key} (M={task.m}, N={task.n}, K={task.k}, "
        f"dtype={task.dtype}) using mode {task.mode} with max {task.max_iterations} iterations. "
        f"Follow the FSM protocol exactly. Work in {settings.project_dir.resolve()}."
    )


def build_command(task: Task) -> list[str]:
    prompt = build_prompt(task)
    project_dir = str(settings.project_dir.resolve())
    return [settings.opencode_bin, "run", "--print-logs", prompt, project_dir]


async def _read_stream(
    stream: asyncio.StreamReader,
    task_id: int,
    level: str,
    session_factory,
) -> None:
    """Read subprocess output line-by-line, store as AgentLog, parse iterations."""
    while True:
        raw = await stream.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue

        async with session_factory() as session:
            log = AgentLog(task_id=task_id, level=level, message=line)
            session.add(log)

            m = ITER_PATTERN.search(line)
            if m:
                iteration = int(m.group(1))
                tflops = float(m.group(2))
                decision = m.group(3).upper()

                existing = await session.execute(
                    select(IterationLog).where(
                        IterationLog.task_id == task_id,
                        IterationLog.iteration == iteration,
                    )
                )
                if not existing.scalar_one_or_none():
                    iter_log = IterationLog(
                        task_id=task_id,
                        iteration=iteration,
                        tflops=tflops,
                        decision=decision,
                    )
                    session.add(iter_log)

                task = await session.get(Task, task_id)
                if task:
                    task.current_iteration = max(task.current_iteration, iteration)
                    if decision == "KEEP" and (task.best_tflops is None or tflops > task.best_tflops):
                        task.best_tflops = tflops
                    task.updated_at = datetime.now(timezone.utc)

            await session.commit()

            if m:
                await event_bus.publish("iteration", {
                    "task_id": task_id,
                    "iteration": int(m.group(1)),
                    "tflops": float(m.group(2)),
                    "decision": m.group(3).upper(),
                })

        await event_bus.publish("agent_log", {
            "task_id": task_id,
            "level": level,
            "message": line,
        })


async def poll_artifacts(task: Task, session_factory) -> None:
    """Read checkpoint and results.tsv from filesystem to update task state."""
    key = task.shape_key
    if task.mode == "from_scratch" and not key.endswith("_fs"):
        key = f"{key}_fs"

    tuning_dir = settings.tuning_dir.resolve()
    checkpoint_path = tuning_dir / "checkpoints" / f"{key}.json"
    results_path = tuning_dir / "logs" / key / "results.tsv"

    update = {}

    if checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text())
            update["current_iteration"] = cp.get("current_iter", task.current_iteration)
            update["best_tflops"] = cp.get("best_tflops", task.best_tflops)
            update["baseline_tflops"] = cp.get("baseline_tflops", task.baseline_tflops)
            update["best_kernel"] = cp.get("best_kernel", task.best_kernel)
            if cp.get("status") == "done" or (
                cp.get("current_iter", 0) >= task.max_iterations
            ):
                update["status"] = "completed"
                update["completed_at"] = datetime.now(timezone.utc)
        except (json.JSONDecodeError, OSError):
            pass

    if results_path.exists():
        try:
            lines = results_path.read_text().strip().split("\n")
            data_lines = [l for l in lines if l and not l.startswith("#") and not l.startswith("iter\t")]
            if data_lines:
                last = data_lines[-1].split("\t")
                if len(last) >= 3:
                    try:
                        iter_num = int(last[0])
                        tflops = float(last[2])
                        update.setdefault("current_iteration", iter_num)
                        if update.get("current_iteration", 0) < iter_num:
                            update["current_iteration"] = iter_num
                    except ValueError:
                        pass
        except OSError:
            pass

    if update:
        async with session_factory() as session:
            t = await session.get(Task, task.id)
            if t:
                changed = False
                for attr, val in update.items():
                    old = getattr(t, attr, None)
                    if old != val:
                        setattr(t, attr, val)
                        changed = True
                if changed:
                    t.updated_at = datetime.now(timezone.utc)
                    await session.commit()
                    await event_bus.publish("task_update", t.to_dict())


async def run_task(task: Task, session_factory) -> int:
    """Launch opencode subprocess, monitor output, return exit code."""
    if settings.mock_mode:
        cmd = ["python3", str(Path(__file__).parent.parent.parent / "scripts" / "mock_opencode.py"),
               task.shape_key, str(task.max_iterations)]
    else:
        cmd = build_command(task)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_task = asyncio.create_task(
        _read_stream(proc.stdout, task.id, "info", session_factory)
    )
    stderr_task = asyncio.create_task(
        _read_stream(proc.stderr, task.id, "error", session_factory)
    )

    poll_task = asyncio.create_task(_poll_loop(task, session_factory, proc))

    await asyncio.gather(stdout_task, stderr_task)
    exit_code = await proc.wait()
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    await poll_artifacts(task, session_factory)
    return exit_code


async def _poll_loop(task: Task, session_factory, proc: asyncio.subprocess.Process) -> None:
    """Periodically poll filesystem artifacts while process is alive."""
    try:
        while proc.returncode is None:
            await asyncio.sleep(settings.heartbeat_sec)
            await poll_artifacts(task, session_factory)
    except asyncio.CancelledError:
        return
