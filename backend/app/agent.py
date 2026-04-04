"""opencode invocation and output monitoring."""

import asyncio
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
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
SESSION_ID_PATTERNS = (
    re.compile(r"sessionID=([A-Za-z0-9_]+)"),
    re.compile(r"/session/([A-Za-z0-9_]+)/message"),
)
FATAL_ERROR_PATTERN = re.compile(
    r"(ProviderModelNotFoundError|Model not found:|EACCES:|stream error|(?:^|\s)fatal error:)",
    re.IGNORECASE,
)


@dataclass
class RunState:
    fatal_error_message: str | None = None


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
        "Follow the FSM protocol exactly. "
        "Do not stop after a single compile, measure, or decide step. "
        "You must persist protocol state updates to the CroqTuner artifacts, execute STORE when required, "
        "and continue looping until the task is genuinely complete according to the FSM completion promise or an unrecoverable error occurs. "
        f"Work in {settings.project_dir.resolve()}."
    )


def build_command(task: Task) -> list[str]:
    prompt = build_prompt(task)
    project_dir = str(settings.project_dir.resolve())
    command = [settings.opencode_bin, "run", "--print-logs"]
    model = task.model or settings.opencode_model
    if model:
        command.extend(["--model", model])
    command.extend([prompt, project_dir])
    return command


def is_fatal_agent_error(line: str) -> bool:
    return FATAL_ERROR_PATTERN.search(line) is not None


def extract_session_id(line: str) -> str | None:
    for pattern in SESSION_ID_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


async def _iter_stream_lines(stream: asyncio.StreamReader):
    buffer = b""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buffer += chunk
        while True:
            newline_index = buffer.find(b"\n")
            if newline_index == -1:
                if len(buffer) > 65536:
                    yield buffer
                    buffer = b""
                break
            yield buffer[:newline_index]
            buffer = buffer[newline_index + 1 :]

    if buffer:
        yield buffer


async def _read_stream(
    stream: asyncio.StreamReader,
    task_id: int,
    level: str,
    session_factory,
    run_state: RunState,
) -> None:
    """Read subprocess output line-by-line, store as AgentLog, parse iterations."""
    async for raw in _iter_stream_lines(stream):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue

        if run_state.fatal_error_message is None and is_fatal_agent_error(line):
            run_state.fatal_error_message = line

        async with session_factory() as session:
            log = AgentLog(task_id=task_id, level=level, message=line)
            session.add(log)
            task_updated = False
            task = None

            session_id = extract_session_id(line)
            if session_id:
                task = await session.get(Task, task_id)
                if task and task.opencode_session_id != session_id:
                    task.opencode_session_id = session_id
                    task.updated_at = datetime.now(timezone.utc)
                    task_updated = True

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

                if task is None:
                    task = await session.get(Task, task_id)
                if task:
                    task.current_iteration = max(task.current_iteration, iteration)
                    if decision == "KEEP" and (task.best_tflops is None or tflops > task.best_tflops):
                        task.best_tflops = tflops
                    task.updated_at = datetime.now(timezone.utc)
                    task_updated = True

            await session.commit()

            if m:
                await event_bus.publish("iteration", {
                    "task_id": task_id,
                    "iteration": int(m.group(1)),
                    "tflops": float(m.group(2)),
                    "decision": m.group(3).upper(),
                })

            if task_updated and task is not None:
                await event_bus.publish("task_update", task.to_dict())

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
    state_path = tuning_dir / "state.json"

    update = {}

    if state_path.exists():
        try:
            state_data = json.loads(state_path.read_text())
            shape_state = state_data.get("shapes", {}).get(key, {})
            if shape_state:
                update["current_iteration"] = shape_state.get("current_iter", task.current_iteration)
                state_best = shape_state.get("best_tflops")
                if state_best not in (None, ""):
                    update["best_tflops"] = state_best
                state_baseline = shape_state.get("baseline_tflops")
                if state_baseline not in (None, ""):
                    update["baseline_tflops"] = state_baseline
                state_best_kernel = shape_state.get("best_kernel")
                if state_best_kernel:
                    update["best_kernel"] = state_best_kernel
                if shape_state.get("status") == "done":
                    update["status"] = "completed"
                    update["completed_at"] = datetime.now(timezone.utc)
        except (json.JSONDecodeError, OSError):
            pass

    if checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text())
            update["current_iteration"] = cp.get("current_iter", task.current_iteration)
            cp_best = cp.get("best_tflops")
            if cp_best not in (None, ""):
                update["best_tflops"] = cp_best
            cp_baseline = cp.get("baseline_tflops")
            if cp_baseline not in (None, ""):
                update["baseline_tflops"] = cp_baseline
            cp_best_kernel = cp.get("best_kernel")
            if cp_best_kernel:
                update["best_kernel"] = cp_best_kernel
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
    run_state = RunState()
    if settings.mock_mode:
        cmd = ["python3", str(Path(__file__).parent.parent.parent / "scripts" / "mock_opencode.py"),
               task.shape_key, str(task.max_iterations)]
    else:
        cmd = build_command(task)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(settings.project_dir.resolve()),
        start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": settings.cuda_visible_devices},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_task = asyncio.create_task(
        _read_stream(proc.stdout, task.id, "info", session_factory, run_state)
    )
    stderr_task = asyncio.create_task(
        _read_stream(proc.stderr, task.id, "error", session_factory, run_state)
    )

    poll_task = asyncio.create_task(_poll_loop(task, session_factory, proc))

    try:
        await asyncio.gather(stdout_task, stderr_task)
        exit_code = await proc.wait()
    except asyncio.CancelledError:
        await _terminate_process(proc)
        raise
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

    await poll_artifacts(task, session_factory)
    if run_state.fatal_error_message:
        async with session_factory() as session:
            db_task = await session.get(Task, task.id)
            if db_task:
                db_task.error_message = run_state.fatal_error_message
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())
        return exit_code if exit_code != 0 else 1
    return exit_code


async def _poll_loop(task: Task, session_factory, proc: asyncio.subprocess.Process) -> None:
    """Periodically poll filesystem artifacts while process is alive."""
    try:
        while proc.returncode is None:
            await asyncio.sleep(settings.heartbeat_sec)
            async with session_factory() as session:
                db_task = await session.get(Task, task.id)
                if db_task and db_task.status == "cancelled":
                    await _terminate_process(proc)
                    return
            await poll_artifacts(task, session_factory)
    except asyncio.CancelledError:
        return


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
        return
    except asyncio.TimeoutError:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await proc.wait()


def terminate_stray_opencode_processes() -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "opencode run --print-logs"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    terminated: list[int] = []
    project_root = str(settings.project_dir.resolve())
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or project_root not in line:
            continue
        pid_text, _, cmdline = line.partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if "opencode run --print-logs" not in cmdline:
            continue
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            continue

    deadline = time.time() + 10
    alive = set(terminated)
    while alive and time.time() < deadline:
        remaining = set()
        for pid in alive:
            try:
                os.kill(pid, 0)
                remaining.add(pid)
            except ProcessLookupError:
                continue
        alive = remaining
        if alive:
            time.sleep(0.2)

    for pid in list(alive):
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue

    return terminated
