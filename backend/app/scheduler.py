"""Heartbeat scheduler: picks pending tasks and dispatches them one at a time."""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from .agent import poll_artifacts, run_task, terminate_stray_opencode_processes
from .database import async_session
from .events import event_bus
from .models import Task
from .runtime_settings import get_default_model

logger = logging.getLogger("croqtuner.scheduler")


class Scheduler:
    def __init__(self) -> None:
        self.running = False
        self.active_task_id: int | None = None
        self._task: asyncio.Task | None = None
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        self.running = True
        await self._recover_stale_tasks()
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _recover_stale_tasks(self) -> None:
        """On startup, handle tasks that were 'running' when the server last died.

        If an opencode process is still alive for a running task, adopt it
        via artifact polling instead of killing it and re-queuing.
        """
        live_pids = _find_live_opencode_pids()

        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.status == "running")
            )
            running_tasks = result.scalars().all()

            adopted_task: Task | None = None
            for task in running_tasks:
                if live_pids and adopted_task is None:
                    adopted_task = task
                    logger.info(
                        "Adopting live task %d (%s) — opencode still running (PIDs %s)",
                        task.id, task.shape_key, live_pids,
                    )
                else:
                    task.status = "pending"
                    task.updated_at = datetime.now(timezone.utc)
                    logger.info("Recovered stale task %d (%s) -> pending", task.id, task.shape_key)

            await session.commit()

        if adopted_task is not None:
            self.active_task_id = adopted_task.id
            task_snapshot = Task(
                id=adopted_task.id,
                shape_key=adopted_task.shape_key,
                dtype=adopted_task.dtype,
                m=adopted_task.m,
                n=adopted_task.n,
                k=adopted_task.k,
                mode=adopted_task.mode,
                max_iterations=adopted_task.max_iterations,
                status=adopted_task.status,
                current_iteration=adopted_task.current_iteration,
                best_tflops=adopted_task.best_tflops,
                baseline_tflops=adopted_task.baseline_tflops,
                best_kernel=adopted_task.best_kernel,
                model=adopted_task.model,
                opencode_session_id=adopted_task.opencode_session_id,
            )
            self._worker = asyncio.create_task(self._adopt_worker(task_snapshot))

    async def _loop(self) -> None:
        while self.running:
            try:
                if self._worker is None or self._worker.done():
                    self._worker = None
                    self.active_task_id = None
                    await self._try_dispatch()
            except Exception:
                logger.exception("Scheduler loop error")
            await asyncio.sleep(5)

    async def _try_dispatch(self) -> None:
        async with async_session() as session:
            result = await session.execute(
                select(Task)
                .where(Task.status == "pending")
                .order_by(Task.created_at.asc())
                .limit(1)
            )
            task = result.scalar_one_or_none()
            if task is None:
                return

            task.status = "running"
            if not task.model:
                task.model = await get_default_model(session)
            task.started_at = datetime.now(timezone.utc)
            task.updated_at = datetime.now(timezone.utc)
            await session.commit()

            self.active_task_id = task.id
            logger.info("Dispatching task %d (%s)", task.id, task.shape_key)
            await event_bus.publish("task_update", task.to_dict())

            task_snapshot = Task(
                id=task.id,
                shape_key=task.shape_key,
                dtype=task.dtype,
                m=task.m,
                n=task.n,
                k=task.k,
                mode=task.mode,
                max_iterations=task.max_iterations,
                status=task.status,
                current_iteration=task.current_iteration,
                best_tflops=task.best_tflops,
                baseline_tflops=task.baseline_tflops,
                best_kernel=task.best_kernel,
                model=task.model,
                opencode_session_id=task.opencode_session_id,
            )

        self._worker = asyncio.create_task(self._run_worker(task_snapshot))

    async def _run_worker(self, task: Task) -> None:
        try:
            exit_code = await run_task(task, async_session)
        except asyncio.CancelledError:
            logger.info("Worker for task %d cancelled", task.id)
            return
        except Exception:
            logger.exception("Worker for task %d crashed", task.id)
            exit_code = -1

        async with async_session() as session:
            db_task = await session.get(Task, task.id)
            if db_task:
                if db_task.status == "cancelled":
                    pass
                elif db_task.current_iteration >= db_task.max_iterations:
                    db_task.status = "completed"
                    db_task.completed_at = datetime.now(timezone.utc)
                elif exit_code == 0:
                    db_task.status = "failed"
                    if not db_task.error_message:
                        db_task.error_message = "opencode exited without completing or persisting tuning progress"
                else:
                    db_task.status = "failed"
                    if exit_code != 0 and not db_task.error_message:
                        db_task.error_message = f"opencode exited with code {exit_code}"
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())

        self.active_task_id = None
        logger.info("Task %d finished with exit_code=%s", task.id, exit_code)


    async def _adopt_worker(self, task: Task) -> None:
        """Poll artifacts for a task whose opencode subprocess is still running
        from before the backend restarted.  When the external process exits,
        do the same final bookkeeping as _run_worker."""
        from .config import settings

        logger.info("Adopt-worker: polling artifacts for task %d (%s)", task.id, task.shape_key)
        try:
            while True:
                await asyncio.sleep(settings.heartbeat_sec)

                async with async_session() as session:
                    db_task = await session.get(Task, task.id)
                    if db_task and db_task.status == "cancelled":
                        terminated = terminate_stray_opencode_processes()
                        logger.info("Adopt-worker: task %d cancelled, terminated %s", task.id, terminated)
                        return

                await poll_artifacts(task, async_session)

                live = _find_live_opencode_pids()
                if not live:
                    logger.info("Adopt-worker: opencode exited for task %d", task.id)
                    break
        except asyncio.CancelledError:
            logger.info("Adopt-worker for task %d cancelled", task.id)
            return

        await poll_artifacts(task, async_session)

        async with async_session() as session:
            db_task = await session.get(Task, task.id)
            if db_task:
                if db_task.status == "cancelled":
                    pass
                elif db_task.current_iteration >= db_task.max_iterations:
                    db_task.status = "completed"
                    db_task.completed_at = datetime.now(timezone.utc)
                elif db_task.status == "running":
                    db_task.status = "failed"
                    if not db_task.error_message:
                        db_task.error_message = "opencode exited without completing (adopted session)"
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())

        self.active_task_id = None
        logger.info("Adopt-worker finished for task %d", task.id)


def _find_live_opencode_pids() -> list[int]:
    import os
    import subprocess

    from .config import settings

    try:
        result = subprocess.run(
            ["pgrep", "-af", "opencode run --print-logs"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    pids: list[int] = []
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
            os.kill(pid, 0)
            pids.append(pid)
        except ProcessLookupError:
            continue
    return pids


scheduler = Scheduler()
