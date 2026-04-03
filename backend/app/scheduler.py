"""Heartbeat scheduler: picks pending tasks and dispatches them one at a time."""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from .agent import run_task
from .database import async_session
from .events import event_bus
from .models import Task

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
        """On startup, re-queue tasks that were 'running' when the server last died."""
        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.status == "running")
            )
            for task in result.scalars().all():
                task.status = "pending"
                task.updated_at = datetime.now(timezone.utc)
                logger.info("Recovered stale task %d (%s) -> pending", task.id, task.shape_key)
            await session.commit()

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
                elif exit_code == 0 or db_task.current_iteration >= db_task.max_iterations:
                    db_task.status = "completed"
                    db_task.completed_at = datetime.now(timezone.utc)
                else:
                    db_task.status = "failed"
                    if exit_code != 0:
                        db_task.error_message = f"opencode exited with code {exit_code}"
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())

        self.active_task_id = None
        logger.info("Task %d finished with exit_code=%s", task.id, exit_code)


scheduler = Scheduler()
