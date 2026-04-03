"""CroqTuner Agent Bot — FastAPI backend."""

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from .database import get_session, init_db
from .events import event_bus
from .models import AgentLog, IterationLog, Task
from .scheduler import scheduler
from .schemas import (
    AgentLogResponse,
    HealthResponse,
    IterationLogResponse,
    TaskCreate,
    TaskResponse,
    TaskUpdate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("croqtuner")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(title="CroqTuner Agent Bot", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tasks", response_model=list[TaskResponse])
async def list_tasks(
    status: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Task).order_by(Task.created_at.desc())
    if status:
        stmt = stmt.where(Task.status == status)
    result = await session.execute(stmt)
    return [TaskResponse(**t.to_dict()) for t in result.scalars().all()]


@app.post("/api/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    body: TaskCreate,
    session: AsyncSession = Depends(get_session),
):
    max_iter = 30 if body.mode == "from_current_best" else 150
    suffix = "_fs" if body.mode == "from_scratch" else ""
    shape_key = f"{body.dtype}_{body.m}x{body.n}x{body.k}{suffix}"

    task = Task(
        shape_key=shape_key,
        dtype=body.dtype,
        m=body.m,
        n=body.n,
        k=body.k,
        mode=body.mode,
        max_iterations=max_iter,
        status="pending",
        current_iteration=0,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    await event_bus.publish("task_update", task.to_dict())
    return TaskResponse(**task.to_dict())


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, session: AsyncSession = Depends(get_session)):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return TaskResponse(**task.to_dict())


@app.patch("/api/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    body: TaskUpdate,
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    if body.status == "cancelled":
        if task.status not in ("pending", "running", "waiting"):
            raise HTTPException(400, "Can only cancel pending, running, or waiting tasks")
        task.status = "cancelled"
        task.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await event_bus.publish("task_update", task.to_dict())
    elif body.status == "pending" and task.status == "waiting":
        task.status = "pending"
        task.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await event_bus.publish("task_update", task.to_dict())
    elif body.status == "waiting" and task.status == "pending":
        if scheduler.active_task_id == task.id:
            raise HTTPException(400, "Cannot demote a task that is running")
        task.status = "waiting"
        task.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await event_bus.publish("task_update", task.to_dict())

    return TaskResponse(**task.to_dict())


@app.delete("/api/tasks/{task_id}", status_code=204)
async def delete_task(task_id: int, session: AsyncSession = Depends(get_session)):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status == "running":
        raise HTTPException(400, "Cannot delete a running task; cancel it first")
    if task.status not in ("pending", "waiting", "completed", "failed", "cancelled"):
        raise HTTPException(400, "Cannot delete this task in its current state")
    await session.delete(task)
    await session.commit()
    await event_bus.publish("task_deleted", {"id": task_id})


@app.get("/api/tasks/{task_id}/logs", response_model=list[IterationLogResponse])
async def get_iteration_logs(task_id: int, session: AsyncSession = Depends(get_session)):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    result = await session.execute(
        select(IterationLog)
        .where(IterationLog.task_id == task_id)
        .order_by(IterationLog.iteration.desc())
    )
    return [IterationLogResponse(**log.to_dict()) for log in result.scalars().all()]


@app.get("/api/tasks/{task_id}/agent-logs", response_model=list[AgentLogResponse])
async def get_agent_logs(
    task_id: int,
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    result = await session.execute(
        select(AgentLog)
        .where(AgentLog.task_id == task_id)
        .order_by(AgentLog.timestamp.desc())
        .limit(limit)
    )
    return [AgentLogResponse(**log.to_dict()) for log in result.scalars().all()]


@app.get("/api/events")
async def sse_events():
    return EventSourceResponse(event_bus.subscribe())


@app.get("/api/health", response_model=HealthResponse)
async def health():
    gpu_info = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu_info = "nvidia-smi not available"

    return HealthResponse(
        status="ok",
        scheduler_running=scheduler.running,
        active_task_id=scheduler.active_task_id,
        gpu_info=gpu_info,
    )
