import pytest
import pytest_asyncio
from sqlalchemy import select

from app.models import AgentLog, IterationLog, Task


@pytest.mark.asyncio
async def test_create_task(db_session):
    task = Task(
        shape_key="f16_768x768x768",
        dtype="f16",
        m=768,
        n=768,
        k=768,
        mode="from_current_best",
        max_iterations=30,
        status="pending",
        current_iteration=0,
    )
    db_session.add(task)
    await db_session.commit()

    result = await db_session.execute(select(Task).where(Task.shape_key == "f16_768x768x768"))
    fetched = result.scalar_one()
    assert fetched.dtype == "f16"
    assert fetched.m == 768
    assert fetched.status == "pending"
    assert fetched.max_iterations == 30


@pytest.mark.asyncio
async def test_task_to_dict(db_session):
    task = Task(
        shape_key="e4m3_512x8192x8192_fs",
        dtype="e4m3",
        m=512,
        n=8192,
        k=8192,
        mode="from_scratch",
        max_iterations=150,
        status="running",
        current_iteration=42,
        best_tflops=123.4,
    )
    db_session.add(task)
    await db_session.commit()

    d = task.to_dict()
    assert d["shape_key"] == "e4m3_512x8192x8192_fs"
    assert d["status"] == "running"
    assert d["best_tflops"] == 123.4
    assert d["current_iteration"] == 42
    assert "created_at" in d


@pytest.mark.asyncio
async def test_iteration_log(db_session):
    task = Task(
        shape_key="f16_256x512x256",
        dtype="f16",
        m=256, n=512, k=256,
        mode="from_current_best",
        max_iterations=30,
    )
    db_session.add(task)
    await db_session.commit()

    log = IterationLog(
        task_id=task.id,
        iteration=5,
        tflops=67.8,
        decision="KEEP",
        bottleneck="smem_bw",
        idea_summary="swizzle 128",
    )
    db_session.add(log)
    await db_session.commit()

    result = await db_session.execute(
        select(IterationLog).where(IterationLog.task_id == task.id)
    )
    fetched = result.scalar_one()
    assert fetched.iteration == 5
    assert fetched.tflops == 67.8
    assert fetched.decision == "KEEP"


@pytest.mark.asyncio
async def test_agent_log(db_session):
    task = Task(
        shape_key="f16_1024x1024x1024",
        dtype="f16",
        m=1024, n=1024, k=1024,
        mode="from_scratch",
        max_iterations=150,
    )
    db_session.add(task)
    await db_session.commit()

    log = AgentLog(task_id=task.id, level="info", message="Compiling iter001...")
    db_session.add(log)
    await db_session.commit()

    d = log.to_dict()
    assert d["level"] == "info"
    assert "Compiling" in d["message"]
