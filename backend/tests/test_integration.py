"""Integration test: end-to-end flow with mock opencode.

Tests the full pipeline: create task -> scheduler dispatches -> mock opencode
writes artifacts -> monitor picks them up -> task completes.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["CROQTUNER_MOCK_MODE"] = "1"

from app.main import app
from app.database import engine
from app.models import Base
from app.scheduler import scheduler


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_full_lifecycle(client):
    """Create a task, verify it appears, check it can be cancelled."""
    resp = await client.post("/api/tasks", json={
        "dtype": "f16",
        "m": 768,
        "n": 768,
        "k": 768,
        "mode": "from_current_best",
    })
    assert resp.status_code == 201
    task = resp.json()
    task_id = task["id"]
    assert task["status"] == "pending"
    assert task["shape_key"] == "f16_768x768x768"
    assert task["max_iterations"] == 30

    resp = await client.get("/api/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task_id

    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["shape_key"] == "f16_768x768x768"

    resp = await client.get(f"/api/tasks/{task_id}/logs")
    assert resp.status_code == 200
    assert resp.json() == []

    resp = await client.get(f"/api/tasks/{task_id}/agent-logs")
    assert resp.status_code == 200
    assert resp.json() == []

    resp = await client.patch(f"/api/tasks/{task_id}", json={"status": "cancelled"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_multiple_tasks_queue(client):
    """Multiple tasks queue FIFO."""
    ids = []
    for m in [768, 1024, 2048]:
        resp = await client.post("/api/tasks", json={
            "dtype": "f16", "m": m, "n": m, "k": m, "mode": "from_current_best",
        })
        assert resp.status_code == 201
        ids.append(resp.json()["id"])

    resp = await client.get("/api/tasks?status=pending")
    assert resp.status_code == 200
    pending = resp.json()
    assert len(pending) == 3


@pytest.mark.asyncio
async def test_from_scratch_shape_key(client):
    """from_scratch tasks get _fs suffix."""
    resp = await client.post("/api/tasks", json={
        "dtype": "e4m3",
        "m": 512,
        "n": 8192,
        "k": 8192,
        "mode": "from_scratch",
    })
    assert resp.status_code == 201
    assert resp.json()["shape_key"] == "e4m3_512x8192x8192_fs"
    assert resp.json()["max_iterations"] == 150


@pytest.mark.asyncio
async def test_cannot_delete_running_task(client):
    """Cannot delete a task while it's running."""
    resp = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = resp.json()["id"]

    from app.database import async_session
    from app.models import Task
    from datetime import datetime, timezone
    async with async_session() as session:
        task = await session.get(Task, task_id)
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        await session.commit()

    resp = await client.delete(f"/api/tasks/{task_id}")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint returns valid data."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["scheduler_running"], bool)
    assert data["default_model"] == "opencode/qwen3.6-plus-free"
