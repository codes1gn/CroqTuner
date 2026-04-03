import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.database import init_db, engine
from app.models import Base


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create fresh tables for each test."""
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
async def test_list_tasks_empty(client):
    resp = await client.get("/api/tasks")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_create_task(client):
    resp = await client.post("/api/tasks", json={
        "dtype": "f16",
        "m": 768,
        "n": 768,
        "k": 768,
        "mode": "from_current_best",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["shape_key"] == "f16_768x768x768"
    assert data["status"] == "pending"
    assert data["max_iterations"] == 30


@pytest.mark.asyncio
async def test_create_task_from_scratch(client):
    resp = await client.post("/api/tasks", json={
        "dtype": "e4m3",
        "m": 512,
        "n": 8192,
        "k": 8192,
        "mode": "from_scratch",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["shape_key"] == "e4m3_512x8192x8192_fs"
    assert data["max_iterations"] == 150


@pytest.mark.asyncio
async def test_create_task_validation(client):
    resp = await client.post("/api/tasks", json={
        "dtype": "f16",
        "m": 64,
        "n": 768,
        "k": 768,
        "mode": "from_current_best",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_task(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 1024, "n": 1024, "k": 1024, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["shape_key"] == "f16_1024x1024x1024"


@pytest.mark.asyncio
async def test_get_task_not_found(client):
    resp = await client.get("/api/tasks/9999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promote_waiting_task(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]
    from app.database import async_session
    from app.models import Task
    async with async_session() as session:
        t = await session.get(Task, task_id)
        assert t is not None
        t.status = "waiting"
        await session.commit()

    resp = await client.patch(f"/api/tasks/{task_id}", json={"status": "pending"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_cancel_task(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.patch(f"/api/tasks/{task_id}", json={"status": "cancelled"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_delete_task(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.delete(f"/api/tasks/{task_id}")
    assert resp.status_code == 204

    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "scheduler_running" in data


@pytest.mark.asyncio
async def test_list_tasks_filter(client):
    await client.post("/api/tasks", json={
        "dtype": "f16", "m": 768, "n": 768, "k": 768, "mode": "from_current_best",
    })
    await client.post("/api/tasks", json={
        "dtype": "f16", "m": 1024, "n": 1024, "k": 1024, "mode": "from_current_best",
    })

    resp = await client.get("/api/tasks?status=pending")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = await client.get("/api/tasks?status=completed")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


@pytest.mark.asyncio
async def test_iteration_logs_empty(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.get(f"/api/tasks/{task_id}/logs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_agent_logs_empty(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.get(f"/api/tasks/{task_id}/agent-logs")
    assert resp.status_code == 200
    assert resp.json() == []
