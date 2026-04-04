import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
import sqlite3

from app.config import settings
from app.main import app
from app.database import async_session, init_db, engine
from app.models import Base, IterationLog, Task
from app.state_seed import seed_tasks_from_state_if_empty


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
    assert data["model"] == settings.opencode_model


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
async def test_get_task_uses_live_fsm_iteration(client):
        state_dir = settings.skills_dir / "fsm-engine" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "loop-state_from_scratch.json").write_text(
                """
{
    "fsm": {
        "current_state": "PROFILE",
        "iteration": 50,
        "shape_key": "f16_768x768x768_fs"
    }
}
                """.strip()
        )

        create = await client.post("/api/tasks", json={
                "dtype": "f16", "m": 768, "n": 768, "k": 768, "mode": "from_scratch",
        })
        task_id = create.json()["id"]

        async with async_session() as session:
                task = await session.get(Task, task_id)
                assert task is not None
                task.current_iteration = 49
                await session.commit()

        resp = await client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["current_iteration"] == 50


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
async def test_promote_invalid_transition_returns_400(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    async with async_session() as session:
        task = await session.get(Task, task_id)
        assert task is not None
        task.status = "completed"
        await session.commit()

    resp = await client.patch(f"/api/tasks/{task_id}", json={"status": "pending"})
    assert resp.status_code == 400
    assert "Cannot change task from" in resp.text


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
async def test_retry_task_creates_new_pending_task(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    original_id = create.json()["id"]

    async with async_session() as session:
        task = await session.get(Task, original_id)
        assert task is not None
        task.status = "failed"
        task.error_message = "boom"
        await session.commit()

    resp = await client.post(f"/api/tasks/{original_id}/retry")
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] != original_id
    assert data["status"] == "pending"
    assert data["current_iteration"] == 0
    assert data["model"] == settings.opencode_model


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "scheduler_running" in data
    assert data["default_model"] == settings.opencode_model
    assert "opencode/qwen3.6-plus-free" in data["available_models"]
    assert "pending" in data["task_counts"]


@pytest.mark.asyncio
async def test_update_default_model(client):
    resp = await client.patch("/api/settings/model", json={"default_model": "opencode/big-pickle"})
    assert resp.status_code == 200
    assert resp.json()["default_model"] == "opencode/big-pickle"

    resp = await client.post("/api/tasks", json={
        "dtype": "f16",
        "m": 768,
        "n": 768,
        "k": 768,
        "mode": "from_current_best",
    })
    assert resp.status_code == 201
    assert resp.json()["model"] == "opencode/big-pickle"


@pytest.mark.asyncio
async def test_create_task_with_explicit_model(client):
    resp = await client.post("/api/tasks", json={
        "dtype": "f16",
        "m": 768,
        "n": 768,
        "k": 768,
        "mode": "from_current_best",
        "model": "opencode/minimax-m2.5-free",
    })
    assert resp.status_code == 201
    assert resp.json()["model"] == "opencode/minimax-m2.5-free"


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
async def test_seed_from_state_when_db_empty():
    settings.tuning_dir.mkdir(parents=True, exist_ok=True)
    (settings.tuning_dir / "state.json").write_text(
        """
{
    "shapes": {
        "f16_768x768x768_fs": {
            "status": "active",
            "mode": "from_scratch",
            "current_iter": 7,
            "best_tflops": 123.4,
            "baseline_tflops": 100.0,
            "best_kernel": "tuning/srcs/f16_768x768x768_fs/seed.cu"
        },
        "f16_1024x1024x1024": {
            "status": "pending",
            "mode": "from_current_best",
            "current_iter": 0,
            "best_tflops": 0,
            "baseline_tflops": 0,
            "best_kernel": ""
        },
        "e4m3_1024x1024x1024": {
            "status": "done",
            "mode": "from_current_best",
            "current_iter": 30,
            "best_tflops": 456.7,
            "baseline_tflops": 450.0,
            "best_kernel": "kernels/gemm_sp_e4m3/e4m3_1024x1024x1024_best.cu"
        }
    }
}
""".strip()
    )

    async with async_session() as session:
        await session.execute(delete(IterationLog))
        await session.execute(delete(Task))
        await session.commit()

    async with async_session() as session:
        seeded = await seed_tasks_from_state_if_empty(session)
        assert seeded is True

        tasks = (await session.execute(select(Task.status))).all()
        statuses = [row[0] for row in tasks]
        assert "waiting" in statuses
        assert "completed" in statuses
        assert statuses.count("pending") == 1

        seeded_tasks = (await session.execute(select(Task))).scalars().all()
        assert all(task.model == settings.opencode_model for task in seeded_tasks)


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
async def test_iteration_logs_reads_shape_results_file(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 768, "n": 768, "k": 768, "mode": "from_scratch",
    })
    task_id = create.json()["id"]
    results_dir = settings.tuning_dir / "logs" / "f16_768x768x768_fs"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.tsv").write_text(
        "\n".join(
            [
                "iter\tkernel\ttflops\teff\tdecision\tbottleneck\tidea",
                "1\titer001.co\t42.0\t2.8\tKEEP\tgrid_too_small\tFirst idea",
                "2\titer002.co\t40.5\t2.7\tDISCARD\toccupancy_limited\tSecond idea",
            ]
        )
    )

    resp = await client.get(f"/api/tasks/{task_id}/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert [entry["iteration"] for entry in data] == [2, 1]
    assert data[0]["kernel_path"] == "iter002.co"
    assert data[1]["decision"] == "KEEP"


@pytest.mark.asyncio
async def test_agent_logs_empty(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.get(f"/api/tasks/{task_id}/agent-logs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_session_history_empty_without_session_id(client):
    create = await client.post("/api/tasks", json={
        "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
    })
    task_id = create.json()["id"]

    resp = await client.get(f"/api/tasks/{task_id}/session-history")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []
    assert resp.json()["session_id"] is None


@pytest.mark.asyncio
async def test_session_history_reads_opencode_db(client, tmp_path):
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("create table session (id text primary key, directory text, title text, time_created integer, time_updated integer)")
        conn.execute("create table message (id text primary key, session_id text, time_created integer, time_updated integer, data text)")
        conn.execute("create table part (id text primary key, message_id text, session_id text, time_created integer, time_updated integer, data text)")
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?)",
            ("ses_test", "/tmp/project", "Demo session", 1000, 2000),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_user",
                "ses_test",
                1000,
                1000,
                '{"role":"user"}',
            ),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_assistant",
                "ses_test",
                2000,
                2000,
                '{"role":"assistant"}',
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_user",
                "msg_user",
                "ses_test",
                1000,
                1000,
                '{"type":"text","text":"Tune this kernel"}',
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_reasoning",
                "msg_assistant",
                "ses_test",
                2000,
                2000,
                '{"type":"reasoning","text":"I should inspect the FSM state first."}',
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_tool",
                "msg_assistant",
                "ses_test",
                3000,
                3000,
                '{"type":"tool","tool":"read","state":{"status":"completed","input":{"filePath":"/tmp/state.json"},"output":"state contents"}}',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    original_db_path = settings.opencode_db_path
    settings.opencode_db_path = db_path
    try:
        create = await client.post("/api/tasks", json={
            "dtype": "f16", "m": 512, "n": 512, "k": 512, "mode": "from_current_best",
        })
        task_id = create.json()["id"]

        async with async_session() as session:
            task = await session.get(Task, task_id)
            assert task is not None
            task.opencode_session_id = "ses_test"
            await session.commit()

        resp = await client.get(f"/api/tasks/{task_id}/session-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "ses_test"
        assert data["session_title"] == "Demo session"
        assert data["session_directory"] == "/tmp/project"
        assert [entry["kind"] for entry in data["entries"]] == ["text", "reasoning", "tool"]
        assert data["entries"][0]["role"] == "user"
        assert data["entries"][1]["text"] == "I should inspect the FSM state first."
        assert "state contents" in data["entries"][2]["text"]
        assert data["entries"][2]["tool"] == "read"
        assert data["entries"][2]["status"] == "completed"
    finally:
        settings.opencode_db_path = original_db_path
