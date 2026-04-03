import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["CROQTUNER_DB_PATH"] = ":memory:"
os.environ["CROQTUNER_MOCK_MODE"] = "1"
os.environ["CROQTUNER_TUNING_DIR"] = "/tmp/croqtuner_test_tuning"
os.environ["CROQTUNER_SKILLS_DIR"] = "/tmp/croqtuner_test_skills"
os.environ["CROQTUNER_PROJECT_DIR"] = "/tmp/croqtuner_test_project"

from app.models import Base


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture
def tmp_tuning_dir(tmp_path):
    tuning = tmp_path / "tuning"
    for sub in ["logs", "srcs", "perf", "checkpoints"]:
        (tuning / sub).mkdir(parents=True)
    return tuning
