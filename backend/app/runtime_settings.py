from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from .config import AVAILABLE_OPENCODE_MODELS, settings
from .models import SystemSetting

DEFAULT_MODEL_KEY = "default_model"


def available_models() -> list[str]:
    return list(AVAILABLE_OPENCODE_MODELS)


async def get_default_model(session: AsyncSession) -> str:
    row = await session.get(SystemSetting, DEFAULT_MODEL_KEY)
    if row is None:
        row = SystemSetting(key=DEFAULT_MODEL_KEY, value=settings.opencode_model)
        session.add(row)
        await session.flush()
    return row.value


async def set_default_model(session: AsyncSession, model: str) -> str:
    row = await session.get(SystemSetting, DEFAULT_MODEL_KEY)
    now = datetime.now(timezone.utc)
    if row is None:
        row = SystemSetting(key=DEFAULT_MODEL_KEY, value=model, updated_at=now)
        session.add(row)
    else:
        row.value = model
        row.updated_at = now
    await session.flush()
    return row.value