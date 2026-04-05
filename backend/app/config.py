import subprocess
from pathlib import Path

from pydantic_settings import BaseSettings

AVAILABLE_VARIANTS = ("", "minimal", "low", "medium", "high", "xhigh", "max")

DEFAULT_OPENCODE_MODEL = "opencode/big-pickle"
DEFAULT_OPENCODE_VARIANT = "high"  # max is 3x cost — not worth it

_cached_models: list[str] | None = None


def fetch_opencode_models() -> list[str]:
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    try:
        result = subprocess.run(
            ["opencode", "models"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            models = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if models:
                _cached_models = models
                return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _cached_models = [DEFAULT_OPENCODE_MODEL]
    return _cached_models


def invalidate_model_cache() -> None:
    global _cached_models
    _cached_models = None


def is_supported_opencode_model(model: str) -> bool:
    return model in fetch_opencode_models()


def is_valid_variant(variant: str) -> bool:
    return variant in AVAILABLE_VARIANTS


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    tuning_dir: Path = _PROJECT_ROOT / "tuning"
    skills_dir: Path = _PROJECT_ROOT / ".claude" / "skills"
    project_dir: Path = _PROJECT_ROOT
    heartbeat_sec: int = 30
    db_path: str = "./data/croqtuner.db"
    opencode_bin: str = "opencode"
    opencode_model: str = DEFAULT_OPENCODE_MODEL
    opencode_variant: str = DEFAULT_OPENCODE_VARIANT
    opencode_db_path: Path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    cuda_visible_devices: str = "0"
    mock_mode: bool = False
    host: str = "0.0.0.0"
    port: int = 8642

    model_config = {"env_prefix": "CROQTUNER_"}


settings = Settings()
