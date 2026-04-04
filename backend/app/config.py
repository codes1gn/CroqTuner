from pathlib import Path

from pydantic_settings import BaseSettings

AVAILABLE_OPENCODE_MODELS = (
    "opencode/qwen3.6-plus-free",
    "opencode/minimax-m2.5-free",
    "opencode/big-pickle",
)


def is_supported_opencode_model(model: str) -> bool:
    return model in AVAILABLE_OPENCODE_MODELS

# backend/app/config.py -> CroqTuner repository root (self-contained project for opencode)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Defaults keep all CroqTuner assets inside this repo:
    - `.claude/skills/` — FSM + tuning skills (same layout as Claude Code)
    - `kernels/` — manifest + gemm_sp registry paths referenced by skills
    - `tuning/` — state, logs, checkpoints (often gitignored; rsync from paper repo)
    `project_dir` is the directory passed to `opencode run` (working tree for the agent).
    """

    tuning_dir: Path = _PROJECT_ROOT / "tuning"
    skills_dir: Path = _PROJECT_ROOT / ".claude" / "skills"
    project_dir: Path = _PROJECT_ROOT
    heartbeat_sec: int = 30
    db_path: str = "./data/croqtuner.db"
    opencode_bin: str = "opencode"
    opencode_model: str = "opencode/qwen3.6-plus-free"
    opencode_db_path: Path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    cuda_visible_devices: str = "0"
    mock_mode: bool = False
    host: str = "0.0.0.0"
    port: int = 8642

    model_config = {"env_prefix": "CROQTUNER_"}


settings = Settings()
