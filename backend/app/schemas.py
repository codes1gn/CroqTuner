from pydantic import BaseModel, field_validator


class TaskCreate(BaseModel):
    dtype: str
    m: int
    n: int
    k: int
    mode: str

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if v not in ("f16", "e4m3"):
            raise ValueError("dtype must be 'f16' or 'e4m3'")
        return v

    @field_validator("m")
    @classmethod
    def validate_m(cls, v: int) -> int:
        if v < 128:
            raise ValueError("M must be >= 128")
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v: int) -> int:
        if v < 256:
            raise ValueError("N must be >= 256")
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v < 128:
            raise ValueError("K must be >= 128")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("from_current_best", "from_scratch"):
            raise ValueError("mode must be 'from_current_best' or 'from_scratch'")
        return v


class TaskUpdate(BaseModel):
    status: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is not None and v not in ("pending", "cancelled", "waiting"):
            raise ValueError("status must be 'pending', 'cancelled', or 'waiting'")
        return v


class TaskResponse(BaseModel):
    id: int
    shape_key: str
    dtype: str
    m: int
    n: int
    k: int
    mode: str
    max_iterations: int
    status: str
    current_iteration: int
    best_tflops: float | None
    baseline_tflops: float | None
    best_kernel: str | None
    error_message: str | None
    created_at: str | None
    updated_at: str | None
    started_at: str | None
    completed_at: str | None


class IterationLogResponse(BaseModel):
    id: int
    task_id: int
    iteration: int
    kernel_path: str | None
    tflops: float | None
    decision: str | None
    bottleneck: str | None
    idea_summary: str | None
    logged_at: str | None


class AgentLogResponse(BaseModel):
    id: int
    task_id: int
    level: str
    message: str
    timestamp: str | None


class HealthResponse(BaseModel):
    status: str
    scheduler_running: bool
    active_task_id: int | None
    gpu_info: str | None
