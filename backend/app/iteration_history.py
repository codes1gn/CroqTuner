from __future__ import annotations

from pathlib import Path

from .config import settings


def read_iteration_history(shape_key: str, task_id: int) -> list[dict]:
    results_path = settings.tuning_dir / "logs" / shape_key / "results.tsv"
    if not results_path.exists():
        return []

    try:
        lines = results_path.read_text(errors="replace").splitlines()
    except OSError:
        return []

    entries: list[dict] = []
    for line in lines:
        if not line.strip() or line.startswith("#") or line.lower().startswith("iter\t"):
            continue

        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 4:
            continue

        try:
            iteration = int(parts[0])
        except ValueError:
            continue

        try:
            tflops = float(parts[2])
        except ValueError:
            tflops = None

        decision = parts[4] if len(parts) > 4 and parts[4] != "BASELINE" else None
        bottleneck = parts[5] if len(parts) > 5 and parts[5] else None
        idea_summary = parts[6] if len(parts) > 6 and parts[6] else None
        kernel_path = parts[1] if len(parts) > 1 and parts[1] else None

        entries.append(
            {
                "id": -iteration,
                "task_id": task_id,
                "iteration": iteration,
                "kernel_path": kernel_path,
                "tflops": tflops,
                "decision": decision,
                "bottleneck": bottleneck,
                "idea_summary": idea_summary,
                "logged_at": None,
            }
        )

    entries.sort(key=lambda entry: entry["iteration"], reverse=True)
    return entries