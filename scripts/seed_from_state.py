#!/usr/bin/env python3
"""
Import tasks from tuning/state.json (and optional results.tsv rows into iteration_logs).

Run from project root after copying croktile_paper/tuning -> CroqTuner/tuning:

  cd /path/to/CroqTuner/backend && source .venv/bin/activate
  python ../scripts/seed_from_state.py

Environment: same as backend (CROQTUNER_DB_PATH, etc.).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Run as if from backend/
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)


async def main() -> None:
    from sqlalchemy import delete, select

    from app.config import settings
    from app.database import async_session, engine, init_db
    from app.models import Base, IterationLog, Task

    state_path = settings.tuning_dir / "state.json"
    if not state_path.exists():
        print(f"Missing {state_path}", file=sys.stderr)
        sys.exit(1)

    import json

    raw = json.loads(state_path.read_text())
    shapes: dict = raw.get("shapes", {})

    await init_db()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    pat = re.compile(r"^(f16|e4m3)_(\d+)x(\d+)x(\d+)(_fs)?$")

    async with async_session() as session:
        await session.execute(delete(IterationLog))
        await session.execute(delete(Task))
        await session.commit()

    # Insert order: completed (historical), waiting (backlog), then active as pending (first to run)
    base_completed = datetime(2020, 1, 1, tzinfo=timezone.utc)
    base_waiting = datetime(2020, 6, 1, tzinfo=timezone.utc)
    active_ts = datetime(1999, 1, 1, tzinfo=timezone.utc)

    completed_rows: list[dict] = []
    waiting_rows: list[dict] = []
    active_row: dict | None = None

    for key, info in shapes.items():
        m = pat.match(key)
        if not m:
            print(f"skip bad key: {key}", file=sys.stderr)
            continue
        dtype, m1, n1, k1, fs = m.groups()
        m_val, n_val, k_val = int(m1), int(n1), int(k1)
        mode = "from_scratch" if fs else "from_current_best"
        max_iter = 150 if mode == "from_scratch" else 30

        st = info.get("status", "pending")
        cur = int(info.get("current_iter", 0) or 0)
        best_tf = info.get("best_tflops")
        base_tf = info.get("baseline_tflops")
        best_k = info.get("best_kernel")

        if st == "done":
            completed_rows.append(
                {
                    "shape_key": key,
                    "dtype": dtype,
                    "m": m_val,
                    "n": n_val,
                    "k": k_val,
                    "mode": info.get("mode", mode),
                    "max_iterations": max_iter,
                    "status": "completed",
                    "current_iteration": cur,
                    "best_tflops": float(best_tf) if best_tf is not None else None,
                    "baseline_tflops": float(base_tf) if base_tf is not None else None,
                    "best_kernel": best_k,
                    "created_at": base_completed + timedelta(seconds=len(completed_rows)),
                }
            )
        elif st == "active":
            active_row = {
                "shape_key": key,
                "dtype": dtype,
                "m": m_val,
                "n": n_val,
                "k": k_val,
                "mode": info.get("mode", mode),
                "max_iterations": max_iter,
                "status": "pending",
                "current_iteration": cur,
                "best_tflops": float(best_tf) if best_tf is not None else None,
                "baseline_tflops": float(base_tf) if base_tf is not None else None,
                "best_kernel": best_k,
                "created_at": active_ts,
            }
        else:
            waiting_rows.append(
                {
                    "shape_key": key,
                    "dtype": dtype,
                    "m": m_val,
                    "n": n_val,
                    "k": k_val,
                    "mode": info.get("mode", mode),
                    "max_iterations": max_iter,
                    "status": "waiting",
                    "current_iteration": cur,
                    "best_tflops": float(best_tf) if best_tf is not None else None,
                    "baseline_tflops": float(base_tf) if base_tf is not None else None,
                    "best_kernel": best_k,
                    "created_at": base_waiting + timedelta(seconds=len(waiting_rows)),
                }
            )

    def _add_task(session, row: dict) -> None:
        ts = row.pop("created_at")
        t = Task(**row)
        t.created_at = ts
        t.updated_at = ts
        session.add(t)

    async with async_session() as session:
        for row in completed_rows:
            r = dict(row)
            _add_task(session, r)
        await session.flush()

        for row in waiting_rows:
            r = dict(row)
            _add_task(session, r)
        await session.flush()

        if active_row:
            r = dict(active_row)
            _add_task(session, r)

        await session.commit()

    # Iteration logs from results.tsv (best-effort)
    logs_dir = settings.tuning_dir / "logs"
    async with async_session() as session:
        result = await session.execute(select(Task.id, Task.shape_key))
        task_by_key = {sk: tid for tid, sk in result.all()}

        imported = 0
        for shape_key, task_id in task_by_key.items():
            tsv = logs_dir / shape_key / "results.tsv"
            if not tsv.exists():
                continue
            lines = tsv.read_text(errors="replace").splitlines()
            data_lines = []
            for line in lines:
                if not line.strip() or line.startswith("#"):
                    continue
                if line.lower().startswith("iter\t"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                try:
                    it = int(parts[0].strip())
                except ValueError:
                    continue
                try:
                    tf = float(parts[2].strip())
                except ValueError:
                    tf = None
                decision = parts[4].strip() if len(parts) > 4 else None
                bottleneck = parts[5].strip() if len(parts) > 5 else None
                idea = parts[6].strip() if len(parts) > 6 else None
                kernel_path = parts[1].strip() if len(parts) > 1 else None
                if decision == "BASELINE":
                    decision = None
                session.add(
                    IterationLog(
                        task_id=task_id,
                        iteration=it,
                        kernel_path=kernel_path or None,
                        tflops=tf,
                        decision=decision,
                        bottleneck=bottleneck or None,
                        idea_summary=idea or None,
                    )
                )
                imported += 1
        await session.commit()

    print(
        f"Seeded {len(completed_rows)} completed, {len(waiting_rows)} waiting, "
        f"{1 if active_row else 0} active-as-pending, {imported} iteration log rows."
    )


if __name__ == "__main__":
    asyncio.run(main())
