"""
Structured trace logger for AgentBench-Fail.

Captures per-step metadata (tool, args, output, latency, tokens) and
writes to JSON and/or SQLite for queryable offline analysis.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class StepTrace:
    step_num: int
    tool_name: str
    tool_args: dict
    output: str
    latency_s: float
    tokens: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskTrace:
    task_id: str
    model: str
    mode: str
    steps: list[StepTrace] = field(default_factory=list)
    success: Optional[bool] = None
    total_tokens: int = 0
    total_latency_s: float = 0.0
    failure_type: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "mode": self.mode,
            "success": self.success,
            "total_tokens": self.total_tokens,
            "total_latency_s": self.total_latency_s,
            "failure_type": self.failure_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [s.to_dict() for s in self.steps],
        }


class TraceLogger:
    """
    Accumulates step-level trace data during a single task run and
    provides save/export methods for post-run analysis.
    """

    def __init__(self):
        self._current: Optional[TaskTrace] = None
        self._steps: list[StepTrace] = []

    def start(self, task_id: str, model: str, mode: str) -> None:
        self._steps = []
        self._current = TaskTrace(task_id=task_id, model=model, mode=mode)

    def log_step(
        self,
        step_num: int,
        tool_name: str,
        tool_args: dict,
        output: str,
        latency_s: float,
        tokens: int = 0,
    ) -> None:
        step = StepTrace(
            step_num=step_num,
            tool_name=tool_name,
            tool_args=tool_args,
            output=output[:2000],  # cap to avoid huge traces
            latency_s=latency_s,
            tokens=tokens,
        )
        self._steps.append(step)

    def finish(
        self,
        success: bool,
        steps_taken: int,
        total_tokens: int,
        total_latency_s: float,
        failure_type: Optional[str] = None,
    ) -> None:
        if self._current:
            self._current.steps = self._steps
            self._current.success = success
            self._current.total_tokens = total_tokens
            self._current.total_latency_s = total_latency_s
            self._current.failure_type = failure_type
            self._current.finished_at = time.time()

    def build_trace(self) -> list[StepTrace]:
        return list(self._steps)

    def save_json(self, path: str | Path) -> None:
        if not self._current:
            raise RuntimeError("No active trace. Call start() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._current.to_dict(), f, indent=2)

    def save_sqlite(self, db_path: str | Path) -> None:
        if not self._current:
            raise RuntimeError("No active trace. Call start() first.")
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_results (
                    task_id TEXT, model TEXT, mode TEXT, success INTEGER,
                    total_tokens INTEGER, total_latency_s REAL, failure_type TEXT,
                    started_at REAL, finished_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS step_traces (
                    task_id TEXT, model TEXT, mode TEXT,
                    step_num INTEGER, tool_name TEXT, tool_args TEXT,
                    output TEXT, latency_s REAL, tokens INTEGER, timestamp REAL
                )
            """)
            t = self._current
            conn.execute(
                "INSERT INTO task_results VALUES (?,?,?,?,?,?,?,?,?)",
                (t.task_id, t.model, t.mode, int(t.success or False),
                 t.total_tokens, t.total_latency_s, t.failure_type,
                 t.started_at, t.finished_at),
            )
            for s in t.steps:
                conn.execute(
                    "INSERT INTO step_traces VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (t.task_id, t.model, t.mode, s.step_num, s.tool_name,
                     json.dumps(s.tool_args), s.output, s.latency_s, s.tokens, s.timestamp),
                )
            conn.commit()
        finally:
            conn.close()


def load_traces_from_dir(directory: str | Path) -> list[dict]:
    """Load all JSON trace files from a directory into a list of dicts."""
    traces = []
    for p in Path(directory).glob("*.json"):
        with open(p, encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces


def load_traces_from_sqlite(db_path: str | Path) -> list[dict]:
    """Load task-level result rows from SQLite (no step details)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM task_results").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
