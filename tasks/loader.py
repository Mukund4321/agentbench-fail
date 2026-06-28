import json
from pathlib import Path
from typing import Optional

TASK_DIR = Path(__file__).parent
_HORIZON_FILES = {
    "short": TASK_DIR / "short_tasks.json",
    "medium": TASK_DIR / "medium_tasks.json",
    "long": TASK_DIR / "long_tasks.json",
}


def load_tasks(horizon: Optional[str] = None) -> list[dict]:
    """Load all tasks, optionally filtered by horizon (short/medium/long)."""
    if horizon:
        horizons = [horizon]
    else:
        horizons = list(_HORIZON_FILES.keys())

    tasks = []
    for h in horizons:
        path = _HORIZON_FILES[h]
        with open(path, "r", encoding="utf-8") as f:
            tasks.extend(json.load(f))
    return tasks


def load_task_by_id(task_id: str) -> dict:
    """Load a single task by its task_id."""
    for task in get_all_tasks():
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Task '{task_id}' not found.")


def get_all_tasks() -> list[dict]:
    return load_tasks()
