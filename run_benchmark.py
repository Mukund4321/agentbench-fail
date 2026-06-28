"""
AgentBench-Fail — Main CLI Entry Point

Usage:
  python run_benchmark.py --mode baseline --model claude --tasks all
  python run_benchmark.py --mode corrected --model gpt --tasks short
  python run_benchmark.py --mode baseline --model claude --tasks medium_001,medium_002
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        description="AgentBench-Fail: Benchmark LLM agent failure modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode", required=True, choices=["baseline", "corrected"],
        help="baseline: no verification; corrected: verification + retry loop",
    )
    p.add_argument(
        "--model", required=True, choices=["claude", "gpt", "openweight"],
        help="LLM backend to use",
    )
    p.add_argument(
        "--tasks", default="all",
        help=(
            "Which tasks to run. Options: 'all', 'short', 'medium', 'long', "
            "or comma-separated task IDs like 'short_001,medium_003'"
        ),
    )
    p.add_argument("--output", default="results/raw_traces/", help="Output directory for traces")
    p.add_argument("--sqlite", default="results/benchmark.db", help="SQLite DB for persistence")
    p.add_argument("--classify", action="store_true", help="Run failure classifier after each task")
    p.add_argument("--verbose", action="store_true", help="Print step-by-step trace to stdout")
    p.add_argument("--dry-run", action="store_true", help="Load tasks but do not call LLM APIs")
    p.add_argument("--limit", type=int, default=None, help="Max number of tasks to run (for testing)")
    return p.parse_args()


def select_tasks(tasks_arg: str) -> list[dict]:
    from tasks.loader import load_tasks, load_task_by_id

    if tasks_arg == "all":
        return load_tasks()
    if tasks_arg in ("short", "medium", "long"):
        return load_tasks(horizon=tasks_arg)
    # Comma-separated IDs
    ids = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    return [load_task_by_id(tid) for tid in ids]


def run_benchmark(args) -> dict:
    from agents.runner import AgentRunner
    from taxonomy.classifier import FailureClassifier

    tasks = select_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = FailureClassifier() if args.classify else None

    summary = {
        "model": args.model,
        "mode": args.mode,
        "total": len(tasks),
        "success": 0,
        "failure": 0,
        "by_horizon": {"short": {"total": 0, "success": 0}, "medium": {"total": 0, "success": 0}, "long": {"total": 0, "success": 0}},
        "by_failure_type": {},
        "total_tokens": 0,
        "total_latency_s": 0.0,
        "results": [],
    }

    print(f"\n{'='*60}")
    print(f"  AgentBench-Fail")
    print(f"  Model: {args.model} | Mode: {args.mode} | Tasks: {len(tasks)}")
    print(f"{'='*60}\n")

    runner = AgentRunner(model_name=args.model, mode=args.mode) if not args.dry_run else None

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        horizon = task.get("horizon", "unknown")
        print(f"[{i:3d}/{len(tasks)}] {task_id} ({horizon}) ...", end=" ", flush=True)

        trace_path = out_dir / f"{task_id}_{args.model}_{args.mode}.json"
        if trace_path.exists():
            print("SKIP (already exists)")
            with open(trace_path, encoding="utf-8") as f:
                existing = json.load(f)
            _update_summary(summary, existing, horizon)
            continue

        if args.dry_run:
            print("DRY-RUN")
            continue

        t0 = time.time()
        try:
            result = runner.run_task(task)
        except Exception as exc:
            print(f"ERROR: {exc}")
            summary["failure"] += 1
            continue

        elapsed = time.time() - t0

        # Classify failure
        failure_type = result.failure_type
        if not result.success and classifier:
            label = classifier.classify(task, result.trace, result.success)
            failure_type = label.primary.value

        status = "OK" if result.success else f"FAIL [{failure_type}]"
        print(f"{status} | steps={result.steps_taken} tokens={result.total_tokens} lat={elapsed:.1f}s")

        # Save trace
        payload = {
            "task_id": task_id,
            "model": args.model,
            "mode": args.mode,
            "horizon": horizon,
            "success": result.success,
            "steps_taken": result.steps_taken,
            "failure_type": failure_type,
            "total_tokens": result.total_tokens,
            "total_latency_s": round(elapsed, 3),
            "final_answer": result.final_answer,
            "error_message": result.error_message,
            "trace": [vars(s) for s in result.trace],
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        _update_summary(summary, payload, horizon)

    _print_summary(summary)
    _save_summary(summary, out_dir.parent / f"summary_{args.model}_{args.mode}.json")
    return summary


def _update_summary(summary: dict, result: dict, horizon: str) -> None:
    summary["total_tokens"] += result.get("total_tokens", 0)
    summary["total_latency_s"] += result.get("total_latency_s", 0.0)
    if result.get("success"):
        summary["success"] += 1
    else:
        summary["failure"] += 1
        ft = result.get("failure_type") or "unknown"
        summary["by_failure_type"][ft] = summary["by_failure_type"].get(ft, 0) + 1
    if horizon in summary["by_horizon"]:
        summary["by_horizon"][horizon]["total"] += 1
        if result.get("success"):
            summary["by_horizon"][horizon]["success"] += 1
    summary["results"].append({
        "task_id": result.get("task_id"),
        "success": result.get("success"),
        "failure_type": result.get("failure_type"),
        "steps": result.get("steps_taken"),
        "tokens": result.get("total_tokens"),
    })


def _print_summary(summary: dict) -> None:
    total = summary["total"]
    success = summary["success"]
    failure = summary["failure"]
    rate = success / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  RESULTS: {args_model} | {args_mode}")
    print(f"  Success: {success}/{total} ({rate:.1%})")
    print(f"  Failure: {failure}/{total}")
    print(f"  Total tokens: {summary['total_tokens']:,}")
    print(f"  Total latency: {summary['total_latency_s']:.1f}s")
    print(f"\n  Failure distribution:")
    for ft, count in sorted(summary["by_failure_type"].items(), key=lambda x: -x[1]):
        print(f"    {ft}: {count}")
    print(f"\n  By horizon:")
    for horizon, counts in summary["by_horizon"].items():
        t = counts["total"]
        s = counts["success"]
        if t > 0:
            print(f"    {horizon}: {s}/{t} ({s/t:.1%} success)")
    print(f"{'='*60}\n")


def _save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = {k: v for k, v in summary.items() if k != "results"}
    trimmed["results_count"] = len(summary["results"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2)
    print(f"Summary saved to {path}")


args_model = "unknown"
args_mode = "unknown"


if __name__ == "__main__":
    args = parse_args()
    args_model = args.model
    args_mode = args.mode

    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results saved.")
        sys.exit(0)
