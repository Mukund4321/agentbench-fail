"""
WAT Tool: run_single_task.py
Executes one benchmark task and saves the trace to results/raw_traces/.
Credentials are read from .env via python-dotenv.
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run a single AgentBench-Fail task.")
    parser.add_argument("--task-id", required=True, help="e.g. short_001")
    parser.add_argument("--model", default="claude", choices=["claude", "gpt", "openweight"])
    parser.add_argument("--mode", default="baseline", choices=["baseline", "corrected"])
    parser.add_argument("--output-dir", default="results/raw_traces")
    parser.add_argument("--sqlite", default="results/benchmark.db", help="SQLite DB path")
    args = parser.parse_args()

    from tasks.loader import load_task_by_id
    from agents.runner import AgentRunner
    from taxonomy.classifier import FailureClassifier

    task = load_task_by_id(args.task_id)
    runner = AgentRunner(model_name=args.model, mode=args.mode)
    result = runner.run_task(task)

    # Classify failure type if task failed
    if not result.success:
        classifier = FailureClassifier()
        label = classifier.classify(task, result.trace, result.success)
        result.failure_type = label.primary.value

    # Save JSON trace
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{args.task_id}_{args.model}_{args.mode}.json"
    payload = {
        "task_id": result.task_id,
        "model": result.model,
        "mode": result.mode,
        "horizon": task.get("horizon"),
        "success": result.success,
        "steps_taken": result.steps_taken,
        "failure_type": result.failure_type,
        "total_tokens": result.total_tokens,
        "total_latency_s": round(result.total_latency_s, 3),
        "final_answer": result.final_answer,
        "error_message": result.error_message,
        "trace": [vars(s) for s in result.trace],
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Also save to SQLite
    from tracer.trace_logger import TraceLogger
    logger = TraceLogger()
    logger.start(result.task_id, result.model, result.mode)
    for s in result.trace:
        logger.log_step(s.step_num, s.tool_name, s.tool_args, s.output, s.latency_s, s.tokens)
    logger.finish(result.success, result.steps_taken, result.total_tokens, result.total_latency_s, result.failure_type)
    logger.save_sqlite(args.sqlite)

    status = "SUCCESS" if result.success else f"FAIL [{result.failure_type}]"
    print(f"[{status}] {args.task_id} | {args.model} | {args.mode}")
    print(f"  Steps: {result.steps_taken} | Tokens: {result.total_tokens} | Latency: {result.total_latency_s:.2f}s")
    print(f"  Trace: {trace_path}")


if __name__ == "__main__":
    main()
