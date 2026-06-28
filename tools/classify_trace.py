"""
WAT Tool: classify_trace.py
Post-hoc failure classification for all trace files in a directory.
Updates each trace JSON with 'failure_type' and 'failure_label' fields.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Classify failure types in all trace files.")
    parser.add_argument("--traces", default="results/raw_traces/", help="Directory of JSON trace files")
    parser.add_argument("--dry-run", action="store_true", help="Print classifications without writing")
    args = parser.parse_args()

    from taxonomy.classifier import FailureClassifier
    from tracer.trace_logger import StepTrace

    classifier = FailureClassifier()
    trace_dir = Path(args.traces)
    files = list(trace_dir.glob("*.json"))
    print(f"Classifying {len(files)} trace files in {trace_dir}...")

    for p in sorted(files):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [SKIP] {p.name}: {e}")
            continue

        if data.get("success", False):
            data["failure_type"] = "success"
            data["failure_label"] = {"primary": "success", "contributing": [], "confidence": 1.0}
        else:
            steps = [StepTrace(**s) for s in data.get("trace", [])]
            task_meta = {"expected_steps": data.get("steps_taken", 1), "goal": ""}
            label = classifier.classify(task_meta, steps, data.get("success", False))
            data["failure_type"] = label.primary.value
            data["failure_label"] = label.to_dict()

        print(f"  {p.name}: {data['failure_type']} (success={data['success']})")

        if not args.dry_run:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    print(f"\nDone. {'(dry-run, no files written)' if args.dry_run else 'Files updated.'}")


if __name__ == "__main__":
    main()
