"""
Recovery rate analysis for AgentBench-Fail.

Compares baseline (no verification) vs corrected (verification + retry) runs to compute:
  - Recovery rate: % of baseline failures recovered by the correction loop
  - Latency overhead: extra seconds added by verification
  - Token cost overhead: extra tokens consumed by verification
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd


_FAILURE_TYPES = [
    "tool_call_error",
    "context_memory_failure",
    "compounding_error",
    "premature_termination",
    "verification_blindness",
]


def load_paired_traces(directory: str | Path) -> tuple[list[dict], list[dict]]:
    """
    Load baseline and corrected trace files from the same directory.
    Matching is done by task_id + model.
    Returns (baseline_traces, corrected_traces).
    """
    all_traces = []
    for p in Path(directory).glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                all_traces.append(json.load(f))
        except (json.JSONDecodeError, KeyError):
            continue

    baseline = [t for t in all_traces if t.get("mode") == "baseline"]
    corrected = [t for t in all_traces if t.get("mode") == "corrected"]
    return baseline, corrected


def compute_recovery_rate(
    baseline: list[dict],
    corrected: list[dict],
) -> pd.DataFrame:
    """
    For each (task_id, model) pair that failed in baseline, check if it succeeded in corrected.

    Returns DataFrame with columns:
      model, failure_type, baseline_failures, recovered, recovery_rate
    """
    # Build lookup: (task_id, model) -> corrected success
    corrected_map = {(t["task_id"], t["model"]): t.get("success", False) for t in corrected}

    # Group baseline failures by (model, failure_type)
    stats: dict[tuple, dict] = defaultdict(lambda: {"baseline_failures": 0, "recovered": 0})

    for t in baseline:
        if t.get("success", False):
            continue  # only care about failures
        key = (t.get("model", "unknown"), t.get("failure_type") or "verification_blindness")
        stats[key]["baseline_failures"] += 1

        # Check if corrected run recovered
        corrected_success = corrected_map.get((t["task_id"], t.get("model", "")), False)
        if corrected_success:
            stats[key]["recovered"] += 1

    rows = []
    for (model, failure_type), c in sorted(stats.items()):
        rate = c["recovered"] / c["baseline_failures"] if c["baseline_failures"] > 0 else 0.0
        rows.append({
            "model": model,
            "failure_type": failure_type,
            "baseline_failures": c["baseline_failures"],
            "recovered": c["recovered"],
            "recovery_rate": round(rate, 4),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model", "failure_type", "baseline_failures", "recovered", "recovery_rate"]
    )


def compute_overhead(
    baseline: list[dict],
    corrected: list[dict],
) -> pd.DataFrame:
    """
    Compute per-model latency and token overhead introduced by the verification loop.

    Returns DataFrame with columns:
      model, baseline_avg_latency_s, corrected_avg_latency_s, latency_overhead_s,
      baseline_avg_tokens, corrected_avg_tokens, token_overhead, token_overhead_pct
    """
    def aggregate(traces: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = defaultdict(lambda: {"latency": [], "tokens": []})
        for t in traces:
            model = t.get("model", "unknown")
            out[model]["latency"].append(t.get("total_latency_s", 0.0))
            out[model]["tokens"].append(t.get("total_tokens", 0))
        return out

    base_agg = aggregate(baseline)
    corr_agg = aggregate(corrected)
    all_models = set(base_agg.keys()) | set(corr_agg.keys())

    rows = []
    for model in sorted(all_models):
        bl = base_agg.get(model, {"latency": [0], "tokens": [0]})
        cr = corr_agg.get(model, {"latency": [0], "tokens": [0]})
        bl_lat = sum(bl["latency"]) / max(len(bl["latency"]), 1)
        cr_lat = sum(cr["latency"]) / max(len(cr["latency"]), 1)
        bl_tok = sum(bl["tokens"]) / max(len(bl["tokens"]), 1)
        cr_tok = sum(cr["tokens"]) / max(len(cr["tokens"]), 1)
        tok_overhead_pct = (cr_tok - bl_tok) / bl_tok if bl_tok > 0 else 0.0
        rows.append({
            "model": model,
            "baseline_avg_latency_s": round(bl_lat, 2),
            "corrected_avg_latency_s": round(cr_lat, 2),
            "latency_overhead_s": round(cr_lat - bl_lat, 2),
            "baseline_avg_tokens": int(bl_tok),
            "corrected_avg_tokens": int(cr_tok),
            "token_overhead": int(cr_tok - bl_tok),
            "token_overhead_pct": round(tok_overhead_pct * 100, 1),
        })
    return pd.DataFrame(rows)


def compute_overall_recovery(baseline: list[dict], corrected: list[dict]) -> dict:
    """Compute aggregate recovery statistics across all models and failure types."""
    baseline_failures = [t for t in baseline if not t.get("success", True)]
    corrected_map = {(t["task_id"], t["model"]): t.get("success", False) for t in corrected}

    total_failures = len(baseline_failures)
    recovered = sum(
        1 for t in baseline_failures
        if corrected_map.get((t["task_id"], t.get("model", "")), False)
    )

    baseline_success = sum(1 for t in baseline if t.get("success", False))
    corrected_success = sum(1 for t in corrected if t.get("success", False))

    return {
        "baseline_total": len(baseline),
        "baseline_successes": baseline_success,
        "baseline_failures": total_failures,
        "baseline_success_rate": round(baseline_success / max(len(baseline), 1), 4),
        "corrected_total": len(corrected),
        "corrected_successes": corrected_success,
        "corrected_success_rate": round(corrected_success / max(len(corrected), 1), 4),
        "total_recovered": recovered,
        "overall_recovery_rate": round(recovered / max(total_failures, 1), 4),
    }


def plot_recovery_by_failure_type(
    df: pd.DataFrame,
    output_path: str | Path = "results/figures/recovery_by_failure_type.png",
    show: bool = False,
) -> Path:
    """Grouped bar chart: recovery rate per failure type, broken down by model."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Run: pip install matplotlib numpy")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = df["model"].unique()
    failure_types = _FAILURE_TYPES
    ft_labels = [ft.replace("_", "\n").title() for ft in failure_types]

    x = np.arange(len(failure_types))
    width = 0.25
    model_colors = {"claude": "#E63946", "gpt": "#457B9D", "openweight": "#F4A261"}

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, model in enumerate(sorted(models)):
        mdf = df[df["model"] == model]
        rates = []
        for ft in failure_types:
            row = mdf[mdf["failure_type"] == ft]
            rates.append(float(row["recovery_rate"].values[0]) if len(row) > 0 else 0.0)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=model.title(),
                      color=model_colors.get(model, "#888"), alpha=0.85, edgecolor="white")

    ax.set_xlabel("Failure Type", fontsize=11)
    ax.set_ylabel("Recovery Rate", fontsize=11)
    ax.set_title("Recovery Rate by Failure Type and Model\n(Corrected vs Baseline)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ft_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(title="Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def plot_overhead_comparison(
    df: pd.DataFrame,
    output_path: str | Path = "results/figures/overhead_comparison.png",
    show: bool = False,
) -> Path:
    """Side-by-side bar chart comparing latency and token overhead by model."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Run: pip install matplotlib numpy")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = df["model"].tolist()
    x = np.arange(len(models))
    colors = ["#E63946", "#457B9D", "#F4A261"]

    # Latency overhead
    ax1.bar(x, df["latency_overhead_s"], color=colors[:len(models)], alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.title() for m in models])
    ax1.set_ylabel("Latency Overhead (seconds)", fontsize=11)
    ax1.set_title("Avg. Latency Overhead\nfrom Verification Loop", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Token overhead
    ax2.bar(x, df["token_overhead_pct"], color=colors[:len(models)], alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.title() for m in models])
    ax2.set_ylabel("Token Overhead (%)", fontsize=11)
    ax2.set_title("Avg. Token Cost Overhead\nfrom Verification Loop", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Cost of Self-Correction: Latency and Token Overhead", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def generate_summary_table(
    recovery_df: pd.DataFrame,
    overhead_df: pd.DataFrame,
    overall: dict,
) -> pd.DataFrame:
    """
    Merge recovery and overhead stats into a single summary DataFrame
    suitable for IEEE paper Table formatting.
    """
    merged = recovery_df.groupby("model").agg(
        avg_recovery_rate=("recovery_rate", "mean"),
        total_baseline_failures=("baseline_failures", "sum"),
        total_recovered=("recovered", "sum"),
    ).reset_index()

    result = merged.merge(overhead_df[["model", "latency_overhead_s", "token_overhead_pct"]], on="model", how="left")
    result["avg_recovery_rate"] = result["avg_recovery_rate"].round(4)
    result = result.sort_values("avg_recovery_rate", ascending=False)
    return result


def generate_latex_recovery_table(summary: pd.DataFrame) -> str:
    """Generate LaTeX table of recovery rates for IEEE paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Recovery rates and verification overhead by model}",
        r"\label{tab:recovery}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Avg.\ Recovery & Failures & Recovered & Latency OH (s) \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['model'].title()} & {row['avg_recovery_rate']:.1%} & "
            f"{int(row['total_baseline_failures'])} & {int(row['total_recovered'])} & "
            f"+{row['latency_overhead_s']:.1f}s \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute recovery analysis from paired trace files.")
    parser.add_argument("--traces", default="results/raw_traces/")
    parser.add_argument("--output", default="results/figures/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    baseline, corrected = load_paired_traces(args.traces)
    print(f"Loaded {len(baseline)} baseline and {len(corrected)} corrected traces.")

    overall = compute_overall_recovery(baseline, corrected)
    print("\nOverall Recovery Stats:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    recovery_df = compute_recovery_rate(baseline, corrected)
    overhead_df = compute_overhead(baseline, corrected)

    print("\nRecovery Rate by Model × Failure Type:")
    print(recovery_df.to_string(index=False))

    print("\nOverhead by Model:")
    print(overhead_df.to_string(index=False))

    summary = generate_summary_table(recovery_df, overhead_df, overall)

    plot_recovery_by_failure_type(recovery_df, Path(args.output) / "recovery_by_failure_type.png")
    plot_overhead_comparison(overhead_df, Path(args.output) / "overhead_comparison.png")

    if args.latex:
        print("\n" + generate_latex_recovery_table(summary))

    print(f"\nFigures saved to {args.output}")
