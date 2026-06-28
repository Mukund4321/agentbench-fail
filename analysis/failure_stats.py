"""
Failure rate aggregation and visualization for AgentBench-Fail.

Produces:
  - Failure rate by model × horizon
  - Failure distribution across the 5 taxonomy categories
  - Horizon length vs failure rate chart
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

_MODELS = ["claude", "gpt", "openweight"]
_HORIZONS = ["short", "medium", "long"]


def load_traces(directory: str | Path, mode: Optional[str] = None) -> list[dict]:
    """
    Load all JSON trace files from a directory.
    Optionally filter by mode ('baseline' or 'corrected').
    """
    traces = []
    for p in Path(directory).glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if mode and data.get("mode") != mode:
                continue
            traces.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return traces


def compute_failure_rates(
    traces: list[dict],
) -> pd.DataFrame:
    """
    Compute failure rate by (model, horizon).
    Returns a DataFrame with columns: model, horizon, total, failures, failure_rate.
    """
    counts: dict[tuple, dict] = defaultdict(lambda: {"total": 0, "failures": 0})
    for t in traces:
        model = t.get("model", "unknown")
        # horizon needs to come from task metadata (may not be in trace directly)
        horizon = t.get("horizon", _infer_horizon(t.get("task_id", "")))
        key = (model, horizon)
        counts[key]["total"] += 1
        if not t.get("success", False):
            counts[key]["failures"] += 1

    rows = []
    for (model, horizon), c in sorted(counts.items()):
        rate = c["failures"] / c["total"] if c["total"] > 0 else 0.0
        rows.append({
            "model": model,
            "horizon": horizon,
            "total": c["total"],
            "failures": c["failures"],
            "failure_rate": round(rate, 4),
        })
    return pd.DataFrame(rows)


def compute_failure_distribution(
    traces: list[dict],
    baseline_only: bool = True,
) -> pd.DataFrame:
    """
    Count failures by taxonomy category.
    Returns a DataFrame with columns: failure_type, count, pct.
    """
    if baseline_only:
        traces = [t for t in traces if t.get("mode") == "baseline"]

    failures = [t for t in traces if not t.get("success", True)]
    type_counts: dict[str, int] = defaultdict(int)
    for t in failures:
        ft = t.get("failure_type") or "verification_blindness"
        type_counts[ft] += 1

    total = sum(type_counts.values()) or 1
    rows = [
        {
            "failure_type": ft,
            "count": type_counts.get(ft, 0),
            "pct": round(type_counts.get(ft, 0) / total * 100, 1),
        }
        for ft in _FAILURE_TYPES
    ]
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def plot_failure_distribution(
    stats: pd.DataFrame,
    output_path: str | Path = "results/figures/failure_distribution.png",
    show: bool = False,
) -> Path:
    """
    Bar chart of failure distribution across taxonomy categories.
    Saves to output_path and optionally displays inline.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("Run: pip install matplotlib")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F", "#9B2226"]
    labels = [ft.replace("_", " ").title() for ft in stats["failure_type"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, stats["count"], color=colors[:len(stats)], edgecolor="white", linewidth=0.8)

    for bar, pct in zip(bars, stats["pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("AgentBench-Fail: Failure Distribution by Taxonomy Category\n(Baseline Runs)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Failure Category", fontsize=11)
    ax.set_ylabel("Number of Failures", fontsize=11)
    ax.set_ylim(0, max(stats["count"]) * 1.2 + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def plot_horizon_vs_failure(
    df: pd.DataFrame,
    output_path: str | Path = "results/figures/horizon_vs_failure.png",
    show: bool = False,
) -> Path:
    """
    Line chart: failure rate by horizon for each model.
    df should be the output of compute_failure_rates().
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        raise ImportError("Run: pip install matplotlib")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    horizon_order = ["short", "medium", "long"]
    model_colors = {"claude": "#E63946", "gpt": "#457B9D", "openweight": "#F4A261"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for model in df["model"].unique():
        mdf = df[df["model"] == model].copy()
        mdf["horizon_idx"] = mdf["horizon"].apply(lambda h: horizon_order.index(h) if h in horizon_order else 0)
        mdf = mdf.sort_values("horizon_idx")
        color = model_colors.get(model, "#888888")
        ax.plot(
            mdf["horizon"],
            mdf["failure_rate"],
            marker="o",
            linewidth=2.2,
            markersize=8,
            label=model.title(),
            color=color,
        )
        for _, row in mdf.iterrows():
            ax.annotate(
                f"{row['failure_rate']:.0%}",
                xy=(row["horizon"], row["failure_rate"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
            )

    ax.set_title("Failure Rate by Task Horizon Length and Model\n(Baseline Runs)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Task Horizon", fontsize=11)
    ax.set_ylabel("Failure Rate", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xticks(horizon_order)
    ax.set_xticklabels(["Short (2-3 steps)", "Medium (5-7 steps)", "Long (10+ steps)"])
    ax.legend(title="Model", loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def plot_failure_heatmap(
    df: pd.DataFrame,
    output_path: str | Path = "results/figures/failure_heatmap.png",
    show: bool = False,
) -> Path:
    """Heatmap of failure rate by (model, horizon)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Run: pip install matplotlib numpy")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = df.pivot_table(values="failure_rate", index="model", columns="horizon", aggfunc="mean")
    pivot = pivot.reindex(columns=["short", "medium", "long"])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Failure Rate")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([c.title() for c in pivot.columns])
    ax.set_yticklabels([m.title() for m in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=12,
                    color="white" if val > 0.5 else "black", fontweight="bold")

    ax.set_title("Failure Rate Heatmap (Model × Horizon)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table from the failure rates DataFrame for IEEE paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Failure rates by model and task horizon (baseline runs)}",
        r"\label{tab:failure-rates}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Horizon & Total & Failures & Failure Rate \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['model'].title()} & {row['horizon'].title()} & "
            f"{row['total']} & {row['failures']} & {row['failure_rate']:.1%} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _infer_horizon(task_id: str) -> str:
    """Infer horizon from task_id prefix."""
    if task_id.startswith("short"):
        return "short"
    elif task_id.startswith("medium"):
        return "medium"
    elif task_id.startswith("long"):
        return "long"
    return "unknown"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute failure statistics from trace files.")
    parser.add_argument("--traces", default="results/raw_traces/", help="Directory of JSON trace files")
    parser.add_argument("--output", default="results/figures/", help="Output directory for figures")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table to stdout")
    args = parser.parse_args()

    traces = load_traces(args.traces, mode="baseline")
    print(f"Loaded {len(traces)} baseline traces.")

    rates_df = compute_failure_rates(traces)
    dist_df = compute_failure_distribution(traces)

    print("\nFailure Rates by Model × Horizon:")
    print(rates_df.to_string(index=False))

    print("\nFailure Distribution:")
    print(dist_df.to_string(index=False))

    plot_failure_distribution(dist_df, Path(args.output) / "failure_distribution.png")
    plot_horizon_vs_failure(rates_df, Path(args.output) / "horizon_vs_failure.png")
    plot_failure_heatmap(rates_df, Path(args.output) / "failure_heatmap.png")

    if args.latex:
        print("\n" + generate_latex_table(rates_df))

    print(f"\nFigures saved to {args.output}")
