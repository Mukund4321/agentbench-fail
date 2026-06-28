"""
WAT Tool: generate_figures.py
Generates all analysis figures for the IEEE paper writeup.
Reads trace files from results/raw_traces/, writes PNGs to results/figures/.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate all AgentBench-Fail analysis figures.")
    parser.add_argument("--traces", default="results/raw_traces/")
    parser.add_argument("--output", default="results/figures/")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX tables")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    from analysis.failure_stats import (
        load_traces, compute_failure_rates, compute_failure_distribution,
        plot_failure_distribution, plot_horizon_vs_failure, plot_failure_heatmap,
        generate_latex_table,
    )
    from analysis.recovery_analysis import (
        load_paired_traces, compute_recovery_rate, compute_overhead,
        compute_overall_recovery, plot_recovery_by_failure_type,
        plot_overhead_comparison, generate_summary_table, generate_latex_recovery_table,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading traces...")
    all_traces = load_traces(args.traces)
    baseline, corrected = load_paired_traces(args.traces)

    print(f"  Total: {len(all_traces)} | Baseline: {len(baseline)} | Corrected: {len(corrected)}")

    # Failure stats
    print("\nComputing failure statistics...")
    rates_df = compute_failure_rates(load_traces(args.traces, mode="baseline"))
    dist_df = compute_failure_distribution(all_traces)

    print("  Plotting failure distribution...")
    plot_failure_distribution(dist_df, out_dir / "failure_distribution.png", show=args.show)

    print("  Plotting horizon vs failure...")
    plot_horizon_vs_failure(rates_df, out_dir / "horizon_vs_failure.png", show=args.show)

    print("  Plotting failure heatmap...")
    plot_failure_heatmap(rates_df, out_dir / "failure_heatmap.png", show=args.show)

    # Recovery analysis
    print("\nComputing recovery analysis...")
    overall = compute_overall_recovery(baseline, corrected)
    recovery_df = compute_recovery_rate(baseline, corrected)
    overhead_df = compute_overhead(baseline, corrected)

    print(f"  Overall recovery rate: {overall.get('overall_recovery_rate', 0):.1%}")
    print(f"  Baseline success rate: {overall.get('baseline_success_rate', 0):.1%}")
    print(f"  Corrected success rate: {overall.get('corrected_success_rate', 0):.1%}")

    print("  Plotting recovery by failure type...")
    plot_recovery_by_failure_type(recovery_df, out_dir / "recovery_by_failure_type.png", show=args.show)

    print("  Plotting overhead comparison...")
    plot_overhead_comparison(overhead_df, out_dir / "overhead_comparison.png", show=args.show)

    summary = generate_summary_table(recovery_df, overhead_df, overall)

    if args.latex:
        print("\n=== LaTeX: Failure Rates Table ===")
        print(generate_latex_table(rates_df))
        print("\n=== LaTeX: Recovery Table ===")
        print(generate_latex_recovery_table(summary))

    print(f"\nAll figures saved to {out_dir}/")
    print("Figures generated:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
