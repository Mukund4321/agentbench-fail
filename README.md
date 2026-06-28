# AgentBench-Fail

**A benchmark for characterizing and correcting failure modes in long-horizon LLM agents**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

---

## Overview

AgentBench-Fail is an empirical benchmark designed to answer a question most agent demos avoid: **where exactly do LLM agents break down on multi-step tasks, and can a lightweight verification loop recover from those failures?**

Most agent projects showcase success cases. This project does the opposite — it deliberately stresses agents across increasing task complexity, systematically classifies *how* and *why* they fail using a structured taxonomy, and then measures how much of that failure is recoverable through self-verification.

This work is motivated by — and methodologically aligned with — current research on agent reliability from groups like METR, the GAIA benchmark authors, and AgentBench, extending their direction with a focus on **failure attribution** and **correction recovery rate**, rather than pass/fail scoring alone.

---

## Why This Project Exists

As LLM agents move from demos to production systems, the central open problem isn't "can an agent use tools" — it's "can we trust an agent on a 10-step task without a human checking every step." Reliability, not capability, is the current bottleneck in agentic AI deployment. This benchmark exists to quantify that gap and test one practical mitigation.

---

## Key Contributions

1. **A failure taxonomy** for multi-step agent execution, covering five categories of breakdown (see below)
2. **A task suite** spanning short (2-3 step), medium (5-7 step), and long (10+ step) horizons
3. **Cross-model comparison** of failure rates and failure types across multiple LLM backends
4. **A self-correction/verification layer** that checks intermediate outputs against task goals and triggers retries
5. **Recovery rate analysis** — quantifying how much failure is recoverable, at what latency/cost tradeoff

---

## Failure Taxonomy

| Category | Description | Example |
|---|---|---|
| **Tool-call error** | Wrong tool selected, malformed arguments, or hallucinated tool that doesn't exist | Agent calls `search_flights()` with a malformed date string |
| **Context/memory failure** | Agent loses track of earlier steps or contradicts its own prior output | Agent re-asks for information already provided in step 2 |
| **Compounding error** | A small early mistake cascades into total task failure | Wrong unit conversion in step 1 propagates through every later calculation |
| **Premature termination** | Agent declares success when the task isn't actually complete | Agent stops after partial data retrieval, reports "done" |
| **Verification blindness** | Agent never checks its own output against the original goal | Agent generates a report that doesn't answer the original question |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Task Suite     │────▶│   Agent Runner    │────▶│   Trace Logger      │
│  (40-50 tasks,   │     │  (LangGraph-based  │     │  (step-by-step JSON │
│   3 horizons)    │     │   orchestration)   │     │   trace + metadata) │
└─────────────────┘     └──────────────────┘     └─────────┬──────────┘
                                                              │
                          ┌───────────────────────────────────┘
                          ▼
                ┌──────────────────────┐
                │  Failure Classifier   │
                │  (taxonomy labeling)  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐      ┌─────────────────────┐
                │  Self-Correction       │─────▶│  Recovery Analysis   │
                │  Verification Loop     │      │  (recovery %, cost   │
                │  (retry/backtrack)      │      │   & latency deltas)  │
                └──────────────────────┘      └─────────────────────┘
```

---

## Tech Stack

- **Orchestration**: LangGraph (explicit state graphs for instrumentable agent execution)
- **Models evaluated**: Claude (Anthropic API), GPT (OpenAI API), open-weight model via Together/Groq
- **Tracing & logging**: Custom structured logger → JSON/SQLite for queryable analysis
- **Analysis**: Python, pandas, matplotlib/plotly
- **Language**: Python 3.10+

---

## Project Structure

```
agentbench-fail/
├── tasks/                      # Task definitions by horizon length
│   ├── short/                  # 2-3 step tasks
│   ├── medium/                 # 5-7 step tasks
│   └── long/                   # 10+ step tasks
├── agents/
│   ├── runner.py                # Core agent execution loop
│   ├── verifier.py               # Self-correction / verification layer
│   └── models.py                 # Model client wrappers (Claude/GPT/OSS)
├── taxonomy/
│   └── classifier.py             # Failure mode classification logic
├── logging/
│   └── trace_logger.py           # Structured trace + metadata capture
├── analysis/
│   ├── failure_stats.py          # Failure rate aggregation
│   ├── recovery_analysis.py      # Correction recovery rate calculations
│   └── notebooks/                # Exploratory analysis notebooks
├── results/
│   ├── raw_traces/                # Per-run JSON traces
│   └── figures/                   # Generated charts for writeup
├── docs/
│   └── writeup.md                 # Blog post / arXiv draft
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Task design**: 40-50 tasks across 3 horizon lengths, each requiring real tool use (mock APIs for determinism — search, calculation, data lookup, file operations)
2. **Baseline run**: Each task run on each model with no correction mechanism, full trace logged
3. **Classification**: Every failure manually + programmatically labeled against the taxonomy
4. **Correction run**: Same tasks re-run with the verification loop active
5. **Comparison**: Recovery rate, added latency, added token cost computed per task category and per model

---

## Results

> *To be populated as the benchmark runs complete.*

| Model | Baseline Success Rate | Post-Correction Success Rate | Recovery Rate | Avg. Latency Overhead | Avg. Token Cost Overhead |
|---|---|---|---|---|---|
| Claude (model TBD) | — | — | — | — | — |
| GPT (model TBD) | — | — | — | — | — |
| Open-weight (model TBD) | — | — | — | — | — |

**Failure distribution by category (baseline):**

> *Chart: failure_distribution.png*

**Failure rate by task horizon length:**

> *Chart: horizon_vs_failure.png*

---

## Key Findings

> *To be written once data collection is complete. Format: 3-5 bullet points stating the most interesting empirical results, e.g. "Compounding errors accounted for X% of long-horizon failures but were the most recoverable category (Y% recovery rate), while verification-blindness failures had the lowest recovery rate (Z%), suggesting agents cannot reliably self-detect this failure type."*

---

## Running the Benchmark

```bash
# Clone and install
git clone https://github.com/Mukund4321/agentbench-fail.git
cd agentbench-fail
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run baseline benchmark (no correction)
python -m agents.runner --mode baseline --model claude --tasks all

# Run with self-correction enabled
python -m agents.runner --mode corrected --model claude --tasks all

# Generate analysis + figures
python -m analysis.recovery_analysis --output results/figures/
```

---

## Limitations

- Mock tool environments are used for determinism; results may not fully generalize to live, non-deterministic APIs
- Failure classification combines automated heuristics with manual labeling — inter-rater reliability not yet formally measured
- Task suite size (40-50) is sufficient for directional findings but smaller than large-scale benchmarks like GAIA or AgentBench

---

## Roadmap

- [ ] Expand task suite to cover web-browsing / computer-use style tasks
- [ ] Add a fourth model for broader cross-model comparison
- [ ] Formalize inter-rater reliability on failure classification
- [ ] Submit preprint to arXiv (cs.AI / cs.CL)
- [ ] Explore IEEE workshop submission pending stronger empirical results

---

## Citation

If you use this benchmark or taxonomy in your own work:

```bibtex
@misc{agentbenchfail2026,
  author = {Mukund Saiteja},
  title = {AgentBench-Fail: Characterizing and Correcting Failure Modes in Long-Horizon LLM Agents},
  year = {2026},
  url = {https://github.com/Mukund4321/agentbench-fail}
}
```

---

## Author

**Mukund Saiteja**
Final-year CSE, SRM Institute of Science & Technology | ML Research Intern, IISc Bengaluru
[GitHub](https://github.com/Mukund4321) · ms9893@srmist.edu.in

---

## License

MIT License — see [LICENSE](LICENSE) for details.
