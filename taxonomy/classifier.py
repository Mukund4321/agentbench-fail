"""
Failure taxonomy classifier for AgentBench-Fail.

Taxonomy (5 categories):
  1. tool_call_error        — wrong tool, malformed args, hallucinated tool
  2. context_memory_failure — agent loses earlier state or contradicts itself
  3. compounding_error      — small early error cascades to full failure
  4. premature_termination  — agent stops before goal is fully achieved
  5. verification_blindness — agent never checks output against goal

Classification combines rule-based heuristics over the trace JSON.
A trace can have multiple contributing failure types; the primary one is returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tracer.trace_logger import StepTrace


class FailureType(str, Enum):
    TOOL_CALL_ERROR = "tool_call_error"
    CONTEXT_MEMORY_FAILURE = "context_memory_failure"
    COMPOUNDING_ERROR = "compounding_error"
    PREMATURE_TERMINATION = "premature_termination"
    VERIFICATION_BLINDNESS = "verification_blindness"
    SUCCESS = "success"


@dataclass
class FailureLabel:
    primary: FailureType
    contributing: list[FailureType] = field(default_factory=list)
    evidence_step: Optional[int] = None
    confidence: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.value,
            "contributing": [f.value for f in self.contributing],
            "evidence_step": self.evidence_step,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


class FailureClassifier:
    """
    Classifies a task trace into one of the 5 taxonomy failure categories.

    Usage:
        label = FailureClassifier().classify(task, trace_steps, success)
    """

    # Error keywords that signal tool-call failures
    _TOOL_ERROR_PATTERNS = [
        r"error\b", r"exception\b", r"traceback", r"valueerror", r"keyerror",
        r"typeerror", r"not found", r"invalid argument", r"malformed", r"no such tool",
        r"tool .* does not exist", r"undefined function", r"404", r"500",
    ]
    _TOOL_ERROR_RE = re.compile("|".join(_TOOL_ERROR_PATTERNS), re.IGNORECASE)

    # Phrases that indicate premature completion
    _PREMATURE_DONE_PATTERNS = [
        r"task (is |has been )?complete",
        r"done\.", r"finished\.", r"i have (completed|finished)",
        r"the (answer|result) is",
        r"here (is|are) the (results?|answers?)",
    ]
    _PREMATURE_RE = re.compile("|".join(_PREMATURE_DONE_PATTERNS), re.IGNORECASE)

    # Context-loss markers: agent re-asks for already-provided info
    _CONTEXT_LOSS_PATTERNS = [
        r"could you (please )?provide", r"please (share|give me|tell me)",
        r"what (is|are) the .*(again|value|input)",
        r"i (don'?t|do not) (have|see) .*(information|data|value)",
        r"as (mentioned|stated|provided) (earlier|above|before)",
    ]
    _CONTEXT_LOSS_RE = re.compile("|".join(_CONTEXT_LOSS_PATTERNS), re.IGNORECASE)

    def classify(
        self,
        task: dict,
        steps: list[StepTrace],
        success: bool,
    ) -> FailureLabel:
        if success:
            return FailureLabel(
                primary=FailureType.SUCCESS,
                confidence=1.0,
                explanation="Task completed successfully.",
            )

        candidates: list[FailureLabel] = []

        lbl = self._check_tool_call_error(steps)
        if lbl:
            candidates.append(lbl)

        lbl = self._check_premature_termination(task, steps)
        if lbl:
            candidates.append(lbl)

        lbl = self._check_context_memory_failure(steps)
        if lbl:
            candidates.append(lbl)

        lbl = self._check_compounding_error(steps)
        if lbl:
            candidates.append(lbl)

        lbl = self._check_verification_blindness(task, steps)
        if lbl:
            candidates.append(lbl)

        if not candidates:
            return FailureLabel(
                primary=FailureType.VERIFICATION_BLINDNESS,
                confidence=0.5,
                explanation="No specific failure pattern detected; defaulting to verification blindness.",
            )

        # Pick the highest-confidence label as primary
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        primary = candidates[0]
        primary.contributing = [c.primary for c in candidates[1:]]
        return primary

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_tool_call_error(self, steps: list[StepTrace]) -> Optional[FailureLabel]:
        for step in steps:
            output = step.output or ""
            if self._TOOL_ERROR_RE.search(output):
                return FailureLabel(
                    primary=FailureType.TOOL_CALL_ERROR,
                    evidence_step=step.step_num,
                    confidence=0.90,
                    explanation=f"Tool error pattern detected in step {step.step_num} output: '{output[:120]}'",
                )
            # Hallucinated tool: tool_name not in known mock tools
            if step.tool_name and step.tool_name not in _KNOWN_MOCK_TOOLS and step.tool_name != "__llm__":
                return FailureLabel(
                    primary=FailureType.TOOL_CALL_ERROR,
                    evidence_step=step.step_num,
                    confidence=0.85,
                    explanation=f"Unrecognized tool '{step.tool_name}' called at step {step.step_num}.",
                )
        return None

    def _check_premature_termination(self, task: dict, steps: list[StepTrace]) -> Optional[FailureLabel]:
        expected_steps = task.get("expected_steps", 1)
        actual_steps = len(steps)

        # Fewer than half expected steps taken → likely cut short
        if actual_steps < max(1, expected_steps // 2):
            return FailureLabel(
                primary=FailureType.PREMATURE_TERMINATION,
                evidence_step=actual_steps,
                confidence=0.88,
                explanation=(
                    f"Only {actual_steps} steps taken; task expected ~{expected_steps}. "
                    "Agent terminated before completing all required steps."
                ),
            )

        # "Done" language in non-final step
        for i, step in enumerate(steps[:-1]):
            if self._PREMATURE_RE.search(step.output or ""):
                return FailureLabel(
                    primary=FailureType.PREMATURE_TERMINATION,
                    evidence_step=step.step_num,
                    confidence=0.80,
                    explanation=f"Agent declared completion prematurely at step {step.step_num}.",
                )
        return None

    def _check_context_memory_failure(self, steps: list[StepTrace]) -> Optional[FailureLabel]:
        for step in steps:
            if self._CONTEXT_LOSS_RE.search(step.output or ""):
                return FailureLabel(
                    primary=FailureType.CONTEXT_MEMORY_FAILURE,
                    evidence_step=step.step_num,
                    confidence=0.82,
                    explanation=f"Context/memory loss detected at step {step.step_num}: agent re-requested already-provided information.",
                )

        # Detect contradiction: same tool called with same args but different expected output
        seen_calls: dict[tuple, str] = {}
        for step in steps:
            if step.tool_name and step.tool_name != "__llm__":
                key = (step.tool_name, str(sorted(step.tool_args.items()) if step.tool_args else []))
                if key in seen_calls and seen_calls[key] != (step.output or ""):
                    return FailureLabel(
                        primary=FailureType.CONTEXT_MEMORY_FAILURE,
                        evidence_step=step.step_num,
                        confidence=0.78,
                        explanation=f"Agent re-called '{step.tool_name}' with identical args at step {step.step_num}, contradicting prior result.",
                    )
                seen_calls[key] = step.output or ""
        return None

    def _check_compounding_error(self, steps: list[StepTrace]) -> Optional[FailureLabel]:
        """Detect cascading errors: error in early step + errors in all subsequent steps."""
        if len(steps) < 3:
            return None

        first_error_idx = None
        for i, step in enumerate(steps):
            if self._TOOL_ERROR_RE.search(step.output or ""):
                first_error_idx = i
                break

        if first_error_idx is None or first_error_idx >= len(steps) - 1:
            return None

        # Check if errors persist in downstream steps
        downstream_errors = sum(
            1 for step in steps[first_error_idx + 1:]
            if self._TOOL_ERROR_RE.search(step.output or "")
        )
        downstream_total = len(steps) - first_error_idx - 1

        if downstream_total > 0 and downstream_errors / downstream_total >= 0.5:
            return FailureLabel(
                primary=FailureType.COMPOUNDING_ERROR,
                evidence_step=steps[first_error_idx].step_num,
                confidence=0.85,
                explanation=(
                    f"Error at step {steps[first_error_idx].step_num} cascaded into "
                    f"{downstream_errors}/{downstream_total} subsequent steps."
                ),
            )
        return None

    def _check_verification_blindness(self, task: dict, steps: list[StepTrace]) -> Optional[FailureLabel]:
        """Detect if agent never cross-checks output against the task goal."""
        # Heuristic: no step output mentions checking, validating, or verifying
        verification_keywords = r"check|verif|validat|confirm|assert|ensure|match|correct"
        vk_re = re.compile(verification_keywords, re.IGNORECASE)

        has_self_check = any(vk_re.search(step.output or "") for step in steps)
        if not has_self_check:
            return FailureLabel(
                primary=FailureType.VERIFICATION_BLINDNESS,
                evidence_step=len(steps),
                confidence=0.72,
                explanation="Agent produced no self-verification steps before declaring the result.",
            )
        return None

    def batch_classify(
        self,
        results: list[dict],  # list of TaskResult dicts from the trace files
    ) -> list[dict]:
        """Classify a batch of trace files and return annotated results."""
        from tracer.trace_logger import StepTrace
        annotated = []
        for r in results:
            steps = [StepTrace(**s) for s in r.get("trace", [])]
            task = {"expected_steps": r.get("steps_taken", 1), "goal": ""}
            label = self.classify(task, steps, r.get("success", False))
            annotated.append({**r, "failure_type": label.primary.value, "failure_label": label.to_dict()})
        return annotated


# Known tools from mock_tools.py — used for hallucination detection
_KNOWN_MOCK_TOOLS = {
    "convert_units", "calculate", "validate_range", "lookup_product", "lookup_user",
    "fetch_weather", "search_database", "format_output", "read_file", "write_file",
    "aggregate_data", "merge_datasets", "compute_stats", "filter_data", "rank_items",
    "classify_bmi", "parse_date", "date_diff", "format_currency", "convert_currency",
    "classify_temperature", "sort_list", "slice_list", "count_words",
    "compute_word_frequency", "calculate_area", "compare_values",
    "compute_fibonacci", "check_prime", "json_extract", "validate_type",
    "string_strip", "string_normalize", "search_in_text", "normalize",
    "classify_score", "fetch_feed", "classify_text", "compute_nutrition",
    "scale_ingredients", "lookup_recipe", "query_inventory", "extract_facts",
    "deduplicate", "format_summary", "resolve_contradictions", "synthesize_content",
    "validate_stats", "format_report", "validate_report", "clean_data",
    "compute_financial_metrics", "compute_growth_rates", "identify_extremes",
    "stress_test", "run_regression", "load_criteria", "fetch_evidence",
    "extract_passage", "evaluate_criterion", "tally_results", "prioritize_failures",
    "generate_remediation", "extract_competitor_data", "normalize_metrics",
    "compute_position_matrix", "identify_gaps", "generate_recommendations",
    "load_constraints", "validate_recommendations", "check_field_presence",
    "check_email_format", "check_consent", "check_minimization", "check_retention",
    "check_anonymization", "check_cross_border", "check_purpose", "check_erasure",
    "check_audit_trail", "compute_compliance_score", "apply_fft",
    "extract_frequencies", "bandpass_filter", "compute_snr", "detect_peaks",
    "classify_peaks", "correlate_events", "compute_probability",
    "extract_entities", "classify_entities", "extract_relations",
    "resolve_coreferences", "validate_ontology", "compute_graph_metrics",
    "detect_clusters", "enrich_entities", "serialize_graph", "validate_schema",
    "build_scenario", "run_dcf", "compute_sensitivity", "identify_breakeven",
    "create_outline", "draft_section", "fact_check", "fix_inaccuracies",
    "check_readability", "adjust_complexity", "seo_analyze", "format_html",
    "validate_html", "parse_logs", "detect_anomalies", "root_cause_analysis",
    "simulate_fix", "verify_fix", "lookup_employee", "determine_permissions",
    "create_iam_account", "assign_policies", "provision_licenses", "create_email",
    "send_email", "update_org_chart", "schedule_meetings", "create_jira_ticket",
    "audit_actions", "validate_checklist", "query_database", "compare_samples",
    "validate_foreign_keys", "check_indexes", "check_triggers",
    "validate_procedures", "run_equivalence_queries", "compute_checksum",
    "generate_migration_report", "profile_data", "handle_missing",
    "encode_features", "split_dataset", "train_model", "evaluate_models",
    "select_best_model", "tune_hyperparameters", "evaluate_model",
    "write_model_card", "acknowledge_alert", "triage_incident", "page_oncall",
    "investigate_logs", "identify_affected", "estimate_impact", "apply_mitigation",
    "verify_mitigation", "restore_service", "conduct_postmortem",
    "extract_action_items", "create_jira_tickets", "load_performance_data",
    "run_swot", "benchmark_competitors", "define_okrs", "generate_options",
    "evaluate_options", "select_option", "build_roadmap", "compute_resources",
    "compute_roi", "identify_risks", "format_deck", "validate_financials",
    "search_flights", "search_hotels", "lookup_visa_fee", "validate_budget",
    "load_calendar", "detect_conflicts", "propose_slots", "validate_schedule",
    "format_schedule", "score_items", "load_expenses", "categorize_expenses",
    "detect_anomalies", "analyze_complexity", "check_naming", "check_docs",
    "detect_duplicates", "prioritize_issues", "lookup_order", "lookup_customer",
    "lookup_address", "lookup_timezone", "convert_timezone", "format_datetime",
    "lookup_client", "compute_credit_score", "compute_debt_ratio",
    "compute_income_stability", "compute_collateral", "compute_payment_history",
    "classify_risk",
}
