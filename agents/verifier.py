"""
Self-correction / verification layer.
After each agent step, Verifier checks whether the intermediate output
is consistent with the task goal and triggers retries if not.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerificationResult:
    passed: bool
    score: float          # 0.0 – 1.0 confidence that output is on-track
    reason: str
    correction_hint: Optional[str] = None


class Verifier:
    """
    Uses a separate LLM call to verify intermediate agent outputs.
    Falls back to heuristic checks if llm is None (for testing).
    """

    SYSTEM_PROMPT = (
        "You are a strict verifier. Given a task goal and an agent's intermediate output, "
        "decide whether the output is correct and on-track to achieve the goal.\n"
        "Respond with JSON: {\"passed\": bool, \"score\": float 0-1, \"reason\": str, "
        "\"correction_hint\": str or null}\n"
        "Be concise. Do not repeat the goal or output."
    )

    def __init__(self, llm=None, score_threshold: float = 0.6, max_retries: int = 2):
        self.llm = llm
        self.score_threshold = score_threshold
        self.max_retries = max_retries

    def verify_step(
        self,
        step_output: str,
        task_goal: str,
        step_number: int,
        context: Optional[str] = None,
    ) -> VerificationResult:
        """Check whether step_output is consistent with task_goal."""
        if self.llm is None:
            return self._heuristic_verify(step_output, task_goal)
        return self._llm_verify(step_output, task_goal, step_number, context)

    def _llm_verify(
        self,
        step_output: str,
        task_goal: str,
        step_number: int,
        context: Optional[str],
    ) -> VerificationResult:
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        ctx_str = f"\nContext so far:\n{context}" if context else ""
        prompt = (
            f"Task goal: {task_goal}\n"
            f"Step {step_number} output:\n{step_output}"
            f"{ctx_str}\n\n"
            "Is this step output correct and on-track?"
        )
        try:
            response = self.llm.invoke(
                [SystemMessage(content=self.SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            raw = response.content.strip()
            # Extract JSON block if wrapped in markdown
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return VerificationResult(
                    passed=bool(data.get("passed", False)),
                    score=float(data.get("score", 0.5)),
                    reason=str(data.get("reason", "")),
                    correction_hint=data.get("correction_hint"),
                )
        except Exception as e:
            return VerificationResult(passed=True, score=0.5, reason=f"Verifier error: {e}")
        return VerificationResult(passed=True, score=0.5, reason="Could not parse verifier response.")

    def _heuristic_verify(self, step_output: str, task_goal: str) -> VerificationResult:
        """Lightweight heuristic check: fail if output is empty or contains error markers."""
        if not step_output or len(step_output.strip()) < 3:
            return VerificationResult(
                passed=False, score=0.0, reason="Empty output.",
                correction_hint="The step returned no content. Retry with correct tool arguments."
            )
        error_markers = ["error:", "exception:", "traceback", "not found", "failed", "undefined"]
        lower = step_output.lower()
        for marker in error_markers:
            if marker in lower:
                return VerificationResult(
                    passed=False, score=0.2, reason=f"Output contains error indicator: '{marker}'.",
                    correction_hint="Fix the tool call that produced this error before proceeding."
                )
        return VerificationResult(passed=True, score=0.8, reason="Heuristic check passed.")

    def build_correction_message(
        self, failed_result: VerificationResult, step_number: int, task_goal: str
    ) -> str:
        """Build a correction prompt to inject into the agent's context."""
        hint = failed_result.correction_hint or "Review the previous step and correct it."
        return (
            f"[VERIFICATION FAILURE at step {step_number}] "
            f"Reason: {failed_result.reason} "
            f"Hint: {hint} "
            f"Task goal: {task_goal}. "
            "Please correct the previous step before continuing."
        )
