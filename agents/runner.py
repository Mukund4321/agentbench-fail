"""
Core agent execution loop using LangGraph.

Architecture:
  agent_node → tool_node → (verifier_node) → agent_node → … → END

Each task is run with a fixed set of mock tools. The runner captures
a full step-by-step trace for downstream failure classification.
"""
from __future__ import annotations

import time
import argparse
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .models import ModelFactory
from .verifier import Verifier, VerificationResult
from tools.mock_tools import get_tools_for_task
from tracer.trace_logger import TraceLogger, StepTrace
from tasks.loader import load_tasks, load_task_by_id


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    task_goal: str
    horizon: str
    steps_taken: int
    max_steps: int
    verification_enabled: bool
    verification_failures: int
    max_verification_retries: int
    trace: list[dict]          # raw step dicts for the logger
    start_time: float
    token_count: int


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    model: str
    mode: str                  # "baseline" | "corrected"
    success: bool
    steps_taken: int
    failure_type: Optional[str]    # taxonomy label or None
    trace: list[StepTrace]
    total_tokens: int
    total_latency_s: float
    final_answer: Optional[str]
    error_message: Optional[str] = None


# ─── Runner ───────────────────────────────────────────────────────────────────

class AgentRunner:
    SYSTEM_PROMPT = (
        "You are a precise task-executing agent. Follow the task goal exactly. "
        "Use the available tools in the correct sequence. "
        "Do NOT declare success until every part of the goal is satisfied. "
        "If a tool returns an error, diagnose the problem and retry with corrected arguments."
    )
    MAX_STEPS_BY_HORIZON = {"short": 6, "medium": 14, "long": 26}

    def __init__(self, model_name: str, mode: str = "baseline", verifier_llm=None):
        self.model_name = model_name
        self.mode = mode
        self.llm = ModelFactory.create(model_name)
        self.verifier = Verifier(llm=verifier_llm) if mode == "corrected" else None
        self.logger = TraceLogger()

    def run_task(self, task: dict) -> TaskResult:
        """Execute a single task and return a TaskResult with full trace."""
        task_id = task["task_id"]
        horizon = task["horizon"]
        goal = task["goal"]
        tools = get_tools_for_task(task)

        llm_with_tools = self.llm.bind_tools(tools)
        max_steps = self.MAX_STEPS_BY_HORIZON.get(horizon, 20)

        graph = self._build_graph(llm_with_tools, tools, max_steps)

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=self._build_task_prompt(task)),
            ],
            "task_id": task_id,
            "task_goal": goal,
            "horizon": horizon,
            "steps_taken": 0,
            "max_steps": max_steps,
            "verification_enabled": self.mode == "corrected",
            "verification_failures": 0,
            "max_verification_retries": 2,
            "trace": [],
            "start_time": time.time(),
            "token_count": 0,
        }

        self.logger.start(task_id, self.model_name, self.mode)
        start = time.time()

        try:
            final_state = graph.invoke(initial_state)
            elapsed = time.time() - start

            last_ai = next(
                (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage)), None
            )
            final_answer = last_ai.content if last_ai else None

            success = self._evaluate_success(task, final_state)
            self.logger.finish(success, final_state["steps_taken"], final_state["token_count"], elapsed)

            return TaskResult(
                task_id=task_id,
                model=self.model_name,
                mode=self.mode,
                success=success,
                steps_taken=final_state["steps_taken"],
                failure_type=None,  # filled by classifier post-hoc
                trace=self.logger.build_trace(),
                total_tokens=final_state["token_count"],
                total_latency_s=elapsed,
                final_answer=final_answer,
            )
        except Exception as exc:
            elapsed = time.time() - start
            self.logger.finish(False, 0, 0, elapsed)
            return TaskResult(
                task_id=task_id,
                model=self.model_name,
                mode=self.mode,
                success=False,
                steps_taken=0,
                failure_type="tool_call_error",
                trace=self.logger.build_trace(),
                total_tokens=0,
                total_latency_s=elapsed,
                final_answer=None,
                error_message=str(exc),
            )

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self, llm_with_tools, tools: list[BaseTool], max_steps: int):
        tool_node = ToolNode(tools)
        verifier = self.verifier

        def agent_node(state: AgentState) -> AgentState:
            t0 = time.time()
            response = llm_with_tools.invoke(state["messages"])
            latency = time.time() - t0

            usage = getattr(response, "usage_metadata", None)
            tokens = (usage.get("total_tokens", 0) if isinstance(usage, dict)
                      else getattr(usage, "total_tokens", 0)) if usage else 0

            step_num = state["steps_taken"] + 1
            tool_calls = getattr(response, "tool_calls", []) or []
            self.logger.log_step(
                step_num=step_num,
                tool_name=tool_calls[0]["name"] if tool_calls else "__llm__",
                tool_args=tool_calls[0]["args"] if tool_calls else {},
                output=response.content or "",
                latency_s=latency,
                tokens=tokens,
            )

            return {
                **state,
                "messages": state["messages"] + [response],
                "steps_taken": step_num,
                "token_count": state["token_count"] + tokens,
            }

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if state["steps_taken"] >= state["max_steps"]:
                return END
            if not getattr(last, "tool_calls", None):
                return END
            return "tools"

        def after_tools(state: AgentState) -> str:
            if state["steps_taken"] >= state["max_steps"]:
                return END
            if verifier and state["verification_enabled"]:
                return "verifier"
            return "agent"

        def verifier_node(state: AgentState) -> AgentState:
            last_tool_msg = next(
                (m for m in reversed(state["messages"])
                 if m.__class__.__name__ == "ToolMessage"), None
            )
            if not last_tool_msg:
                return state

            result: VerificationResult = verifier.verify_step(
                step_output=last_tool_msg.content,
                task_goal=state["task_goal"],
                step_number=state["steps_taken"],
                context=None,
            )

            if not result.passed and state["verification_failures"] < state["max_verification_retries"]:
                correction = verifier.build_correction_message(
                    result, state["steps_taken"], state["task_goal"]
                )
                return {
                    **state,
                    "messages": state["messages"] + [HumanMessage(content=correction)],
                    "verification_failures": state["verification_failures"] + 1,
                }
            return state

        def verifier_router(state: AgentState) -> str:
            return "agent"

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)

        graph_builder.set_entry_point("agent")
        graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

        if verifier:
            graph_builder.add_node("verifier", verifier_node)
            graph_builder.add_conditional_edges("tools", after_tools,
                                                {"verifier": "verifier", "agent": "agent", END: END})
            graph_builder.add_conditional_edges("verifier", verifier_router, {"agent": "agent"})
        else:
            graph_builder.add_conditional_edges("tools", after_tools,
                                                {"agent": "agent", END: END})

        return graph_builder.compile()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_task_prompt(task: dict) -> str:
        lines = [
            f"TASK: {task['name']}",
            f"DESCRIPTION: {task['description']}",
            f"GOAL: {task['goal']}",
        ]
        if task.get("inputs"):
            lines.append(f"INPUTS: {task['inputs']}")
        return "\n".join(lines)

    @staticmethod
    def _evaluate_success(task: dict, state: AgentState) -> bool:
        """
        Heuristic success check: look for ground truth keys in the last AI message.
        A proper implementation compares structured output to ground_truth.
        """
        gt = task.get("ground_truth", {})
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if not last_ai or not last_ai.content:
            return False
        content = last_ai.content.lower()
        # Simple: check that at least half of ground truth values appear in output
        gt_values = [str(v).lower() for v in gt.values()]
        matches = sum(1 for v in gt_values if v in content)
        return matches >= max(1, len(gt_values) // 2)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run AgentBench-Fail for a single task.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", default="claude", choices=["claude", "gpt", "openweight"])
    parser.add_argument("--mode", default="baseline", choices=["baseline", "corrected"])
    parser.add_argument("--output", default="results/raw_traces/")
    args = parser.parse_args()

    import json, os
    from pathlib import Path

    task = load_task_by_id(args.task_id)
    runner = AgentRunner(model_name=args.model, mode=args.mode)
    result = runner.run_task(task)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task_id}_{args.model}_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": result.task_id,
                "model": result.model,
                "mode": result.mode,
                "success": result.success,
                "steps_taken": result.steps_taken,
                "failure_type": result.failure_type,
                "total_tokens": result.total_tokens,
                "total_latency_s": result.total_latency_s,
                "final_answer": result.final_answer,
                "trace": [vars(s) for s in result.trace],
            },
            f,
            indent=2,
        )
    print(f"[{'OK' if result.success else 'FAIL'}] {args.task_id} | {args.model} | {args.mode}")
    print(f"  steps={result.steps_taken}, tokens={result.total_tokens}, latency={result.total_latency_s:.2f}s")
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
