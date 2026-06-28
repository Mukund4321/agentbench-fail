"""
Model client wrappers for Claude, GPT, and open-weight models.
All models are wrapped as LangChain ChatModels for LangGraph compatibility.
"""
import os
from typing import Optional


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val


class ModelFactory:
    """Create LangChain chat models by provider name."""

    SUPPORTED = {"claude", "gpt", "openweight"}

    @staticmethod
    def create(model_name: str, temperature: float = 0.0):
        """
        model_name: one of 'claude', 'gpt', 'openweight'
        Returns a LangChain BaseChatModel with tools-bind support.
        """
        if model_name == "claude":
            return ModelFactory._claude(temperature)
        elif model_name == "gpt":
            return ModelFactory._gpt(temperature)
        elif model_name == "openweight":
            return ModelFactory._openweight(temperature)
        else:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {ModelFactory.SUPPORTED}.")

    @staticmethod
    def _claude(temperature: float = 0.0):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Run: pip install langchain-anthropic")
        api_key = _require_env("ANTHROPIC_API_KEY")
        model_id = os.environ.get("CLAUDE_MODEL_ID", "claude-sonnet-4-6")
        return ChatAnthropic(
            model=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4096,
        )

    @staticmethod
    def _gpt(temperature: float = 0.0):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        api_key = _require_env("OPENAI_API_KEY")
        model_id = os.environ.get("GPT_MODEL_ID", "gpt-4o")
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4096,
        )

    @staticmethod
    def _openweight(temperature: float = 0.0):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        api_key = _require_env("TOGETHER_API_KEY")
        model_id = os.environ.get("OPENWEIGHT_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            temperature=temperature,
            max_tokens=4096,
        )
