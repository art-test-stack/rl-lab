"""Inference adapters with lazy imports of optional backend dependencies."""

from .base import (
    ChatPrompt,
    CompletionEngine,
    EngineCapabilities,
    Generation,
    GenerationConfig,
    LogLikelihoodRequest,
    Message,
    ModelInput,
    TextPrompt,
)
from .router import EngineConfig, EngineRouter, create_engine
from .transformers import TransformersEngine
from .vllm import VLLMEngine

__all__ = [
    "ChatPrompt",
    "CompletionEngine",
    "EngineCapabilities",
    "EngineConfig",
    "EngineRouter",
    "Generation",
    "GenerationConfig",
    "LogLikelihoodRequest",
    "Message",
    "ModelInput",
    "TextPrompt",
    "TransformersEngine",
    "VLLMEngine",
    "create_engine",
]
