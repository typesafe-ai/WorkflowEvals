from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, model_validator
from typesafe_sdk.constants import DEFAULT_BASE_URL

from .contract import ModelIdentity, Record

Provider = Literal["openai", "anthropic", "fireworks", "groq", "cerebras", "typesafe"]
PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "fireworks", "groq", "cerebras", "typesafe")
# Hosts serving models from multiple labs.
OPEN_MODEL_HOSTS: frozenset[Provider] = frozenset({"fireworks", "groq", "cerebras"})
# Reject unsupported effort levels rather than letting PydanticAI remap them.
GRADED_EFFORTS = ("default", "off", "low", "medium", "high")


class ModelConfig(Record):
    model: str = Field(min_length=1)
    provider: Provider
    # default leaves reasoning settings to the provider.
    thinking: Literal["default", "off", "minimal", "low", "medium", "high", "xhigh", "max"] = "default"
    # all sends the node together; one sends independent requests per question.
    question_mode: Literal["all", "one"] = "all"
    base_url: str | None = None
    timeout_s: float = Field(default=600, gt=0)
    max_output_tokens: int = Field(default=64000, gt=0)
    malformed_retries: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def applicable_settings(self):
        if self.provider == "typesafe" and self.model != "jev-1.13.0":
            raise ValueError("TypeSafe supports only jev-1.13.0")
        if self.provider == "typesafe" and self.thinking not in ("default", "off"):
            raise ValueError("TypeSafe has no thinking setting; use --thinking off")
        if self.base_url and self.provider != "typesafe":
            raise ValueError("--base-url currently applies to TypeSafe only")
        if self.provider in ("groq", "cerebras") and self.thinking not in GRADED_EFFORTS:
            raise ValueError(
                f"{self.provider} grades reasoning effort as low, medium, or high; "
                f"--thinking {self.thinking} would be silently changed, so it is rejected"
            )
        return self

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            name=f"{self.provider}:{self.model}",
            lab="other" if self.provider in OPEN_MODEL_HOSTS else self.provider,
            effort=None if self.provider == "typesafe" else self.thinking,
        )


def configuration(model: str, thinking: str | None = None, **kwargs) -> ModelConfig:
    provider, separator, model = model.partition(":")
    if not separator or not model.strip():
        raise ValueError("Use provider:model, e.g. openai:gpt-6-astra")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; choose {', '.join(PROVIDERS)}")
    if provider == "typesafe":
        kwargs["base_url"] = (
            kwargs.get("base_url")
            or os.getenv("TYPESAFE_BASE_URL")
            or DEFAULT_BASE_URL
        )
    # Default to joint questions for TypeSafe and independent questions for LLMs.
    question_mode = kwargs.pop("question_mode", None) or ("all" if provider == "typesafe" else "one")
    return ModelConfig(
        model=model,
        provider=provider,
        thinking=thinking or ("off" if provider == "typesafe" else "default"),
        question_mode=question_mode,
        **kwargs,
    )


def resolve_model(config: ModelConfig):
    if config.provider == "typesafe":
        return config.model
    settings = {"timeout": config.timeout_s, "max_tokens": config.max_output_tokens}
    explicit = config.thinking != "default"
    if config.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        if explicit:
            settings["thinking"] = False if config.thinking == "off" else config.thinking
        if config.thinking == "off":
            settings["anthropic_thinking"] = {"type": "disabled"}
        return AnthropicModel(config.model, settings=settings)
    if config.provider in ("groq", "cerebras"):
        if explicit:
            settings["thinking"] = False if config.thinking == "off" else config.thinking
        if config.provider == "groq":
            from pydantic_ai.models.groq import GroqModel
            from pydantic_ai.providers.groq import GroqProvider

            return GroqModel(config.model, provider=GroqProvider(), settings=settings)
        from pydantic_ai.providers.cerebras import CerebrasProvider

        return _chat_model(config, CerebrasProvider(), settings)
    if explicit:
        settings["openai_reasoning_effort"] = "none" if config.thinking == "off" else config.thinking
    if config.provider == "openai":
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(config.model, settings=settings)
    from pydantic_ai.providers.fireworks import FireworksProvider

    return _chat_model(config, FireworksProvider(), settings)


def _chat_model(config: ModelConfig, provider, settings: dict):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles import merge_profile
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    return OpenAIChatModel(
        config.model,
        provider=provider,
        settings=settings,
        profile=lambda base: merge_profile(base, OpenAIModelProfile(supports_json_schema_output=True)),
    )


def make_client(config: ModelConfig):
    from .clients import QuestionClient

    return QuestionClient(config)
