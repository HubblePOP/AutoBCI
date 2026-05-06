from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ProviderProtocol = Literal["openai_compatible", "anthropic_compatible", "fake"]


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    protocol: ProviderProtocol
    base_url: str
    api_key_env: str | None
    default_model: str
    capabilities: tuple[str, ...]
    capability_profile: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = list(self.capabilities)
        return payload


_OPENAI_JSON_BODY = {"response_format": {"type": "json_object"}}


def _profile(
    *,
    chat: bool,
    json_schema: bool,
    tool_calls: str,
    streaming: bool,
    reasoning: str,
    context: str,
    coding_suitability: str,
) -> dict[str, Any]:
    return {
        "chat": chat,
        "json_schema": json_schema,
        "tool_calls": tool_calls,
        "streaming": streaming,
        "reasoning": reasoning,
        "context": context,
        "coding_suitability": coding_suitability,
    }


OPENAI_COMPATIBLE_PROFILE = _profile(
    chat=True,
    json_schema=True,
    tool_calls="adapter_json_actions",
    streaming=False,
    reasoning="provider_dependent",
    context="provider_default",
    coding_suitability="smoke_supported",
)


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        protocol="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        capabilities=("chat_completions", "json_task"),
        capability_profile=OPENAI_COMPATIBLE_PROFILE,
        extra_body=_OPENAI_JSON_BODY,
    ),
    "kimi": ProviderPreset(
        name="kimi",
        protocol="openai_compatible",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        default_model="moonshot-v1-8k",
        capabilities=("chat_completions", "json_task"),
        capability_profile=OPENAI_COMPATIBLE_PROFILE,
        extra_body=_OPENAI_JSON_BODY,
    ),
    "glm": ProviderPreset(
        name="glm",
        protocol="openai_compatible",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
        default_model="glm-4-plus",
        capabilities=("chat_completions", "json_task"),
        capability_profile=OPENAI_COMPATIBLE_PROFILE,
        extra_body=_OPENAI_JSON_BODY,
    ),
    "minimax": ProviderPreset(
        name="minimax",
        protocol="openai_compatible",
        base_url="https://api.minimax.chat/v1",
        api_key_env="MINIMAX_API_KEY",
        default_model="abab6.5s-chat",
        capabilities=("chat_completions", "json_task"),
        capability_profile=OPENAI_COMPATIBLE_PROFILE,
        extra_body=_OPENAI_JSON_BODY,
    ),
    "openai": ProviderPreset(
        name="openai",
        protocol="openai_compatible",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        capabilities=("chat_completions", "json_task"),
        capability_profile=OPENAI_COMPATIBLE_PROFILE,
        extra_body=_OPENAI_JSON_BODY,
    ),
    "anthropic": ProviderPreset(
        name="anthropic",
        protocol="anthropic_compatible",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-3-5-sonnet-latest",
        capabilities=("messages",),
        capability_profile=_profile(
            chat=True,
            json_schema=False,
            tool_calls="not_enabled_in_this_slice",
            streaming=False,
            reasoning="provider_dependent",
            context="provider_default",
            coding_suitability="declared_not_live",
        ),
        extra_body={},
    ),
    "fake": ProviderPreset(
        name="fake",
        protocol="fake",
        base_url="fake://local",
        api_key_env=None,
        default_model="fake-json-v1",
        capabilities=("json_task", "edit_turn"),
        capability_profile=_profile(
            chat=True,
            json_schema=True,
            tool_calls="deterministic_json_actions",
            streaming=False,
            reasoning="none",
            context="local_test",
            coding_suitability="ci_smoke",
        ),
        extra_body={},
    ),
}


def list_provider_presets() -> list[str]:
    return sorted(PROVIDER_PRESETS)


def get_provider_preset(name: str) -> ProviderPreset:
    key = str(name or "").strip().lower()
    try:
        return PROVIDER_PRESETS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {name}") from exc
