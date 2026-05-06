from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .presets import ProviderPreset


class ProviderError(RuntimeError):
    error_code = "provider_error"


class MissingProviderKey(ProviderError):
    error_code = "missing_api_key"

    def __init__(self, provider: str, env_name: str) -> None:
        super().__init__(f"{provider} requires {env_name}")
        self.provider = provider
        self.env_name = env_name


class UnsupportedProviderProtocol(ProviderError):
    error_code = "unsupported_protocol"


@dataclass(frozen=True)
class ProviderClient:
    preset: ProviderPreset
    model: str
    timeout_seconds: float = 30.0

    def generate_json(self, task: dict[str, Any]) -> dict[str, Any]:
        if self.preset.protocol == "fake":
            return self._fake_json(task)
        if self.preset.protocol == "anthropic_compatible":
            raise UnsupportedProviderProtocol(
                "Anthropic-compatible live calls are intentionally stubbed in this first runtime slice."
            )
        if self.preset.protocol == "openai_compatible":
            return self._openai_compatible_json(task)
        raise UnsupportedProviderProtocol(f"Unsupported provider protocol: {self.preset.protocol}")

    def _fake_json(self, task: dict[str, Any]) -> dict[str, Any]:
        prompt = str(task.get("prompt") or task.get("message") or "").strip()
        return {
            "ok": True,
            "provider": "fake",
            "model": self.model,
            "echo": prompt[:200],
            "proposal": {
                "hypothesis": "Provider runtime plumbing can be tested safely with the fake provider.",
                "why_this_change": "The fake provider gives CI a deterministic JSON path without secrets or network.",
                "changes_summary": "Return a typed proposal shell and keep file writes inside the runtime/provider layer.",
                "change_bucket": "runtime_provider",
                "track_comparison_note": "No research track comparison is claimed by this runtime-only turn.",
                "files_touched": [],
                "next_step": "Run the provider/runtime test file and inspect the ledgers.",
                "search_queries": [],
                "research_evidence": [],
            },
        }

    def _openai_compatible_json(self, task: dict[str, Any]) -> dict[str, Any]:
        if not self.preset.api_key_env:
            raise MissingProviderKey(self.preset.name, "")
        api_key = os.environ.get(self.preset.api_key_env)
        if not api_key:
            raise MissingProviderKey(self.preset.name, self.preset.api_key_env)
        url = self.preset.base_url.rstrip("/") + "/chat/completions"
        prompt = str(task.get("prompt") or task.get("message") or "Return a JSON object.")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not include markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(task.get("temperature", 0.2)),
        }
        body.update(self.preset.extra_body)
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"provider_http_error:{exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError("provider_network_error") from exc
        content = payload["choices"][0]["message"]["content"]
        return json.loads(content)
