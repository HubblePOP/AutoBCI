from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback keeps config readable enough.
    tomllib = None  # type: ignore[assignment]

from .presets import get_provider_preset


CONFIG_ENV = "AUTOBCI_PROVIDER_CONFIG"
DEFAULT_PROVIDER_ENV = "AUTOBCI_DEFAULT_PROVIDER"
DEFAULT_MODEL_ENV = "AUTOBCI_DEFAULT_MODEL"


def get_provider_config_path() -> Path:
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    if platform.system().lower().startswith("win"):
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "AutoBci" / "providers.toml"
    return Path.home() / ".config" / "autobci" / "providers.toml"


def load_provider_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else get_provider_config_path()
    if not path.exists():
        return {}
    if tomllib is None:
        return {}
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload if isinstance(payload, dict) else {}


def _toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_provider_config(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    path = Path(config_path) if config_path is not None else get_provider_config_path()
    lines: list[str] = []
    default_provider = str(config.get("default_provider") or "").strip()
    default_model = str(config.get("default_model") or "").strip()
    if default_provider:
        lines.append(f"default_provider = {_toml_quote(default_provider)}")
    if default_model:
        lines.append(f"default_model = {_toml_quote(default_model)}")
    providers = config.get("providers")
    if isinstance(providers, dict):
        for name in sorted(providers):
            if not isinstance(providers[name], dict):
                continue
            lines.append("")
            lines.append(f"[providers.{name}]")
            for key, value in sorted(providers[name].items()):
                if value is None:
                    continue
                lines.append(f"{key} = {_toml_quote(str(value))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def resolve_default_provider(config: dict[str, Any] | None = None) -> str:
    env_value = os.environ.get(DEFAULT_PROVIDER_ENV)
    if env_value:
        return env_value.strip().lower()
    cfg_value = (config or {}).get("default_provider")
    if cfg_value:
        return str(cfg_value).strip().lower()
    return "fake"


def resolve_model(provider_name: str, config: dict[str, Any] | None = None, override: str | None = None) -> str:
    if override:
        return override
    env_value = os.environ.get(DEFAULT_MODEL_ENV)
    if env_value:
        return env_value.strip()
    cfg = config or {}
    provider_cfg = cfg.get("providers", {})
    if isinstance(provider_cfg, dict):
        item = provider_cfg.get(provider_name, {})
        if isinstance(item, dict) and item.get("model"):
            return str(item["model"]).strip()
    if cfg.get("default_model"):
        return str(cfg["default_model"]).strip()
    return get_provider_preset(provider_name).default_model


def set_default_provider(provider_name: str, *, model: str | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    preset = get_provider_preset(provider_name)
    path = Path(config_path) if config_path is not None else get_provider_config_path()
    config = load_provider_config(path)
    providers = config.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    provider_cfg = providers.get(preset.name)
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}
    resolved_model = model or resolve_model(preset.name, config)
    provider_cfg["model"] = resolved_model
    providers[preset.name] = provider_cfg
    config["providers"] = providers
    config["default_provider"] = preset.name
    config["default_model"] = resolved_model
    write_provider_config(config, path)
    return {
        "ok": True,
        "config_path": str(path),
        "default_provider": preset.name,
        "default_model": resolved_model,
    }
