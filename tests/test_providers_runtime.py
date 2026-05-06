from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_provider_presets_and_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bci_autoresearch.providers import get_provider_config_path, list_provider_statuses, set_default_provider
    from bci_autoresearch.providers.presets import get_provider_preset, list_provider_presets

    config_path = tmp_path / "providers.toml"
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOBCI_DEFAULT_PROVIDER", "fake")
    monkeypatch.setenv("AUTOBCI_DEFAULT_MODEL", "fake-json-v1")

    assert get_provider_config_path() == config_path
    assert set(list_provider_presets()) == {"deepseek", "kimi", "glm", "minimax", "openai", "anthropic", "fake"}
    fake = get_provider_preset("fake")
    assert fake.protocol == "fake"
    assert fake.api_key_env is None
    assert fake.default_model == "fake-json-v1"
    assert {"chat", "json_schema", "tool_calls", "streaming", "reasoning", "context", "coding_suitability"} <= set(
        fake.capability_profile
    )
    openai = get_provider_preset("openai")
    assert openai.protocol == "openai_compatible"
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openai.capability_profile["coding_suitability"] == "smoke_supported"

    result = set_default_provider("fake", model="fake-json-v2", config_path=config_path)
    assert result["ok"] is True
    assert "api_key" not in json.dumps(result).lower()

    env_statuses = list_provider_statuses(config_path=config_path)
    assert env_statuses["default_model"] == "fake-json-v1"
    monkeypatch.delenv("AUTOBCI_DEFAULT_MODEL", raising=False)

    statuses = list_provider_statuses(config_path=config_path)
    by_name = {item["name"]: item for item in statuses["providers"]}
    assert statuses["default_provider"] == "fake"
    assert statuses["default_model"] == "fake-json-v2"
    assert by_name["fake"]["ready"] is True
    assert by_name["openai"]["ready"] is False
    assert by_name["openai"]["missing_api_key_env"] == "OPENAI_API_KEY"
    assert by_name["openai"]["capability_profile"]["json_schema"] is True


def test_provider_test_fake_succeeds_and_real_missing_key_is_structured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import test_provider

    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    fake_result = test_provider("fake", repo_root=tmp_path)
    assert fake_result["ok"] is True
    assert fake_result["provider"] == "fake"
    assert fake_result["response"]["ok"] is True

    openai_result = test_provider("openai", repo_root=tmp_path)
    assert openai_result["ok"] is False
    assert openai_result["error_code"] == "missing_api_key"
    assert openai_result["missing_api_key_env"] == "OPENAI_API_KEY"
    assert "traceback" not in json.dumps(openai_result).lower()


def test_json_task_fake_writes_redacted_stable_ledgers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bci_autoresearch.agent_runtime.runtime import run_json_task

    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-leak")
    task = {"provider": "fake", "prompt": "Return a JSON object about strict causal work.", "schema": {"ok": "bool"}}

    result = run_json_task(task, repo_root=tmp_path)

    assert result["ok"] is True
    assert result["provider"] == "fake"
    assert result["json"]["ok"] is True
    monitor = tmp_path / "artifacts" / "monitor"
    for name in ["provider_trace.jsonl", "agent_tool_ledger.jsonl"]:
        path = monitor / name
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "sk-secret-should-not-leak" not in text
        assert "api_key" not in text.lower()
        assert json.loads(text.splitlines()[-1])["event_type"]
    sessions = list((monitor / "agent_sessions").glob("*.jsonl"))
    assert sessions
    assert json.loads(sessions[-1].read_text(encoding="utf-8").splitlines()[-1])["event_type"] == "json_task"


def test_edit_turn_fake_returns_typescript_compatible_proposal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bci_autoresearch.agent_runtime.runtime import run_edit_turn

    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    payload = {
        "threadId": "thread-123",
        "provider": "fake",
        "message": "Try a small provider layer change without touching data/raw.",
    }

    result = run_edit_turn(payload, repo_root=tmp_path)

    assert result["threadId"] == "thread-123"
    assert isinstance(result["items"], list)
    proposal = result["proposal"]
    assert {
        "hypothesis",
        "why_this_change",
        "changes_summary",
        "change_bucket",
        "track_comparison_note",
        "files_touched",
        "next_step",
        "search_queries",
        "research_evidence",
    } <= set(proposal)
    assert proposal["files_touched"] == []
    assert proposal["change_bucket"] == "runtime_provider"


def test_runtime_cli_json_task_and_edit_turn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["AUTOBCI_PROVIDER_CONFIG"] = str(tmp_path / "providers.toml")
    env["AUTOBCI_ROOT"] = str(tmp_path)

    json_input = tmp_path / "json_input.json"
    json_output = tmp_path / "json_output.json"
    json_input.write_text(json.dumps({"provider": "fake", "prompt": "json please"}), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "bci_autoresearch.agent_runtime", "json-task", "--input", str(json_input), "--output", str(json_output)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert json.loads(json_output.read_text(encoding="utf-8"))["ok"] is True

    edit_input = tmp_path / "edit_input.json"
    edit_output = tmp_path / "edit_output.json"
    edit_input.write_text(json.dumps({"provider": "fake", "threadId": "cli-thread", "message": "draft"}), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "bci_autoresearch.agent_runtime", "edit-turn", "--input", str(edit_input), "--output", str(edit_output)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    edit_result = json.loads(edit_output.read_text(encoding="utf-8"))
    assert edit_result["threadId"] == "cli-thread"
    assert "proposal" in edit_result


def test_runtime_safety_checks_block_raw_alignment_split_and_out_of_scope_paths(tmp_path: Path) -> None:
    from bci_autoresearch.agent_runtime.safety import check_runtime_safety

    allowed = ["src/bci_autoresearch/providers", "src/bci_autoresearch/agent_runtime", "tests"]
    ok = check_runtime_safety(
        {
            "files_touched": ["src/bci_autoresearch/providers/client.py"],
            "changes_summary": "Add fake provider runtime support.",
        },
        allowed_dirs=allowed,
    )
    assert ok["ok"] is True

    raw = check_runtime_safety({"files_touched": ["data/raw/secret.rhd"]}, allowed_dirs=allowed)
    assert raw["ok"] is False
    assert raw["violations"][0]["code"] == "raw_data_forbidden"

    sensitive = check_runtime_safety(
        {"files_touched": ["src/bci_autoresearch/agent_runtime/safety.py"], "changes_summary": "change split and primary metric"},
        allowed_dirs=allowed,
    )
    assert sensitive["ok"] is False
    assert {item["code"] for item in sensitive["violations"]} >= {"sensitive_term", "alignment_or_metric_sensitive"}

    scope = check_runtime_safety({"files_touched": ["scripts/convert_session.py"]}, allowed_dirs=allowed)
    assert scope["ok"] is False
    assert "outside_allowed_dirs" in {item["code"] for item in scope["violations"]}
