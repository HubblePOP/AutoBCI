from __future__ import annotations

import json
import os
import stat
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
    monkeypatch.setenv("AUTOBCI_DEFAULT_PROVIDER", "xiaomi")
    monkeypatch.setenv("AUTOBCI_DEFAULT_MODEL", "mimo-v2-pro")

    assert get_provider_config_path() == config_path
    assert set(list_provider_presets()) == {"deepseek", "kimi", "glm", "minimax", "openai", "anthropic", "xiaomi"}
    with pytest.raises(ValueError):
        get_provider_preset("fake")
    openai = get_provider_preset("openai")
    assert openai.protocol == "pi"
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openai.capability_profile["coding_suitability"] == "smoke_supported"
    mimo = get_provider_preset("mimo")
    assert mimo.name == "xiaomi"
    assert mimo.protocol == "pi"
    assert mimo.api_key_env == "XIAOMI_API_KEY"
    assert mimo.default_model == "mimo-v2-pro"
    assert mimo.pi_provider == "xiaomi"

    result = set_default_provider("xiaomi", model="mimo-v2-pro", config_path=config_path)
    assert result["ok"] is True
    assert "api_key" not in json.dumps(result).lower()

    env_statuses = list_provider_statuses(config_path=config_path)
    assert env_statuses["default_model"] == "mimo-v2-pro"
    monkeypatch.delenv("AUTOBCI_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("AUTOBCI_DEFAULT_PROVIDER", raising=False)

    statuses = list_provider_statuses(config_path=config_path)
    by_name = {item["name"]: item for item in statuses["providers"]}
    assert statuses["default_provider"] == "xiaomi"
    assert statuses["default_model"] == "mimo-v2-pro"
    assert by_name["openai"]["ready"] is False
    assert by_name["openai"]["missing_api_key_env"] == "OPENAI_API_KEY"
    assert by_name["openai"]["capability_profile"]["json_schema"] is True
    assert by_name["xiaomi"]["provider_runtime"] == "pi-ai"
    assert by_name["xiaomi"]["pi_provider"] == "xiaomi"


def test_provider_test_unknown_provider_fails_and_real_missing_key_is_structured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import test_provider

    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    unknown_result = test_provider("fake", repo_root=tmp_path)
    assert unknown_result["ok"] is False
    assert unknown_result["provider"] == "fake"
    assert unknown_result["error_code"] == "provider_error"
    assert "Unknown provider" in unknown_result["message"]

    openai_result = test_provider("openai", repo_root=tmp_path)
    assert openai_result["ok"] is False
    assert openai_result["error_code"] == "missing_api_key"
    assert openai_result["missing_api_key_env"] == "OPENAI_API_KEY"
    assert "traceback" not in json.dumps(openai_result).lower()


def test_agent_model_config_uses_agent_override_without_leaking_global_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import list_provider_statuses
    from bci_autoresearch.providers.config import resolve_agent_provider_model

    config_path = tmp_path / "providers.toml"
    config_path.write_text(
        "\n".join(
            [
                'default_provider = "openai"',
                'default_model = "gpt-5.5"',
                "",
                "[providers.openai]",
                'model = "gpt-5.5"',
                "",
                "[agents.intake]",
                'provider = "minimax"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(config_path))
    monkeypatch.delenv("AUTOBCI_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("AUTOBCI_INTAKE_PROVIDER", raising=False)
    monkeypatch.delenv("AUTOBCI_INTAKE_MODEL", raising=False)

    resolved = resolve_agent_provider_model("intake")
    statuses = list_provider_statuses()
    by_name = {item["name"]: item for item in statuses["providers"]}

    assert resolved == {"agent": "intake", "provider": "minimax", "model": "MiniMax-M2.7", "live": True}
    assert statuses["default_provider"] == "openai"
    assert statuses["default_model"] == "gpt-5.5"
    assert by_name["openai"]["model"] == "gpt-5.5"
    assert by_name["minimax"]["model"] == "MiniMax-M2.7"
    assert by_name["deepseek"]["model"] == "deepseek-v4-flash"


def test_provider_secret_file_makes_provider_ready_and_stays_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import list_provider_statuses
    from bci_autoresearch.providers.config import get_provider_secrets_path, write_provider_secret

    config_path = tmp_path / "providers.toml"
    secrets_path = tmp_path / "provider_secrets.toml"
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOBCI_PROVIDER_SECRETS", str(secrets_path))
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    result = write_provider_secret("minimax", "mini-secret-should-not-leak")
    statuses = list_provider_statuses()
    by_name = {item["name"]: item for item in statuses["providers"]}

    assert result["ok"] is True
    assert get_provider_secrets_path() == secrets_path
    assert stat.S_IMODE(secrets_path.stat().st_mode) == 0o600
    assert by_name["minimax"]["ready"] is True
    assert by_name["minimax"]["missing_api_key_env"] is None
    assert "mini-secret-should-not-leak" not in json.dumps(statuses)


def test_mimo_alias_saves_xiaomi_secret_and_marks_provider_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import list_provider_statuses
    from bci_autoresearch.providers.config import write_provider_secret

    config_path = tmp_path / "providers.toml"
    secrets_path = tmp_path / "provider_secrets.toml"
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOBCI_PROVIDER_SECRETS", str(secrets_path))
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)

    result = write_provider_secret("mimo", "mimo-secret-should-not-leak")
    statuses = list_provider_statuses()
    by_name = {item["name"]: item for item in statuses["providers"]}

    assert result["provider"] == "xiaomi"
    assert by_name["xiaomi"]["ready"] is True
    assert by_name["xiaomi"]["model"] == "mimo-v2-pro"
    assert "mimo-secret-should-not-leak" not in json.dumps(statuses)


def test_pi_runtime_client_uses_secret_as_child_env_without_putting_key_in_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import test_provider
    from bci_autoresearch.providers.config import write_provider_secret

    config_path = tmp_path / "providers.toml"
    secrets_path = tmp_path / "provider_secrets.toml"
    capture_path = tmp_path / "runner_payload.json"
    runner = tmp_path / "capture_pi_runner.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, pathlib, sys",
                "payload = json.loads(sys.stdin.read())",
                "pathlib.Path(os.environ['AUTOBCI_PI_CAPTURE']).write_text(json.dumps(payload), encoding='utf-8')",
                "assert payload['provider'] == 'xiaomi'",
                "assert payload['model'] == 'mimo-v2-pro'",
                "assert 'api_key' not in json.dumps(payload).lower()",
                "assert 'mimo-secret-for-request' not in json.dumps(payload)",
                "assert os.environ.get('XIAOMI_API_KEY') == 'mimo-secret-for-request'",
                "print(json.dumps({'ok': True, 'provider': payload['provider'], 'model': payload['model'], 'json': {'ok': True, 'source': 'pi-runner'}}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOBCI_PROVIDER_SECRETS", str(secrets_path))
    monkeypatch.setenv("AUTOBCI_PI_RUNNER", str(runner))
    monkeypatch.setenv("AUTOBCI_PI_CAPTURE", str(capture_path))
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
    write_provider_secret("mimo", "mimo-secret-for-request")

    result = test_provider("mimo", model="mimo-v2-pro")
    captured = json.loads(capture_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["provider"] == "xiaomi"
    assert result["model"] == "mimo-v2-pro"
    assert result["response"] == {"ok": True, "source": "pi-runner"}
    assert "mimo-secret-for-request" not in json.dumps(captured)


def test_pi_runtime_failure_is_structured_and_does_not_change_agent_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bci_autoresearch.providers import set_agent_model, test_provider
    from bci_autoresearch.providers.config import resolve_agent_provider_model, write_provider_secret

    config_path = tmp_path / "providers.toml"
    secrets_path = tmp_path / "provider_secrets.toml"
    runner = tmp_path / "failing_pi_runner.py"
    runner.write_text(
        "#!/usr/bin/env python3\nimport sys\nsys.stderr.write('pi runner exploded\\n')\nsys.exit(17)\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(config_path))
    monkeypatch.setenv("AUTOBCI_PROVIDER_SECRETS", str(secrets_path))
    monkeypatch.setenv("AUTOBCI_PI_RUNNER", str(runner))
    write_provider_secret("mimo", "mimo-secret-for-request")
    set_agent_model("intake", "openai", model="gpt-5.5")

    result = test_provider("mimo", model="mimo-v2-pro")
    resolved = resolve_agent_provider_model("intake")

    assert result["ok"] is False
    assert result["provider"] == "xiaomi"
    assert result["error_code"] == "pi_runtime_error"
    assert "pi runner exploded" in result["message"]
    assert resolved["provider"] == "openai"


def test_pi_runtime_preserves_child_provider_error_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bci_autoresearch.providers import test_provider

    runner = tmp_path / "provider_error_pi_runner.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "print(json.dumps({'ok': False, 'error_code': 'pi_provider_error', 'message': '401 invalid api key'}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    monkeypatch.setenv("AUTOBCI_PI_RUNNER", str(runner))
    monkeypatch.setenv("XIAOMI_API_KEY", "mimo-secret-for-request")

    result = test_provider("mimo", model="mimo-v2-pro")

    assert result["ok"] is False
    assert result["provider"] == "xiaomi"
    assert result["error_code"] == "pi_provider_error"
    assert result["message"] == "401 invalid api key"
    assert "mimo-secret-for-request" not in json.dumps(result)


def test_json_task_pi_runner_writes_redacted_stable_ledgers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bci_autoresearch.agent_runtime.runtime import run_json_task

    runner = tmp_path / "json_pi_runner.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "payload = json.loads(sys.stdin.read())",
                "assert payload['provider'] == 'xiaomi'",
                "print(json.dumps({'ok': True, 'json': {'ok': True, 'source': 'pi-runner'}}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    monkeypatch.setenv("AUTOBCI_PI_RUNNER", str(runner))
    monkeypatch.setenv("XIAOMI_API_KEY", "mimo-secret-should-not-leak")
    task = {"provider": "xiaomi", "model": "mimo-v2-pro", "prompt": "Return a JSON object about strict causal work.", "schema": {"ok": "bool"}}

    result = run_json_task(task, repo_root=tmp_path)

    assert result["ok"] is True
    assert result["provider"] == "xiaomi"
    assert result["json"]["ok"] is True
    monitor = tmp_path / "artifacts" / "monitor"
    for name in ["provider_trace.jsonl", "agent_tool_ledger.jsonl"]:
        path = monitor / name
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "mimo-secret-should-not-leak" not in text
        assert "api_key" not in text.lower()
        assert json.loads(text.splitlines()[-1])["event_type"]
    sessions = list((monitor / "agent_sessions").glob("*.jsonl"))
    assert sessions
    assert json.loads(sessions[-1].read_text(encoding="utf-8").splitlines()[-1])["event_type"] == "json_task"


def test_edit_turn_pi_runner_returns_typescript_compatible_proposal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bci_autoresearch.agent_runtime.runtime import run_edit_turn

    runner = tmp_path / "edit_pi_runner.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "print(json.dumps({'ok': True, 'json': {'proposal': {'hypothesis': 'Provider runtime is wired.', 'why_this_change': 'Exercise the Pi runner path.', 'changes_summary': 'Return a typed proposal shell.', 'change_bucket': 'runtime_provider', 'track_comparison_note': 'No research track comparison is claimed.', 'files_touched': [], 'next_step': 'Run provider tests.', 'search_queries': [], 'research_evidence': []}}}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("AUTOBCI_PROVIDER_CONFIG", str(tmp_path / "providers.toml"))
    monkeypatch.setenv("AUTOBCI_PI_RUNNER", str(runner))
    monkeypatch.setenv("XIAOMI_API_KEY", "mimo-secret-for-edit")
    payload = {
        "threadId": "thread-123",
        "provider": "xiaomi",
        "model": "mimo-v2-pro",
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
    runner = tmp_path / "cli_pi_runner.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "print(json.dumps({'ok': True, 'json': {'ok': True, 'proposal': {'hypothesis': 'CLI path is wired.', 'why_this_change': 'Exercise subprocess CLI.', 'changes_summary': 'Return a typed proposal shell.', 'change_bucket': 'runtime_provider', 'track_comparison_note': 'No research track comparison is claimed.', 'files_touched': [], 'next_step': 'Run provider tests.', 'search_queries': [], 'research_evidence': []}}}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["AUTOBCI_PROVIDER_CONFIG"] = str(tmp_path / "providers.toml")
    env["AUTOBCI_ROOT"] = str(tmp_path)
    env["AUTOBCI_PI_RUNNER"] = str(runner)
    env["XIAOMI_API_KEY"] = "mimo-secret-for-cli"

    json_input = tmp_path / "json_input.json"
    json_output = tmp_path / "json_output.json"
    json_input.write_text(json.dumps({"provider": "xiaomi", "model": "mimo-v2-pro", "prompt": "json please"}), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "bci_autoresearch.agent_runtime", "json-task", "--input", str(json_input), "--output", str(json_output)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert json.loads(json_output.read_text(encoding="utf-8"))["ok"] is True

    edit_input = tmp_path / "edit_input.json"
    edit_output = tmp_path / "edit_output.json"
    edit_input.write_text(
        json.dumps({"provider": "xiaomi", "model": "mimo-v2-pro", "threadId": "cli-thread", "message": "draft"}),
        encoding="utf-8",
    )
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
            "changes_summary": "Add provider runtime support.",
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
