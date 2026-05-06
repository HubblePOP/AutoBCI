from __future__ import annotations

import io
import json
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import unittest
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AutoBciShellTests(unittest.TestCase):
    @staticmethod
    def _make_temp_repo() -> Path:
        repo_root = Path(tempfile.mkdtemp(prefix="autobci-shell-test-"))
        (repo_root / "artifacts" / "monitor").mkdir(parents=True, exist_ok=True)
        return repo_root

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _display_width(text: str) -> int:
        width = 0
        for char in text:
            if unicodedata.combining(char):
                continue
            width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        return width

    def test_build_tui_screen_defaults_to_chat_first_status_rule(self) -> None:
        from bci_autoresearch.product_shell.cli import build_tui_screen

        snapshot = {
            "campaign_id": "overnight-2026-04-20-cli",
            "stage": "formal_eval",
            "current_track_id": "feature_gru_mainline",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [
                {
                    "algorithm_label": "Feature GRU",
                    "best_val_r_label": "0.4512",
                    "best_method_display_label": "Feature GRU · 标准主线",
                    "best_promotable": True,
                }
            ],
            "program_state": {
                "program_id": "gait_phase_binary_v0",
                "version": "0.1",
                "status": "frozen",
                "task_type": "binary_classification",
                "primary_metric": "test_balanced_accuracy",
            },
            "recent_messages": [
                {
                    "message_type": "policy_decision",
                    "created_at": "2026-04-23T10:01:00Z",
                    "source_role": "guard",
                    "target_role": "director_executor",
                    "decision": "deny",
                    "reason": "raw data is read only",
                },
                {
                    "message_type": "judge_report",
                    "created_at": "2026-04-23T10:02:00Z",
                    "source_role": "judge",
                    "target_role": "intake",
                    "verdict": "pass_with_warnings",
                    "recommended_next_action": "复核数据划分",
                },
            ],
        }

        rendered = build_tui_screen(snapshot, last_message="ready")

        self.assertIn("AutoBCI  ProgramMD:frozen", rendered)
        self.assertIn("Intake: 描述你的研究问题，可以是不确定的。", rendered)
        self.assertIn("› 描述研究问题，或输入 /help", rendered)
        self.assertIn("gait_phase_binary_v0", rendered)
        self.assertIn("┊ run trail: 2 events", rendered)
        self.assertIn("1 guard deny", rendered)
        self.assertIn("judge warning", rendered)
        self.assertNotIn("ProgramMD / Amendment", rendered)
        self.assertNotIn("System Events", rendered)
        self.assertNotIn("可用命令", rendered)
        self.assertNotIn("输出", rendered)
        self.assertNotIn("policy_decision", rendered)
        self.assertNotIn("judge_report", rendered)
        self.assertNotIn("Director / 判断链", rendered)
        self.assertNotIn("Executor / 执行链", rendered)

    def test_build_intake_chat_view_model_uses_history_program_and_collapsed_trail(self) -> None:
        from bci_autoresearch.product_shell.cli import build_intake_chat_view_model

        snapshot = {
            "stage": "idle",
            "dashboard_url": "http://127.0.0.1:8878/",
            "program_state": {
                "program_id": "gait_phase_binary_v0",
                "version": "0.1",
                "status": "draft",
                "task_type": "binary_classification",
                "primary_metric": "test_balanced_accuracy",
            },
            "recent_messages": [
                {
                    "message_type": "amendment_request",
                    "created_at": "2026-04-23T10:03:00Z",
                    "source_role": "director_executor",
                    "target_role": "intake",
                    "reason": "需要修改数据划分",
                    "risk": "high",
                }
            ],
        }
        history = [
            {
                "created_at": "2026-04-23T10:00:00Z",
                "role": "user",
                "text": "我想看看步态二分类能不能做起来",
                "intent_kind": "draft_program",
            },
            {
                "created_at": "2026-04-23T10:00:01Z",
                "role": "intake",
                "text": "已生成 ProgramMD 草案，等待确认。",
                "intent_kind": "draft_program",
            },
        ]

        model = build_intake_chat_view_model(
            snapshot,
            session_history=history,
            output_history=["ready"],
            boot_mode=False,
        )

        self.assertEqual(model["program"]["status"], "draft")
        self.assertEqual(model["program"]["program_id"], "gait_phase_binary_v0")
        self.assertEqual(model["ui_phase"], "drafting_program")
        self.assertIn("ProgramMD:draft", model["status_rule"])
        self.assertEqual([item["role"] for item in model["conversation_items"]], ["user", "intake"])
        self.assertEqual(model["program_card"]["title"], "ProgramMD Draft")
        self.assertIn("Amendment Request", model["system_trail"]["expanded"][0])
        self.assertIn("1 events", model["system_trail"]["collapsed"])
        self.assertEqual(model["system_event_items"][0]["message_type"], "amendment_request")
        self.assertIn("ProgramMD:draft", model["header_text"])

    def test_cold_start_view_hides_empty_program_card_and_stale_run_trail(self) -> None:
        from bci_autoresearch.product_shell.cli import build_tui_screen

        rendered = build_tui_screen(
            {
                "program_state": {},
                "autoresearch_status": {},
                "recent_control_events": [
                    {
                        "recorded_at": "2026-04-23T10:00:00Z",
                        "action": "old_director_event",
                        "message": "stale event",
                        "ok": True,
                    }
                ],
            }
        )

        self.assertIn("Director:-", rendered)
        self.assertNotIn("ProgramMD Draft", rendered)
        self.assertNotIn("run trail", rendered)
        self.assertNotIn("stale event", rendered)

    def test_pending_program_draft_populates_program_card(self) -> None:
        from bci_autoresearch.control_plane.programs import build_gait_phase_program_draft
        from bci_autoresearch.product_shell.cli import build_intake_chat_view_model

        model = build_intake_chat_view_model(
            {"program_state": {}, "autoresearch_status": {}},
            session_history=[
                {
                    "created_at": "2026-04-24T01:00:00Z",
                    "role": "user",
                    "text": "我想看看步态二分类能不能做起来",
                    "intent_kind": "draft_program",
                }
            ],
            pending_action={
                "user_intent_kind": "draft_program",
                "program_draft": build_gait_phase_program_draft("我想看看步态二分类能不能做起来"),
            },
        )

        card = model["program_card"]
        rendered_rows = dict(card["rows"])
        self.assertEqual(rendered_rows["研究目标"], "gait_phase_binary_v0")
        self.assertEqual(rendered_rows["任务类型"], "binary_classification")
        self.assertEqual(rendered_rows["主指标"], "test_balanced_accuracy")
        self.assertEqual(card["missing"], [])

    def test_system_event_items_redact_director_scratchpad(self) -> None:
        from bci_autoresearch.product_shell.cli import build_system_event_items

        items = build_system_event_items(
            {
                "recent_messages": [
                    {
                        "message_type": "judge_report",
                        "created_at": "2026-04-23T10:04:00Z",
                        "source_role": "judge",
                        "target_role": "intake",
                        "verdict": "pass",
                        "recommended_next_action": "继续执行",
                        "director_scratchpad": "不应该出现在 TUI",
                    }
                ]
            }
        )

        rendered = json.dumps(items, ensure_ascii=False)
        self.assertIn("judge_report", rendered)
        self.assertNotIn("director_scratchpad", rendered)
        self.assertNotIn("不应该出现在 TUI", rendered)

    def test_handle_command_persists_full_intake_history(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("ProgramMD 草案", message)
        current = json.loads((repo_root / "artifacts" / "monitor" / "intake_sessions" / "current.json").read_text(encoding="utf-8"))
        rows = self._read_jsonl(Path(current["path"]))
        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertEqual(rows[0]["text"], "我想看看步态二分类能不能做起来")
        self.assertEqual(rows[0]["visibility"], "intake_only")
        self.assertEqual(rows[1]["intent_kind"], "draft_program")

    def test_greeting_starts_intake_conversation_instead_of_showing_help(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "你好",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Intake Agent", message)
        self.assertIn("描述", message)
        self.assertIn("不确定", message)
        self.assertNotIn("可用命令", message)
        current = json.loads((repo_root / "artifacts" / "monitor" / "intake_sessions" / "current.json").read_text(encoding="utf-8"))
        rows = self._read_jsonl(Path(current["path"]))
        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertEqual(rows[1]["intent_kind"], "intake_chat")
        self.assertNotIn("pending_action", session_state)

    def test_transcript_rows_deduplicate_latest_persisted_intake_reply(self) -> None:
        from bci_autoresearch.product_shell.cli import build_transcript_rows

        long_reply = (
            "我理解你要做什么：生成 ProgramMD 草案\n"
            "这会变成什么研究动作：draft_program\n"
            "命令预览：programs/gait_phase_binary_v0/program.json <= frozen ProgramMD after approve\n"
            "边界说明：ProgramMD 草案只在 approve 后冻结；不会直接启动实验。\n"
            "现在状态：等待确认。输入 approve 执行，或输入 cancel 取消。"
        )
        rows = build_transcript_rows(
            [
                {
                    "created_at": "2026-04-24T01:00:00Z",
                    "role": "user",
                    "text": "我想看看步态二分类能不能做起来",
                    "intent_kind": "draft_program",
                },
                {
                    "created_at": "2026-04-24T01:00:01Z",
                    "role": "intake",
                    "text": long_reply,
                    "intent_kind": "draft_program",
                },
            ],
            output_history=[long_reply],
        )

        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertIn("ProgramMD 草案", rows[-1]["text"])

    def test_transcript_rows_show_inflight_user_message_and_thinking_state(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_transcript_rows

        inflight = build_inflight_turn("你好", turn_id="turn-test")
        rows = build_transcript_rows([], inflight_turn=inflight)

        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertEqual(rows[0]["text"], "你好")
        self.assertEqual(rows[0]["intent_kind"], "inflight_user")
        self.assertIn("Intake 正在思考", rows[1]["text"])
        self.assertEqual(rows[1]["intent_kind"], "inflight")

    def test_transcript_rows_do_not_duplicate_inflight_user_after_persist(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_transcript_rows

        inflight = build_inflight_turn("你好", turn_id="turn-test")
        rows = build_transcript_rows(
            [
                {
                    "created_at": "2026-04-24T01:00:00Z",
                    "role": "user",
                    "text": "你好",
                    "intent_kind": "pending",
                }
            ],
            inflight_turn=inflight,
        )

        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertEqual(sum(1 for row in rows if row["role"] == "user"), 1)
        self.assertIn("Intake 正在思考", rows[-1]["text"])

    def test_status_rule_shows_intake_thinking_for_inflight_turn(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_status_rule_model

        status = build_status_rule_model(
            {"experiment_state": {"title": "测试项目", "status": "active", "session_id": "intake-a"}},
            {"status": "not_started"},
            "idle",
            inflight_turn=build_inflight_turn("你好", turn_id="turn-test"),
        )

        self.assertIn("Intake:thinking", status)

    def test_build_tui_screen_renders_inflight_turn_before_agent_reply(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_tui_screen

        rendered = build_tui_screen(
            {"program_state": {}, "autoresearch_status": {}},
            inflight_turn=build_inflight_turn("你好", turn_id="turn-test"),
        )

        self.assertIn("› 你好", rendered)
        self.assertIn("Intake 正在思考", rendered)

    def test_model_intake_agent_selects_reply_tool_for_greeting(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        agent_output = {
            "tool_name": "reply",
            "message": "你好，我会先帮你把模糊想法整理成任务契约。",
            "normalized_request": "你好",
            "reason": "寒暄，不需要执行工具。",
        }

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn", return_value=agent_output) as agent,
        ):
            should_quit, message = handle_command(
                "你好",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("任务契约", message)
        self.assertNotIn("可用命令", message)
        agent.assert_called_once()
        current = json.loads((repo_root / "artifacts" / "monitor" / "intake_sessions" / "current.json").read_text(encoding="utf-8"))
        rows = self._read_jsonl(Path(current["path"]))
        self.assertEqual(rows[1]["intent_kind"], "intake_chat")

    def test_model_intake_agent_selects_draft_program_tool(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        agent_output = {
            "tool_name": "draft_program",
            "message": "这已经足够形成步态二分类 ProgramMD 草案。",
            "normalized_request": "我想看看步态二分类能不能做起来",
            "reason": "用户给出了明确研究任务。",
        }

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn", return_value=agent_output),
        ):
            should_quit, message = handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("等待确认", message)
        self.assertEqual(session_state["pending_action"]["user_intent_kind"], "draft_program")

    def test_intake_agent_schema_exposes_autoresearch_toolbox_names(self) -> None:
        from bci_autoresearch.product_shell.cli import _intake_agent_output_schema

        schema = _intake_agent_output_schema()
        enum_values = schema["properties"]["tool_name"]["enum"]

        self.assertIn("plan_autoresearch", enum_values)
        self.assertIn("run_bare_probe", enum_values)
        self.assertNotIn("run_smoke", enum_values)

    def test_model_intake_agent_selects_plan_autoresearch_tool(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        agent_output = {
            "tool_name": "plan_autoresearch",
            "message": "我会先用 AutoResearch 方法论规划 assisted/bare 对照，不会直接启动实验。",
            "normalized_request": "用 AutoResearch 方法论规划步态二分类",
            "reason": "用户要求制定研究计划。",
        }

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn", return_value=agent_output),
        ):
            should_quit, message = handle_command(
                "用 AutoResearch 方法论规划步态二分类",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("AutoResearch 方法论", message)
        self.assertIn("不会直接启动实验", message)
        self.assertNotIn("pending_action", session_state)

    def test_intake_agent_uses_native_runtime_json_task(self) -> None:
        from bci_autoresearch.product_shell import cli

        calls: list[dict[str, object]] = []

        def fake_json_task(payload: dict[str, object], *, repo_root: Path | str | None = None) -> dict[str, object]:
            calls.append({"payload": payload, "repo_root": repo_root})
            return {
                "ok": True,
                "json": {
                    "tool_name": "reply",
                    "message": "你好，我先帮你梳理任务契约。",
                    "normalized_request": "你好",
                    "reason": "寒暄。",
                },
            }

        fake_runtime = types.SimpleNamespace(run_json_task=fake_json_task)
        with (
            patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": fake_runtime}),
            patch("bci_autoresearch.product_shell.cli.subprocess.run") as subprocess_run,
        ):
            repo_root = self._make_temp_repo()
            output = cli.run_codex_intake_agent_turn(
                "你好",
                {"program_state": {}, "autoresearch_status": {}},
                repo_root=repo_root,
                timeout_seconds=3,
            )

        self.assertEqual(output["tool_name"], "reply")
        payload = calls[0]["payload"]
        self.assertEqual(payload["model"], "gpt-5.4-mini")
        self.assertIn("JSON schema", payload["prompt"])
        self.assertEqual(Path(str(calls[0]["repo_root"])), repo_root)
        subprocess_run.assert_not_called()

    def test_provider_commands_use_provider_module(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        fake_providers = types.SimpleNamespace(
            provider_list=lambda: {
                "ok": True,
                "default_provider": "codex",
                "providers": [
                    {"name": "codex", "ready": True},
                    {"name": "kimi", "ready": False},
                ],
            },
            provider_test=lambda name: {"name": name, "ok": True, "message": "连接正常"},
            provider_set=lambda name: {"name": name, "ok": True, "message": "已设为默认 provider"},
        )

        with patch.dict(sys.modules, {"bci_autoresearch.providers": fake_providers}):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "provider", "list"]), 0)
            listed = out.getvalue()
            self.assertIn("codex", listed)
            self.assertIn("当前", listed)

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "provider", "test", "codex"]), 0)
            self.assertIn("连接正常", out.getvalue())

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "provider", "set", "kimi"]), 0)
            self.assertIn("已设为默认 provider", out.getvalue())

    def test_doctor_json_reports_core_checks(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        (repo_root / "src").mkdir(parents=True, exist_ok=True)
        fake_providers = types.SimpleNamespace(get_provider_config_status=lambda: {"ok": True, "current": "codex"})

        with (
            patch.dict(sys.modules, {"bci_autoresearch.providers": fake_providers}),
            patch("bci_autoresearch.product_shell.cli.is_dashboard_running", return_value=True),
            patch("bci_autoresearch.product_shell.cli.shutil.which", return_value="/usr/bin/tool"),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "doctor", "--json"]), 0)

        payload = json.loads(out.getvalue())
        self.assertEqual(payload["repo_root"]["path"], str(repo_root.resolve()))
        self.assertTrue(payload["python"]["ok"])
        self.assertTrue(payload["node"]["ok"])
        self.assertTrue(payload["npm"]["ok"])
        self.assertTrue(payload["provider_config"]["ok"])
        self.assertTrue(payload["dashboard_port"]["ok"])
        self.assertIn("windows_readiness", payload)

    def test_windows_doctor_command_prints_readiness(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        with (
            patch("bci_autoresearch.product_shell.cli.is_dashboard_running", return_value=False),
            patch("bci_autoresearch.product_shell.cli.shutil.which", return_value="/usr/bin/tool"),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "windows", "doctor"]), 0)

        text = out.getvalue()
        self.assertIn("Windows readiness", text)
        self.assertIn("session_cache", text)

    def test_model_intake_agent_timeout_falls_back_to_local_tool_choice(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch(
                "bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn",
                side_effect=TimeoutError("model timed out"),
            ),
        ):
            should_quit, message = handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("ProgramMD 草案", message)
        self.assertIn("等待确认", message)
        self.assertNotIn("timed out", message)
        self.assertEqual(session_state["pending_action"]["user_intent_kind"], "draft_program")

    def test_model_intake_agent_invalid_json_task_falls_back_to_local_tool_choice(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch(
                "bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn",
                return_value={"proposal": {"hypothesis": "not an intake tool"}},
            ),
        ):
            should_quit, message = handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("ProgramMD 草案", message)
        self.assertEqual(session_state["pending_action"]["user_intent_kind"], "draft_program")

    def test_model_timeout_status_question_answers_new_old_experiment_boundary(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {
            "program_state": {},
            "autoresearch_status": {
                "campaign_id": "old-joint-angle-campaign",
                "stage": "done",
                "active_track_id": "relative_origin_xyz",
                "frozen_baseline": {"dataset_name": "walk_matched_v1_64clean_joints"},
            },
        }

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch(
                "bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn",
                side_effect=TimeoutError("model timed out"),
            ),
        ):
            should_quit, message = handle_command(
                "现在 AutoResearch 从零开始了吗？",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("旧 AutoResearch", message)
        self.assertIn("当前 Intake 实验", message)
        self.assertIn("bare run", message)
        self.assertIn("old-joint-angle-campaign", message)
        self.assertNotIn("timed out", message)
        self.assertNotIn("codex exec", message)

    def test_shell_commands_do_not_pollute_intake_transcript(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("bci_autoresearch.product_shell.cli.format_status_summary", return_value="当前状态：idle"),
        ):
            should_quit, message = handle_command(
                "/status",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("当前状态", message)
        sessions_dir = repo_root / "artifacts" / "monitor" / "intake_sessions"
        current = json.loads((sessions_dir / "current.json").read_text(encoding="utf-8"))
        self.assertEqual(self._read_jsonl(Path(current["path"])), [])
        history_files = list(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else []
        self.assertEqual(history_files, [])

    def test_new_command_rotates_intake_session_without_carrying_old_history(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command, read_current_intake_history
        from bci_autoresearch.control_plane import get_control_plane_paths

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            first_current = json.loads((repo_root / "artifacts" / "monitor" / "intake_sessions" / "current.json").read_text(encoding="utf-8"))
            should_quit, message = handle_command(
                "/new",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        second_current = json.loads((repo_root / "artifacts" / "monitor" / "intake_sessions" / "current.json").read_text(encoding="utf-8"))
        self.assertFalse(should_quit)
        self.assertIn("新的 Intake 会话", message)
        self.assertNotEqual(first_current["session_id"], second_current["session_id"])
        self.assertEqual(read_current_intake_history(get_control_plane_paths(repo_root)), [])

    def test_new_command_creates_experiment_workspace_without_carrying_pending_action(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            first_current = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
            first_manifest_path = Path(first_current["path"])
            first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(first_manifest["status"], "active")
            self.assertEqual(first_manifest["pending_action"]["user_intent_kind"], "draft_program")

            should_quit, message = handle_command(
                "/new",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        second_current = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
        first_manifest_after = json.loads(first_manifest_path.read_text(encoding="utf-8"))
        second_manifest = json.loads(Path(second_current["path"]).read_text(encoding="utf-8"))

        self.assertFalse(should_quit)
        self.assertIn("新的实验工作区", message)
        self.assertNotEqual(first_current["experiment_id"], second_current["experiment_id"])
        self.assertEqual(first_manifest_after["status"], "archived")
        self.assertEqual(first_manifest_after["pending_action"]["user_intent_kind"], "draft_program")
        self.assertEqual(second_manifest["status"], "active")
        self.assertIsNone(second_manifest.get("pending_action"))
        self.assertNotIn("pending_action", session_state)

    def test_archive_and_resume_restore_pending_action_and_history(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.cli import handle_command, read_current_intake_history

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            first_current = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
            first_experiment_id = first_current["experiment_id"]

            should_quit, archive_message = handle_command(
                "/archive",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            current_after_archive = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
            archived_manifest = json.loads(Path(first_current["path"]).read_text(encoding="utf-8"))
            new_manifest = json.loads(Path(current_after_archive["path"]).read_text(encoding="utf-8"))

            self.assertFalse(should_quit)
            self.assertIn("已归档实验工作区", archive_message)
            self.assertEqual(archived_manifest["status"], "archived")
            self.assertEqual(new_manifest["status"], "active")
            self.assertIsNone(new_manifest.get("pending_action"))
            self.assertNotIn("pending_action", session_state)

            should_quit, resume_message = handle_command(
                f"/resume {first_experiment_id}",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已恢复实验工作区", resume_message)
        self.assertEqual(session_state["pending_action"]["user_intent_kind"], "draft_program")
        restored_rows = read_current_intake_history(get_control_plane_paths(repo_root))
        self.assertEqual(restored_rows[0]["text"], "我想看看步态二分类能不能做起来")

    def test_clear_archives_current_experiment_without_deleting_artifacts(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        artifact = repo_root / "artifacts" / "monitor" / "autoresearch_runs" / "run-1" / "result.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"ok": true}', encoding="utf-8")
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            first_current = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
            should_quit, message = handle_command(
                "/clear",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        first_manifest = json.loads(Path(first_current["path"]).read_text(encoding="utf-8"))
        self.assertFalse(should_quit)
        self.assertIn("已归档当前实验，并开始新的实验工作区", message)
        self.assertEqual(first_manifest["status"], "archived")
        self.assertTrue(artifact.exists())

    def test_experiments_command_lists_workspaces(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            handle_command(
                "/archive",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "/experiments",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("实验工作区列表", message)
        self.assertIn("archived", message)
        self.assertIn("active", message)

    def test_tui_status_rule_includes_experiment_state(self) -> None:
        from bci_autoresearch.product_shell.cli import build_tui_screen

        rendered = build_tui_screen(
            {
                "experiment_state": {
                    "experiment_id": "exp-gait-001",
                    "title": "步态二分类",
                    "status": "active",
                },
                "program_state": {"status": "draft", "program_id": "gait_phase_binary_v0"},
            }
        )

        self.assertIn("Project:步态二分类 · active", rendered)
        self.assertIn("ProgramMD:draft", rendered)

    def test_projects_alias_lists_lifecycle_projects(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "/projects",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("项目列表", message)
        self.assertIn("active", message)

    def test_snapshot_and_fork_commands_use_lifecycle_store(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, snapshot_message = handle_command(
                "/snapshot",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            snapshot_id = snapshot_message.split("：", 1)[1].split("。", 1)[0]
            should_quit, fork_message = handle_command(
                f"/fork {snapshot_id}",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已保存快照", snapshot_message)
        self.assertIn("已从快照分支", fork_message)
        self.assertNotIn("pending_action", session_state)

    def test_continue_and_reset_current_run_commands_do_not_create_new_project(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "我想看看步态二分类能不能做起来",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            first_project_id = session_state["experiment_id"]
            should_quit, continue_message = handle_command(
                "/continue",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertEqual(session_state["experiment_id"], first_project_id)
            should_quit, reset_message = handle_command(
                "/reset current run",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("继续当前项目", continue_message)
        self.assertIn("已重置当前 run", reset_message)
        self.assertEqual(session_state["experiment_id"], first_project_id)
        self.assertNotIn("pending_action", session_state)

    def test_status_rule_uses_project_session_program_run_lifecycle_labels(self) -> None:
        from bci_autoresearch.product_shell.cli import build_tui_screen

        rendered = build_tui_screen(
            {
                "experiment_state": {
                    "project_id": "proj-gait-001",
                    "title": "步态二分类",
                    "status": "active",
                    "session_id": "intake-001",
                    "program_id": "gait_phase_binary_v0",
                    "program_status": "draft",
                    "active_run_id": "run-001",
                },
                "program_state": {"status": "draft", "program_id": "gait_phase_binary_v0"},
            }
        )

        self.assertIn("Project:步态二分类 · active", rendered)
        self.assertIn("Session:intake-001", rendered)
        self.assertIn("ProgramMD:draft", rendered)
        self.assertIn("Run:run-001", rendered)

    def test_build_tui_screen_expands_events_only_when_requested(self) -> None:
        from bci_autoresearch.product_shell.cli import build_tui_screen

        snapshot = {
            "program_state": {"status": "frozen", "program_id": "gait_phase_binary_v0"},
            "recent_messages": [
                {
                    "message_type": "policy_decision",
                    "created_at": "2026-04-23T10:01:00Z",
                    "decision": "deny",
                    "reason": "raw data is read only",
                }
            ],
        }

        collapsed = build_tui_screen(snapshot)
        expanded = build_tui_screen(snapshot, show_events=True)

        self.assertIn("1 guard deny", collapsed)
        self.assertNotIn("policy_decision", collapsed)
        self.assertIn("policy_decision", expanded)
        self.assertIn("raw data is read only", expanded)

    def test_build_tui_screen_keeps_program_card_width_aligned_for_chinese_content(self) -> None:
        from bci_autoresearch.product_shell.cli import build_tui_screen

        snapshot = {
            "campaign_id": "director-1776218391",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "program_state": {
                "status": "draft",
                "program_id": "gait_phase_binary_v0",
                "task_type": "binary_classification",
                "primary_metric": "test_balanced_accuracy",
            },
            "algorithm_family_bests": [
                {
                    "algorithm_label": "Feature GRU",
                    "best_val_r_label": "-",
                    "best_method_display_label": "Feature GRU · 相对 RSCA 三方向坐标",
                    "best_promotable": True,
                }
            ],
        }

        rendered = build_tui_screen(snapshot, last_message="")
        panel_lines = [
            line for line in rendered.splitlines()
            if line.startswith(("┌", "├", "│", "└"))
        ]

        self.assertTrue(panel_lines)
        expected = self._display_width(panel_lines[0])
        for line in panel_lines:
            self.assertEqual(
                self._display_width(line),
                expected,
                msg=f"panel line display width drifted: {line!r}",
            )

    def test_build_rich_startup_screen_contains_boot_copy(self) -> None:
        from rich.console import Console
        from rich.layout import Layout

        from bci_autoresearch.product_shell.cli import build_rich_startup_screen

        renderable = build_rich_startup_screen()
        self.assertIsInstance(renderable, Layout)

        console = Console(record=True, force_terminal=True, width=100, color_system=None)
        console.print(renderable)
        rendered = console.export_text()

        self.assertIn("AutoBCI", rendered)
        self.assertIn("ProgramMD:not_started", rendered)
        self.assertIn("描述你的研究问题，可以是不确定的", rendered)
        self.assertIn("› 描述研究问题，或输入 /help", rendered)
        self.assertNotIn("ProgramMD / Amendment", rendered)
        self.assertNotIn("System Events", rendered)
        self.assertNotIn("输出", rendered)
        self.assertNotIn("AUTORESEARCH SHELL", rendered)
        self.assertNotIn("Director online", rendered)

    def test_build_rich_main_screen_contains_chat_first_surface(self) -> None:
        from rich.console import Console
        from rich.layout import Layout

        from bci_autoresearch.product_shell.cli import build_rich_main_screen

        snapshot = {
            "campaign_id": "director-1776218391",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [
                {
                    "algorithm_label": "Feature GRU",
                    "best_val_r_label": "0.4512",
                    "best_method_display_label": "Feature GRU · 相对 RSCA 三方向坐标",
                    "best_promotable": True,
                }
            ],
        }

        renderable = build_rich_main_screen(snapshot, last_message="dashboard 已打开：http://127.0.0.1:8878/")
        self.assertIsInstance(renderable, Layout)

        console = Console(record=True, force_terminal=True, width=120, color_system=None)
        console.print(renderable)
        rendered = console.export_text()

        self.assertIn("AutoBCI", rendered)
        self.assertIn("ProgramMD", rendered)
        self.assertIn("描述你的研究问题，可以是不确定的", rendered)
        self.assertIn("dashboard 已打开", rendered)
        self.assertIn("› 描述研究问题，或输入 /help", rendered)
        self.assertNotIn("ProgramMD / Amendment", rendered)
        self.assertNotIn("System Events", rendered)
        self.assertNotIn("输出", rendered)
        self.assertNotIn("program show", rendered)
        self.assertNotIn("Director / 判断链", rendered)
        self.assertNotIn("Executor / 执行链", rendered)

    def test_startup_and_ready_screens_share_same_root_layout_regions(self) -> None:
        from rich.layout import Layout

        from bci_autoresearch.product_shell.cli import build_rich_main_screen, build_rich_startup_screen

        snapshot = {
            "campaign_id": "director-1776218391",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
        }

        startup = build_rich_startup_screen()
        ready = build_rich_main_screen(snapshot, last_message="ready")

        self.assertIsInstance(startup, Layout)
        self.assertIsInstance(ready, Layout)
        self.assertEqual([child.name for child in startup.children], ["header", "transcript", "composer"])
        self.assertEqual([child.name for child in ready.children], ["header", "transcript", "composer"])

    def test_build_rich_main_screen_shows_active_prompt_and_echoes_last_command(self) -> None:
        from rich.console import Console

        from bci_autoresearch.product_shell.cli import build_rich_main_screen

        snapshot = {
            "campaign_id": "director-1776218391",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [
                {
                    "algorithm_label": "Feature GRU",
                    "best_val_r_label": "0.4512",
                    "best_method_display_label": "Feature GRU · 相对 RSCA 三方向坐标",
                    "best_promotable": True,
                }
            ],
        }

        console = Console(record=True, force_terminal=True, width=120, color_system=None)
        console.print(
            build_rich_main_screen(
                snapshot,
                last_message="可用命令：status | dashboard | report latest | help | quit",
                last_command="help",
            )
        )
        rendered = console.export_text()

        self.assertIn("› help", rendered)
        self.assertIn("可用命令：status | dashboard | report latest", rendered)
        self.assertIn("help", rendered)
        self.assertIn("quit", rendered)

    def test_build_rich_main_screen_keeps_layout_stable_under_narrow_width(self) -> None:
        from rich.console import Console

        from bci_autoresearch.product_shell.cli import build_rich_main_screen

        snapshot = {
            "campaign_id": "director-1776218391-with-extra-runtime-suffix-for-layout-check",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay_with_long_candidate_suffix",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [
                {
                    "algorithm_label": "Feature GRU",
                    "best_val_r_label": "0.4512",
                    "best_method_display_label": "Feature GRU · 相对 RSCA 三方向坐标 · 带额外很长的说明用于检查窄终端截断是否稳定",
                    "best_promotable": True,
                }
            ],
        }

        console = Console(record=True, force_terminal=True, width=88, color_system=None)
        console.print(build_rich_main_screen(snapshot, last_message="dashboard 已打开：http://127.0.0.1:8878/，现在检查窄终端布局是否会被顶坏。"))
        rendered = console.export_text()

        for line in rendered.splitlines():
            if line.strip():
                self.assertLessEqual(
                    self._display_width(line),
                    88,
                    msg=f"rich line overflowed terminal width: {line!r}",
                )

    def test_build_agent_work_items_uses_real_director_and_executor_sources(self) -> None:
        from bci_autoresearch.product_shell.cli import build_agent_work_items

        snapshot = {
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "latest_retrieval_packet": {
                "recorded_at": "2026-04-20T10:00:00Z",
                "current_problem_statement": "继续按当前关键问题推进。",
                "relevant_evidence": [{"source": "paper-1"}, {"source": "paper-2"}],
            },
            "latest_decision_packet": {
                "recorded_at": "2026-04-20T10:01:00Z",
                "research_judgment_delta": "先保持当前最可信候选在队列前排。",
                "recommended_queue": ["track_alpha", "track_beta"],
            },
            "latest_judgment_updates": [
                {
                    "recorded_at": "2026-04-20T10:02:00Z",
                    "topic_id": "canonical_mainline",
                    "reason": "track_alpha 当前更值得优先 formal。",
                    "next_recommended_action": "优先 formal：track_alpha",
                }
            ],
            "recent_control_events": [
                {
                    "recorded_at": "2026-04-20T10:03:00Z",
                    "action": "execute",
                    "ok": True,
                    "message": "开始执行当前 smoke 命令。",
                }
            ],
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "relative_origin_xyz_xgboost_replay",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py --final-eval",
            },
        }

        items = build_agent_work_items(snapshot, limit=6)

        self.assertTrue(any(item["role"] == "Director" for item in items))
        self.assertTrue(any(item["role"] == "Executor" for item in items))
        self.assertTrue(any(item["title"] == "更新推荐执行队列" for item in items))
        self.assertTrue(any(item["title"] == "当前执行" for item in items))
        self.assertTrue(any(item["source"] == "latest_judgment_updates" for item in items))

    def test_infer_active_agent_prefers_executor_when_current_command_is_running(self) -> None:
        from bci_autoresearch.product_shell.cli import build_agent_work_items, infer_active_agent

        snapshot = {
            "stage": "smoke",
            "latest_decision_packet": {
                "recorded_at": "2026-04-20T10:01:00Z",
                "research_judgment_delta": "继续保持当前候选。",
                "recommended_queue": ["track_alpha"],
            },
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "track_alpha",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
            },
        }

        items = build_agent_work_items(snapshot, limit=6)
        active = infer_active_agent(snapshot, items)

        self.assertEqual(active, "Executor")

    def test_build_shell_view_model_returns_intake_workspace_model(self) -> None:
        from bci_autoresearch.product_shell.cli import build_shell_view_model

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "latest_decision_packet": {
                "recorded_at": "2026-04-20T10:01:00Z",
                "research_judgment_delta": "继续保持当前候选。",
                "recommended_queue": ["track_alpha"],
            },
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "track_alpha",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
            },
            "recent_control_events": [
                {
                    "recorded_at": "2026-04-20T10:03:00Z",
                    "action": "execute",
                    "ok": True,
                    "message": "开始执行当前 smoke 命令。",
                }
            ],
        }

        model = build_shell_view_model(
            snapshot,
            boot_mode=False,
            output_history=["AutoBCI> help", "可用命令：status | dashboard | report latest | help | quit"],
            ui_tick=1,
        )

        self.assertIn("AutoBCI  ProgramMD:", model["header_text"])
        self.assertEqual(model["run_status"], "live")
        self.assertIn("conversation_items", model)
        self.assertIn("system_event_items", model)
        self.assertIn("program", model)
        self.assertIn("status_rule", model)
        self.assertIn("system_trail", model)
        self.assertNotIn("director_items", model)
        self.assertNotIn("executor_items", model)

    def test_thinking_label_has_fixed_width_across_ui_ticks(self) -> None:
        from bci_autoresearch.product_shell.cli import _thinking_label

        labels = {_thinking_label(True, ui_tick) for ui_tick in range(1, 7)}
        widths = {len(label) for label in labels}

        self.assertEqual(len(widths), 1)
        self.assertEqual(labels, {"  ✦ thinking..."})

    def test_agent_fragments_do_not_repeat_title_inside_pane_body(self) -> None:
        from bci_autoresearch.product_shell.cli import _pt_agent_fragments

        fragments = _pt_agent_fragments(
            [
                {
                    "time_label": "04-15 10:13",
                    "title": "当前执行",
                    "detail": "relative_origin_xyz_xgboost_replay",
                    "result": "结果：smoke",
                    "next": "下一步：等待回写",
                    "is_handoff": False,
                }
            ],
            role="Executor",
            title="Executor  ✦ thinking",
            boot_mode=False,
            active=False,
            reveal_step=0,
        )
        text = "".join(fragment for _style, fragment in fragments)

        self.assertNotIn("Executor  ✦ thinking", text)
        self.assertIn("当前执行", text)

    def test_executor_fragments_add_live_performance_lines_for_active_latest_item(self) -> None:
        from bci_autoresearch.product_shell.cli import _pt_agent_fragments

        fragments = _pt_agent_fragments(
            [
                {
                    "time_label": "04-15 10:13",
                    "title": "当前执行",
                    "detail": "relative_origin_xyz_xgboost_replay",
                    "result": "smoke",
                    "next": ".venv/bin/python scripts/train_tree_baseline.py",
                    "is_handoff": False,
                    "source": "autoresearch_status",
                }
            ],
            role="Executor",
            title="Executor  ✦ thinking...",
            boot_mode=False,
            active=True,
            reveal_step=2,
        )
        text = "".join(fragment for _style, fragment in fragments)

        self.assertNotIn("thinking", text)
        self.assertIn("当前执行", text)
        self.assertIn("结果：smoke", text)
        self.assertIn("下一步：.venv/bin/python scripts/train_tree_baseline.py", text)

    def test_build_agent_work_items_derives_multiple_executor_rows_from_active_track_state(self) -> None:
        from bci_autoresearch.product_shell.cli import build_agent_work_items

        snapshot = {
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "relative_origin_xyz_xgboost_replay",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
                "candidate": {
                    "stage": "editing",
                    "track_id": "relative_origin_xyz_xgboost_replay",
                    "next_step": "等待候选改动生成。",
                    "track_goal": "用非线性树模型检查结构信息是否更容易承接。",
                },
                "track_states": [
                    {
                        "track_id": "relative_origin_xyz_xgboost_replay",
                        "stage": "smoke",
                        "track_goal": "用非线性树模型检查结构信息是否更容易承接。",
                        "last_result_summary": "当前 smoke 已起跑，等待结果回写。",
                        "updated_at": "2026-04-20T10:03:00Z",
                    }
                ],
            },
        }

        items = build_agent_work_items(snapshot, limit=10)
        executor_items = [item for item in items if item["role"] == "Executor"]

        self.assertGreaterEqual(len(executor_items), 3)
        self.assertTrue(any(item["title"] == "当前执行" for item in executor_items))
        self.assertTrue(any(item["title"] == "候选阶段" for item in executor_items))
        self.assertTrue(any(item["title"] == "执行主线" for item in executor_items))

    def test_build_agent_work_items_adds_director_and_executor_handoff_rows(self) -> None:
        from bci_autoresearch.product_shell.cli import build_agent_work_items

        snapshot = {
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "latest_decision_packet": {
                "recorded_at": "2026-04-20T10:01:00Z",
                "research_judgment_delta": "先把当前最可信候选留在队列前排。",
                "recommended_queue": [
                    "relative_origin_xyz_xgboost_replay",
                    "feature_gru_mainline",
                ],
            },
            "latest_judgment_updates": [
                {
                    "recorded_at": "2026-04-20T10:02:00Z",
                    "topic_id": "canonical_mainline",
                    "reason": "当前 smoke 已起跑，等待结果回写。",
                    "next_recommended_action": "等待 smoke 结果回写后重新判断。",
                }
            ],
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "relative_origin_xyz_xgboost_replay",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
                "candidate": {
                    "stage": "formal_ready",
                    "track_id": "relative_origin_xyz_xgboost_replay",
                    "next_step": "等待 Director 重新排序 formal 队列。",
                    "track_goal": "检查相对坐标三方向是否更容易承接。",
                },
                "track_states": [
                    {
                        "track_id": "relative_origin_xyz_xgboost_replay",
                        "stage": "smoke",
                        "track_goal": "检查相对坐标三方向是否更容易承接。",
                        "last_result_summary": "smoke 结果已回写，等待下一轮判断。",
                        "updated_at": "2026-04-20T10:03:00Z",
                    }
                ],
            },
            "recent_control_events": [
                {
                    "recorded_at": "2026-04-20T10:03:00Z",
                    "action": "execute",
                    "ok": True,
                    "message": "开始执行当前 smoke 命令。",
                }
            ],
        }

        items = build_agent_work_items(snapshot, limit=12)

        self.assertTrue(any(item["title"] == "Director -> Executor" for item in items))
        self.assertTrue(any(item["title"] == "Executor -> Director" for item in items))

    def test_auto_refresh_interval_is_fixed_to_two_seconds(self) -> None:
        from bci_autoresearch.product_shell.cli import AUTO_REFRESH_INTERVAL_SECONDS

        self.assertEqual(AUTO_REFRESH_INTERVAL_SECONDS, 2.0)

    def test_active_reveal_interval_is_fixed_to_five_seconds(self) -> None:
        from bci_autoresearch.product_shell.cli import ACTIVE_REVEAL_INTERVAL_SECONDS

        self.assertEqual(ACTIVE_REVEAL_INTERVAL_SECONDS, 5.0)

    def test_build_shell_view_model_keeps_legacy_history_out_of_intake_conversation(self) -> None:
        from bci_autoresearch.product_shell.cli import build_shell_view_model

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "latest_decision_packet": {
                "recorded_at": "2026-04-20T10:01:00Z",
                "research_judgment_delta": "继续保持当前候选。",
                "recommended_queue": ["track_alpha"],
            },
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "track_alpha",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
            },
        }
        director_history = [
            {"role": "Director", "recorded_at": "2026-04-20T09:59:00Z", "time_label": "04-20 09:59", "title": "判断 B", "detail": "B", "result": "B", "next": "B", "source": "x", "sort_at": 0},
            {"role": "Director", "recorded_at": "2026-04-20T10:00:00Z", "time_label": "04-20 10:00", "title": "判断 A", "detail": "A", "result": "A", "next": "A", "source": "x", "sort_at": 1},
        ]
        executor_history = [
            {"role": "Executor", "recorded_at": "2026-04-20T10:02:00Z", "time_label": "04-20 10:02", "title": "候选阶段", "detail": "exec C", "result": "editing", "next": "next", "source": "x", "sort_at": 0},
            {"role": "Executor", "recorded_at": "2026-04-20T10:03:00Z", "time_label": "04-20 10:03", "title": "执行主线", "detail": "exec B", "result": "stage", "next": "next", "source": "x", "sort_at": 1},
            {"role": "Executor", "recorded_at": "2026-04-20T10:04:00Z", "time_label": "04-20 10:04", "title": "当前执行", "detail": "exec A", "result": "smoke", "next": "next", "source": "x", "sort_at": 2},
        ]

        model = build_shell_view_model(
            snapshot,
            boot_mode=False,
            output_history=["已接入当前研究态。"],
            ui_tick=1,
            director_history=director_history,
            executor_history=executor_history,
            reveal_counts={"Director": 2, "Executor": 1},
        )

        self.assertEqual(model["conversation_items"], [])
        expanded = "\n".join(model["system_trail"]["expanded"])
        self.assertIn("判断 B", expanded)
        self.assertIn("当前执行", expanded)

    def test_build_shell_view_model_preserves_legacy_history_order_in_system_trail(self) -> None:
        from bci_autoresearch.product_shell.cli import build_shell_view_model

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "latest_decision_packet": {
                "recorded_at": "2026-04-20T10:01:00Z",
                "research_judgment_delta": "继续保持当前候选。",
                "recommended_queue": ["track_alpha"],
            },
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "track_alpha",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
            },
        }
        executor_history = [
            {"role": "Executor", "recorded_at": "2026-04-20T10:02:00Z", "time_label": "04-20 10:02", "title": "候选阶段", "detail": "exec C", "result": "editing", "next": "next", "source": "x", "sort_at": 0},
            {"role": "Executor", "recorded_at": "2026-04-20T10:03:00Z", "time_label": "04-20 10:03", "title": "执行主线", "detail": "exec B", "result": "stage", "next": "next", "source": "x", "sort_at": 1},
            {"role": "Executor", "recorded_at": "2026-04-20T10:04:00Z", "time_label": "04-20 10:04", "title": "当前执行", "detail": "exec A", "result": "smoke", "next": "next", "source": "x", "sort_at": 2},
        ]

        model = build_shell_view_model(
            snapshot,
            boot_mode=False,
            output_history=["已接入当前研究态。"],
            ui_tick=1,
            director_history=[],
            executor_history=executor_history,
            reveal_counts={"Director": 1, "Executor": 2},
        )

        self.assertEqual(model["conversation_items"], [])
        expanded = "\n".join(model["system_trail"]["expanded"])
        self.assertLess(expanded.index("exec C"), expanded.index("exec B"))
        self.assertLess(expanded.index("exec B"), expanded.index("exec A"))

    def test_pt_style_declares_active_and_inactive_pane_states(self) -> None:
        from bci_autoresearch.product_shell.cli import _build_prompt_toolkit_style

        style = _build_prompt_toolkit_style()
        rules = getattr(style, "style_rules", [])

        self.assertIn(("director.frame.active", "bg:#242b35"), rules)
        self.assertIn(("director.frame.inactive", "bg:#171c23"), rules)
        self.assertIn(("executor.frame.active", "bg:#202934"), rules)
        self.assertIn(("executor.frame.inactive", "bg:#171c23"), rules)

    def test_header_fragments_include_framework_benchmark_banner(self) -> None:
        from bci_autoresearch.product_shell.cli import _pt_header_fragments

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "framework_benchmark": {
                "total_iterations": 962,
                "breakthrough_rate": 0.0112,
                "cost_per_breakthrough": 89.3,
                "diversity_index": 0.75,
                "direction_switches": 194,
                "autonomous_duration_minutes": 0,
                "iterations_per_hour": 5.5,
            },
        }

        text = "".join(fragment for _style, fragment in _pt_header_fragments(snapshot, boot_mode=False))
        self.assertIn("Framework Benchmark", text)
        self.assertIn("总迭代 962", text)
        self.assertIn("突破率 1.1%", text)
        self.assertIn("吞吐 5.5/h", text)

    def test_header_fragments_use_separate_benchmark_banner_label(self) -> None:
        from bci_autoresearch.product_shell.cli import _pt_header_fragments

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "framework_benchmark": {
                "total_iterations": 962,
                "breakthrough_rate": 0.0112,
                "cost_per_breakthrough": 89.3,
                "diversity_index": 0.75,
                "iterations_per_hour": 5.5,
            },
        }

        fragments = _pt_header_fragments(snapshot, boot_mode=False)

        self.assertIn(("class:header.banner.label", " 框架基准 / Framework Benchmark "), fragments)
        self.assertTrue(any(style == "class:header.banner.value" for style, _fragment in fragments))

    def test_dashboard_command_starts_server_and_opens_browser_when_needed(self) -> None:
        from bci_autoresearch.product_shell.cli import run_dashboard_command

        popen = Mock()
        open_browser = Mock(return_value=True)

        with patch("bci_autoresearch.product_shell.cli.is_dashboard_running", side_effect=[False, True]):
            message = run_dashboard_command(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                popen_factory=popen,
                browser_opener=open_browser,
                python_executable="/usr/bin/python3",
            )

        self.assertIn("http://127.0.0.1:8878/", message)
        popen.assert_called_once()
        open_browser.assert_called_once_with("http://127.0.0.1:8878/")

    def test_dashboard_command_reuses_running_server(self) -> None:
        from bci_autoresearch.product_shell.cli import run_dashboard_command

        popen = Mock()
        open_browser = Mock(return_value=True)

        with patch("bci_autoresearch.product_shell.cli.is_dashboard_running", return_value=True):
            message = run_dashboard_command(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                popen_factory=popen,
                browser_opener=open_browser,
                python_executable="/usr/bin/python3",
            )

        self.assertIn("dashboard 已打开", message)
        popen.assert_not_called()
        open_browser.assert_called_once_with("http://127.0.0.1:8878/")

    def test_handle_command_routes_natural_language_status_request_to_read_status(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}

        with patch("bci_autoresearch.product_shell.cli.format_status_summary", return_value="当前状态：feature_gru_mainline"):
            should_quit, message = handle_command(
                "现在在跑什么？",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("read_status", message)
        self.assertIn("直接执行", message)
        self.assertIn("当前状态：feature_gru_mainline", message)
        self.assertNotIn("pending_action", session_state)

    def test_handle_command_stages_proposal_draft_until_approved(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {
            "autoresearch_status": {
                "active_track_id": "feature_gru_mainline",
                "track_states": [
                    {
                        "track_id": "feature_gru_mainline",
                        "track_goal": "继续提升同试次纯脑电主线。",
                        "smoke_command": ".venv/bin/python scripts/train_feature_gru.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml",
                    }
                ],
            }
        }

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, first_message = handle_command(
                "把这条 feature_gru 路线换成 feature_tcn，先补两个 smoke，对照同一数据划分",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("等待确认", first_message)
        self.assertIn("draft_proposal", first_message)
        self.assertEqual((repo_root / "artifacts" / "monitor" / "topics.inbox.json").exists(), False)
        pending = session_state.get("pending_action")
        self.assertIsInstance(pending, dict)
        assert isinstance(pending, dict)
        self.assertEqual(pending.get("user_intent_kind"), "draft_proposal")

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, approve_message = handle_command(
                "approve",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已写入候选研究对象", approve_message)
        topics = json.loads((repo_root / "artifacts" / "monitor" / "topics.inbox.json").read_text(encoding="utf-8"))
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]["scope_label"], "chat_shell_proposal")
        self.assertIn("feature_tcn", topics[0]["goal"])
        self.assertNotIn("pending_action", session_state)

    def test_handle_command_routes_boundary_change_to_amendment_draft(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {
            "autoresearch_status": {
                "active_track_id": "feature_gru_mainline",
                "track_states": [
                    {
                        "track_id": "feature_gru_mainline",
                        "track_goal": "继续提升同试次纯脑电主线。",
                    }
                ],
            }
        }

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, first_message = handle_command(
                "把 canonical split 改成随机划分，再把 primary metric 换成 test r",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("等待确认", first_message)
        self.assertIn("draft_amendment", first_message)

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, approve_message = handle_command(
                "approve",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已写入 amendment 草案", approve_message)
        amendments = json.loads((repo_root / "artifacts" / "monitor" / "amendments.inbox.json").read_text(encoding="utf-8"))
        self.assertEqual(len(amendments), 1)
        self.assertEqual(amendments[0]["kind"], "program_amendment_draft")
        self.assertIn("primary metric", amendments[0]["normalized_request"])
        self.assertNotIn("pending_action", session_state)

    def test_handle_command_run_smoke_requires_confirmation_and_reuses_active_track_smoke_command(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {
            "autoresearch_status": {
                "active_track_id": "feature_tcn_mainline",
                "track_states": [
                    {
                        "track_id": "feature_tcn_mainline",
                        "track_goal": "验证 Feature TCN 主线 smoke。",
                        "smoke_command": ".venv/bin/python scripts/train_feature_tcn.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 4 --batch-size 64",
                    }
                ],
            }
        }

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, first_message = handle_command(
                "run smoke",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("等待确认", first_message)
        self.assertIn("run_smoke", first_message)
        self.assertIn("train_feature_tcn.py", first_message)

        popen = Mock()
        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("bci_autoresearch.product_shell.cli.subprocess.Popen", popen),
        ):
            should_quit, approve_message = handle_command(
                "approve",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已启动受控 smoke", approve_message)
        command = popen.call_args.args[0]
        self.assertIn("scripts/train_feature_tcn.py", command)
        self.assertNotIn("supervise", " ".join(command))
        self.assertNotIn("pending_action", session_state)

    def test_handle_command_writes_structured_trace_without_full_transcript(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {
            "autoresearch_status": {
                "active_track_id": "feature_gru_mainline",
                "track_states": [
                    {
                        "track_id": "feature_gru_mainline",
                        "track_goal": "继续提升同试次纯脑电主线。",
                    }
                ],
            }
        }

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "把这条 feature_gru 路线换成 feature_tcn，先补两个 smoke",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        rows = self._read_jsonl(repo_root / "artifacts" / "monitor" / "control_events.jsonl")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("session_id", row)
        self.assertIn("turn_id", row)
        self.assertEqual(row["user_intent_kind"], "draft_proposal")
        self.assertEqual(row["requires_confirmation"], True)
        self.assertEqual(row["confirmation_result"], "pending")
        self.assertIn("proposed_action", row)
        self.assertIn("command_preview", row)
        self.assertIn("artifact_refs", row)
        self.assertNotIn("conversation_transcript", row)

    def test_main_enters_tui_and_quit_exits_cleanly(self) -> None:
        from bci_autoresearch.product_shell.cli import main

        fake_snapshot = {
            "campaign_id": "overnight-2026-04-20-cli",
            "stage": "formal_eval",
            "current_track_id": "feature_gru_mainline",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
        }

        stdout = io.StringIO()
        inputs = iter(["quit"])

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("builtins.input", side_effect=lambda prompt="": next(inputs)),
            redirect_stdout(stdout),
        ):
            exit_code = main([])

        rendered = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("AutoBCI  ProgramMD", rendered)
        self.assertIn("描述你的研究问题，可以是不确定的", rendered)
        self.assertIn("AutoBCI 已退出。", rendered)
        self.assertNotIn("ProgramMD / Amendment", rendered)

    def test_pyproject_declares_autobci_console_script(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('autobci = "bci_autoresearch.product_shell.cli:main"', pyproject)
        self.assertIn("rich>=13.7", pyproject)
        self.assertIn("prompt_toolkit>=3.0", pyproject)

    def test_palette_stays_on_graphite_bronze_direction(self) -> None:
        from bci_autoresearch.product_shell.cli import PALETTE

        self.assertEqual(PALETTE["background"], "#101318")
        self.assertEqual(PALETTE["panel_bg"], "#181d24")
        self.assertEqual(PALETTE["panel_alt"], "#1c222b")
        self.assertEqual(PALETTE["border"], "#55606f")
        self.assertEqual(PALETTE["accent"], "#d0aa6f")
        self.assertEqual(PALETTE["success"], "#8bb8d9")

    def test_readline_enablement_is_optional(self) -> None:
        from bci_autoresearch.product_shell.cli import _maybe_enable_readline

        with patch("importlib.import_module", side_effect=ImportError):
            self.assertFalse(_maybe_enable_readline())

    def test_export_debug_renderables_writes_svg_and_html(self) -> None:
        from bci_autoresearch.product_shell.cli import export_debug_renderables

        snapshot = {
            "campaign_id": "director-1776218391",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_debug_renderables(
                snapshot=snapshot,
                output_dir=Path(tmpdir),
                last_message="ready",
                last_command="help",
                width=120,
            )

            self.assertIn("startup_svg", paths)
            self.assertIn("main_svg", paths)
            self.assertIn("startup_html", paths)
            self.assertIn("main_html", paths)
            for exported in paths.values():
                self.assertTrue(Path(exported).exists(), msg=f"missing export: {exported}")

    def test_rich_tui_enables_auto_refresh_for_terminal_resize_repaint(self) -> None:
        from bci_autoresearch.product_shell import cli

        snapshot = {
            "campaign_id": "director-1776218391",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
        }
        captured: dict[str, object] = {}

        class FakeLive:
            def __init__(self, *args, **kwargs) -> None:
                captured["kwargs"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def refresh(self) -> None:
                return None

            def update(self, *args, **kwargs) -> None:
                return None

        class FakeConsole:
            def show_cursor(self, show: bool = True) -> None:
                return None

            def input(self, prompt="") -> str:
                raise EOFError

        with (
            patch.object(cli, "Live", FakeLive),
            patch.object(cli, "Console", lambda: FakeConsole()),
            patch.object(cli, "build_status_snapshot", return_value=snapshot),
        ):
            exit_code = cli._run_rich_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                sleep_fn=lambda _seconds: None,
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(captured["kwargs"]["auto_refresh"])
        self.assertGreater(captured["kwargs"]["refresh_per_second"], 0)

    def test_requirements_include_rich(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("rich>=13.7", requirements)
        self.assertIn("prompt_toolkit>=3.0", requirements)

    def test_slash_command_completer_exposes_demo_commands(self) -> None:
        from prompt_toolkit.completion import Completer
        from prompt_toolkit.document import Document

        from bci_autoresearch.product_shell.cli import SLASH_COMMANDS, build_slash_command_completer

        required = {
            "/new clean",
            "/continue",
            "/dashboard",
            "/program show",
            "/approve",
            "/cancel",
            "/status",
            "/report latest",
            "/events",
            "/help",
            "/quit",
        }
        self.assertTrue(required.issubset(set(SLASH_COMMANDS)))

        completer = build_slash_command_completer()
        self.assertIsNotNone(completer)
        self.assertIsInstance(completer, Completer)
        root_items = {item.text for item in completer.get_completions(Document("/", cursor_position=1), Mock())}
        self.assertTrue(required.issubset(root_items))
        dashboard_items = [item.text for item in completer.get_completions(Document("/d", cursor_position=2), Mock())]
        self.assertEqual(dashboard_items, ["/dashboard"])
        new_items = {item.text for item in completer.get_completions(Document("/new", cursor_position=4), Mock())}
        self.assertIn("/new clean", new_items)
        program_completions = list(completer.get_completions(Document("/program ", cursor_position=9), Mock()))
        self.assertEqual({item.text for item in program_completions}, {"/program show"})
        self.assertEqual(program_completions[0].start_position, -9)
        plain_items = list(completer.get_completions(Document("dashboard", cursor_position=9), Mock()))
        self.assertEqual(plain_items, [])

    def test_prompt_toolkit_tui_attaches_slash_completer_to_input_area(self) -> None:
        from bci_autoresearch.product_shell import cli

        fake_snapshot = {
            "campaign_id": "overnight-2026-04-20-cli",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
            "autoresearch_status": {"updated_at": "2026-04-20T10:04:00Z"},
        }
        captured: dict[str, object] = {"thread_targets": []}
        original_text_area = cli.PTTextArea

        class FakeApplication:
            def __init__(self, *args, **kwargs) -> None:
                captured["kwargs"] = kwargs

            def run(self, pre_run=None):
                if pre_run:
                    pre_run()
                return 0

            def invalidate(self) -> None:
                return None

            def exit(self, result=0) -> None:
                return None

            @property
            def layout(self):
                return captured["kwargs"]["layout"]

        class FakeThread:
            def __init__(self, *, target, daemon=True) -> None:
                captured["thread_targets"].append(getattr(target, "__name__", "unknown"))

            def start(self) -> None:
                return None

        def capture_text_area(*args, **kwargs):
            captured["text_area_kwargs"] = kwargs
            return original_text_area(*args, **kwargs)

        with (
            patch.object(cli, "PTApplication", FakeApplication),
            patch.object(cli, "PTTextArea", capture_text_area),
            patch.object(cli, "build_status_snapshot", return_value=fake_snapshot),
            patch.object(cli, "patch_stdout"),
            patch.object(cli.threading, "Thread", FakeThread),
        ):
            exit_code = cli._run_prompt_toolkit_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                sleep_fn=lambda _seconds: None,
            )

        self.assertEqual(exit_code, 0)
        text_area_kwargs = captured["text_area_kwargs"]
        self.assertIsNotNone(text_area_kwargs["completer"])
        self.assertTrue(text_area_kwargs["complete_while_typing"])

    def test_prompt_toolkit_style_includes_global_background_for_padding(self) -> None:
        from bci_autoresearch.product_shell.cli import _build_prompt_toolkit_style, PALETTE

        style = _build_prompt_toolkit_style()
        self.assertIsNotNone(style)
        rules = getattr(style, "style_rules", [])
        self.assertIn(("app", f"bg:{PALETTE['background']} {PALETTE['text']}"), rules)

    def test_run_tui_prefers_prompt_toolkit_path_when_available(self) -> None:
        from bci_autoresearch.product_shell import cli

        with (
            patch.object(cli, "PROMPT_TOOLKIT_AVAILABLE", True),
            patch.object(cli.sys.stdout, "isatty", return_value=True),
            patch.object(cli, "_run_prompt_toolkit_tui", return_value=7) as pt_runner,
            patch.object(cli, "_run_rich_tui", return_value=8) as rich_runner,
        ):
            exit_code = cli.run_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
            )

        self.assertEqual(exit_code, 7)
        pt_runner.assert_called_once()
        rich_runner.assert_not_called()

    def test_prompt_toolkit_tui_enables_mouse_support_for_scrollable_history(self) -> None:
        from bci_autoresearch.product_shell import cli

        fake_snapshot = {
            "campaign_id": "overnight-2026-04-20-cli",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
            "autoresearch_status": {"updated_at": "2026-04-20T10:04:00Z"},
        }
        captured: dict[str, object] = {}

        class FakeApplication:
            def __init__(self, *args, **kwargs) -> None:
                captured["kwargs"] = kwargs
                captured["layout"] = kwargs.get("layout")

            def run(self, pre_run=None):
                if pre_run:
                    pre_run()
                return 0

            def invalidate(self) -> None:
                return None

            def exit(self, result=0) -> None:
                return None

            @property
            def layout(self):
                return captured["kwargs"]["layout"]

        with (
            patch.object(cli, "PTApplication", FakeApplication),
            patch.object(cli, "build_status_snapshot", return_value=fake_snapshot),
            patch.object(cli, "patch_stdout"),
        ):
            exit_code = cli._run_prompt_toolkit_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                sleep_fn=lambda _seconds: None,
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(captured["kwargs"]["mouse_support"])

    def test_terminal_runtime_profile_switches_to_ghostty_compatibility_mode(self) -> None:
        from bci_autoresearch.product_shell import cli

        with patch.dict(
            cli.os.environ,
            {
                "TERM": "xterm-ghostty",
                "TERM_PROGRAM": "ghostty",
                "GHOSTTY_RESOURCES_DIR": "/Applications/Ghostty.app/Contents/Resources",
            },
            clear=False,
        ):
            profile = cli._terminal_runtime_profile()

        self.assertTrue(profile["is_ghostty"])
        self.assertFalse(profile["mouse_support"])
        self.assertFalse(profile["animate_ui"])
        self.assertTrue(profile["defer_repaint_while_typing"])
        self.assertIsNone(profile["cursor"])

    def test_prompt_toolkit_tui_uses_ghostty_compatibility_profile(self) -> None:
        from bci_autoresearch.product_shell import cli

        fake_snapshot = {
            "campaign_id": "overnight-2026-04-20-cli",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
            "autoresearch_status": {"updated_at": "2026-04-20T10:04:00Z"},
        }
        captured: dict[str, object] = {"thread_targets": []}

        class FakeApplication:
            def __init__(self, *args, **kwargs) -> None:
                captured["kwargs"] = kwargs
                captured["layout"] = kwargs.get("layout")

            def run(self, pre_run=None):
                if pre_run:
                    pre_run()
                return 0

            def invalidate(self) -> None:
                return None

            def exit(self, result=0) -> None:
                return None

            @property
            def layout(self):
                return captured["kwargs"]["layout"]

        class FakeThread:
            def __init__(self, *, target, daemon=True) -> None:
                captured["thread_targets"].append(getattr(target, "__name__", "unknown"))
                self._target = target

            def start(self) -> None:
                return None

        with (
            patch.object(cli, "PTApplication", FakeApplication),
            patch.object(cli, "build_status_snapshot", return_value=fake_snapshot),
            patch.object(cli, "patch_stdout"),
            patch.object(cli.threading, "Thread", FakeThread),
            patch.dict(
                cli.os.environ,
                {
                    "TERM": "xterm-ghostty",
                    "TERM_PROGRAM": "ghostty",
                    "GHOSTTY_RESOURCES_DIR": "/Applications/Ghostty.app/Contents/Resources",
                },
                clear=False,
            ),
        ):
            exit_code = cli._run_prompt_toolkit_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                sleep_fn=lambda _seconds: None,
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse(captured["kwargs"]["mouse_support"])
        self.assertIsNone(captured["kwargs"]["cursor"])
        self.assertIn("_finish_boot", captured["thread_targets"])
        self.assertIn("_auto_refresh_loop", captured["thread_targets"])
        self.assertNotIn("_ui_animation_loop", captured["thread_targets"])

    def test_shell_view_model_is_stable_across_ui_ticks(self) -> None:
        from bci_autoresearch.product_shell import cli

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "autoresearch_status": {
                "updated_at": "2026-04-20T10:04:00Z",
                "stage": "smoke",
                "active_track_id": "track_alpha",
                "current_command": ".venv/bin/python scripts/train_tree_baseline.py",
            },
            "recent_control_events": [
                {
                    "recorded_at": "2026-04-20T10:03:00Z",
                    "action": "execute",
                    "ok": True,
                    "message": "开始执行当前 smoke 命令。",
                }
            ],
        }

        model_a = cli.build_shell_view_model(
            snapshot,
            boot_mode=False,
            output_history=["已接入当前研究态。"],
            ui_tick=1,
        )
        model_b = cli.build_shell_view_model(
            snapshot,
            boot_mode=False,
            output_history=["已接入当前研究态。"],
            ui_tick=2,
        )

        self.assertEqual(model_a["header_text"], model_b["header_text"])
        self.assertEqual(model_a["run_status"], model_b["run_status"])

    def test_prompt_toolkit_tui_registers_scroll_key_bindings_for_pane_history(self) -> None:
        from bci_autoresearch.product_shell import cli

        fake_snapshot = {
            "campaign_id": "overnight-2026-04-20-cli",
            "stage": "smoke",
            "current_track_id": "relative_origin_xyz_xgboost_replay",
            "dashboard_url": "http://127.0.0.1:8878/",
            "algorithm_family_bests": [],
            "autoresearch_status": {"updated_at": "2026-04-20T10:04:00Z"},
        }
        captured: dict[str, object] = {}

        class FakeApplication:
            def __init__(self, *args, **kwargs) -> None:
                captured["kwargs"] = kwargs
                captured["key_bindings"] = kwargs.get("key_bindings")

            def run(self, pre_run=None):
                if pre_run:
                    pre_run()
                return 0

            def invalidate(self) -> None:
                return None

            def exit(self, result=0) -> None:
                return None

            @property
            def layout(self):
                return captured["kwargs"]["layout"]

        with (
            patch.object(cli, "PTApplication", FakeApplication),
            patch.object(cli, "build_status_snapshot", return_value=fake_snapshot),
            patch.object(cli, "patch_stdout"),
        ):
            exit_code = cli._run_prompt_toolkit_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
                sleep_fn=lambda _seconds: None,
            )

        self.assertEqual(exit_code, 0)
        bindings = captured["key_bindings"].bindings
        keys = {tuple(binding.keys) for binding in bindings}
        self.assertIn(("<scroll-up>",), keys)
        self.assertIn(("<scroll-down>",), keys)

    def test_pane_control_mouse_scroll_invokes_callback(self) -> None:
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

        from bci_autoresearch.product_shell.cli import _PaneFormattedTextControl

        seen: list[int] = []
        control = _PaneFormattedTextControl(
            text=[("", "hello")],
            on_scroll=lambda delta: seen.append(delta),
            on_focus=lambda: None,
        )

        control.mouse_handler(
            MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.SCROLL_UP,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            )
        )
        control.mouse_handler(
            MouseEvent(
                position=Point(x=0, y=0),
                event_type=MouseEventType.SCROLL_DOWN,
                button=MouseButton.NONE,
                modifiers=frozenset(),
            )
        )

        self.assertEqual(seen, [-3, 3])

    def test_agents_points_cli_and_demo_work_to_dev_pack(self) -> None:
        agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("memory/docs/dev_pack_2026_04_20", agents_text)
        self.assertIn("CLI", agents_text)
        self.assertIn("dashboard", agents_text)
