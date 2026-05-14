from __future__ import annotations

import io
import json
import os
import sqlite3
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
    def _write_director_fixture(repo_root: Path) -> None:
        handoff_dir = repo_root / "memory" / "docs" / "dev_pack_2026_04_20" / "08_LOCAL_AGENT_HANDOFF"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        (handoff_dir / "DIRECTOR_AGENT.md").write_text(
            "# Director Agent\n\n只生成研究队列，不启动 Executor，不写正式执行 manifest。\n",
            encoding="utf-8",
        )
        (repo_root / "artifacts" / "monitor" / "rsvp_ship_image_autoresearch_latest.json").write_text(
            json.dumps(
                {
                    "run_id": "rsvp-ship-image-shell-test",
                    "program_id": "rsvp_ship_crossmodal_v0",
                    "dataset_name": "Downloads/RSVP跨模态数据",
                    "status": "completed_image_only",
                    "target_mode": "rsvp_ship_image_classification",
                    "primary_metric": "test_balanced_accuracy",
                    "benchmark_primary_score": 0.8886,
                    "test_primary_metric": 0.8696,
                    "eeg_status": "blocked_missing_eeg_or_events",
                    "selected_model": {
                        "model_id": "image_logistic_baseline",
                        "feature_view": "grayscale_32x32_flat",
                        "algorithm": "numpy weighted logistic regression",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

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
    def _make_intake_stub_runtime(calls: list[dict[str, object]] | None = None) -> types.SimpleNamespace:
        def run_json_task(payload: dict[str, object], *, repo_root: Path | str | None = None) -> dict[str, object]:
            if calls is not None:
                calls.append({"payload": payload, "repo_root": repo_root})
            prompt = str(payload.get("prompt") or "")
            user_text = prompt.rsplit("用户输入：", 1)[-1].strip() if "用户输入：" in prompt else prompt
            lowered = user_text.lower()
            if "状态" in user_text or "status" in lowered:
                tool_name = "read_status"
                message = "我先读取当前 AutoBCI 状态。"
                reason = "用户询问状态。"
            elif "program" in lowered or "任务" in user_text or "ship" in lowered or "船" in user_text:
                tool_name = "draft_program"
                message = "这足够形成纯图像 ship / not-ship 二分类 Program 草案。"
                reason = "用户描述了研究任务。"
            else:
                tool_name = "reply"
                message = "你好，我先帮你梳理任务契约。"
                reason = "寒暄。"
            return {
                "ok": True,
                "json": {
                    "tool_name": tool_name,
                    "message": message,
                    "normalized_request": user_text,
                    "reason": reason,
                },
            }

        return types.SimpleNamespace(run_json_task=run_json_task)

    @staticmethod
    def _write_frozen_crossmodal_program(repo_root: Path) -> None:
        program_dir = repo_root / "programs" / "rsvp_ship_crossmodal_v0"
        program_dir.mkdir(parents=True, exist_ok=True)
        (program_dir / "program.json").write_text(
            json.dumps(
                {
                    "program_id": "rsvp_ship_crossmodal_v0",
                    "version": "0.1",
                    "status": "frozen",
                    "research_goal": {
                        "task_type": "cross_modal_binary_classification",
                        "statement": "比较图像识别船与脑电识别船",
                        "scientific_question": "图像和脑电能否识别 ship / not-ship。",
                    },
                    "metrics": {"primary": "test_balanced_accuracy"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

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

        self.assertIn("AutoBCI  Phase:已冻结  Program:frozen", rendered)
        self.assertIn("描述你的研究任务", rendered)
        self.assertNotIn("研究计划流", rendered)
        self.assertIn("› 直接描述任务，或输入 / 查看高级命令", rendered)
        self.assertIn("gait_phase_binary_v0", rendered)
        self.assertIn("┊ run trail: 2 events", rendered)
        self.assertIn("1 guard deny", rendered)
        self.assertIn("judge warning", rendered)
        self.assertNotIn("Program / Amendment", rendered)
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
                "text": "已生成 Program 草案，等待确认。",
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
        self.assertIn("Program:draft", model["status_rule"])
        self.assertEqual([item["role"] for item in model["conversation_items"]], ["user", "intake"])
        self.assertEqual(model["program_card"]["title"], "研究计划 / Program")
        self.assertIn("Amendment Request", model["system_trail"]["expanded"][0])
        self.assertIn("1 events", model["system_trail"]["collapsed"])
        self.assertEqual(model["system_event_items"][0]["message_type"], "amendment_request")
        self.assertIn("Program:draft", model["header_text"])

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

        self.assertNotIn("Director:-", rendered)
        self.assertNotIn("Program Draft", rendered)
        self.assertNotIn("run trail", rendered)
        self.assertNotIn("stale event", rendered)

    def test_system_event_items_filters_quit_control_events(self) -> None:
        from bci_autoresearch.product_shell.cli import build_system_event_items

        items = build_system_event_items(
            {
                "recent_control_events": [
                    {
                        "recorded_at": "2026-05-12T06:00:00Z",
                        "action": "quit",
                        "message": "AutoBCI 已退出。",
                        "ok": True,
                    },
                    {
                        "recorded_at": "2026-05-12T06:01:00Z",
                        "action": "execute",
                        "message": "开始执行固定评估。",
                        "ok": True,
                    },
                ]
            }
        )

        rendered = json.dumps(items, ensure_ascii=False)
        self.assertIn("开始执行固定评估", rendered)
        self.assertNotIn("AutoBCI 已退出", rendered)
        self.assertNotIn('"quit"', rendered)

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
        self.assertIn("还没有生成 Program", message)
        current = json.loads((repo_root / "artifacts" / "monitor" / "intake_sessions" / "current.json").read_text(encoding="utf-8"))
        rows = self._read_jsonl(Path(current["path"]))
        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertEqual(rows[0]["text"], "我想看看步态二分类能不能做起来")
        self.assertEqual(rows[0]["visibility"], "intake_only")
        self.assertEqual(rows[1]["intent_kind"], "program_discussion")
        self.assertNotIn("program_draft", session_state["pending_action"])

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
        self.assertIn("研究计划助手", message)
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
            "我理解你要做什么：生成 Program 草案\n"
            "这会变成什么研究动作：draft_program\n"
            "命令预览：programs/gait_phase_binary_v0/program.json <= frozen Program after approve\n"
            "边界说明：Program 草案只在 approve 后冻结；不会直接启动实验。\n"
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
        self.assertIn("Program 草案", rows[-1]["text"])

    def test_transcript_rows_preserve_long_markdown_reply_without_preview_ellipsis(self) -> None:
        from bci_autoresearch.product_shell.cli import _pt_transcript_fragments, build_transcript_rows, build_timeline_items

        long_reply = "\n".join(
            [
                "第一段：**关键判断** 是先做纯图像 ship / not-ship 二分类。",
                "第二段：**数据来源** 是本地任务目录，不在这里改 data/raw。",
                "第三段：**指标** 是 test_balanced_accuracy，同时保留混淆矩阵。",
                "第四段：**风险** 是背景捷径、重复样本和 lucky split。",
                "第五段：**下一步** 是冻结 Program 后进入研究闭环。",
            ]
        )
        rows = build_timeline_items(
            [
                {
                    "created_at": "2026-05-12T01:00:00Z",
                    "role": "intake",
                    "text": long_reply,
                    "intent_kind": "draft_program",
                }
            ]
        )

        self.assertEqual(rows[-1]["text"], long_reply)
        self.assertNotIn("…", rows[-1]["text"])

        transcript_rows = build_transcript_rows(rows)
        fragments = _pt_transcript_fragments({"transcript_rows": transcript_rows, "system_trail": {"show_default": False}})
        rendered_text = "".join(fragment for _style, fragment in fragments)
        bold_styles = [style for style, fragment in fragments if fragment == "关键判断"]

        self.assertIn("第五段", rendered_text)
        self.assertIn("关键判断", rendered_text)
        self.assertNotIn("**", rendered_text)
        self.assertTrue(any("markdown.bold" in style for style in bold_styles))

    def test_transcript_rows_show_inflight_user_message_and_thinking_state(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_transcript_rows

        inflight = build_inflight_turn("你好", turn_id="turn-test")
        rows = build_transcript_rows([], inflight_turn=inflight)

        self.assertEqual([row["role"] for row in rows], ["user", "intake"])
        self.assertEqual(rows[0]["text"], "你好")
        self.assertEqual(rows[0]["intent_kind"], "inflight_user")
        self.assertIn("正在整理", rows[1]["text"])
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
        self.assertIn("正在整理", rows[-1]["text"])

    def test_status_rule_shows_model_thinking_for_inflight_turn(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_status_rule_model

        status = build_status_rule_model(
            {"experiment_state": {"title": "测试项目", "status": "active", "session_id": "intake-a"}},
            {"status": "not_started"},
            "idle",
            inflight_turn=build_inflight_turn("你好", turn_id="turn-test"),
        )

        self.assertIn("模型:thinking", status)
        self.assertNotIn("Intake:", status)
        self.assertNotIn("Director:", status)
        self.assertNotIn("Executor:", status)
        self.assertNotIn("Judge:", status)
        self.assertNotIn("Guard:", status)

    def test_build_tui_screen_renders_inflight_turn_before_agent_reply(self) -> None:
        from bci_autoresearch.product_shell.cli import build_inflight_turn, build_tui_screen

        rendered = build_tui_screen(
            {"program_state": {}, "autoresearch_status": {}},
            inflight_turn=build_inflight_turn("你好", turn_id="turn-test"),
        )

        self.assertIn("› 你好", rendered)
        self.assertIn("正在整理", rendered)

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

    def test_model_intake_agent_draft_tool_is_blocked_until_explicit_program_request(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        agent_output = {
            "tool_name": "draft_program",
            "message": "这已经足够形成步态二分类 Program 草案。",
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
        self.assertIn("不生成 Program", message)
        self.assertNotIn("program_draft", session_state.get("pending_action", {}))

    def test_discussion_then_explicit_program_request_generates_draft(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "我想先聊一下纯图像 ship / not-ship 二分类，只用图像不用脑电，这个方向你怎么看？",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("还没有生成 Program", message)
            self.assertNotIn("program_draft", session_state["pending_action"])

            should_quit, message = handle_command(
                "可以了，按现在聊的版本生成 Program markdown。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        pending = session_state["pending_action"]
        self.assertEqual(pending["program_draft"]["program_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(pending["program_draft"]["research_goal"]["task_type"], "image_binary_classification")

    def test_ship_rsvp_cross_modal_request_stages_program_contract_not_proposal(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = (
            "现在生成 Program：我想比较图像识别出船和用脑电识别出船，看看哪种方式效果更好。"
            "数据在 Downloads/RSVP跨模态数据，脑电可能更困难。"
        )

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        self.assertIn("图像识别船", message)
        self.assertIn("脑电识别船", message)
        pending = session_state.get("pending_action")
        self.assertIsInstance(pending, dict)
        assert isinstance(pending, dict)
        self.assertEqual(pending["user_intent_kind"], "draft_program")
        self.assertEqual(pending["program_draft"]["program_id"], "rsvp_ship_crossmodal_v0")
        self.assertEqual(
            pending["program_draft"]["data_boundary"]["dataset_name"],
            "Downloads/RSVP跨模态数据",
        )

    def test_ship_rsvp_image_only_request_stages_image_only_program_contract(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = (
            "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
            "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
            "先形成 Program，不要启动 Executor。"
        )

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        self.assertIn("纯图像", message)
        pending = session_state.get("pending_action")
        self.assertIsInstance(pending, dict)
        assert isinstance(pending, dict)
        draft = pending["program_draft"]
        self.assertEqual(draft["program_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(draft["research_goal"]["task_type"], "image_binary_classification")
        self.assertEqual(draft["data_boundary"]["dataset_name"], "Downloads/RSVP跨模态数据")
        self.assertEqual(draft["search_space"]["allowed_feature_families"], ["image_pixels_or_embeddings"])
        self.assertNotIn("eeg_balanced_accuracy", draft["metrics"]["secondary"])

    def test_data_command_saves_local_dataset_path(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        dataset = repo_root / "fixtures" / "RSVP图像数据"
        dataset.mkdir(parents=True)
        session_state: dict[str, object] = {}

        should_quit, message = handle_command(
            f"/data {dataset}",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )

        self.assertFalse(should_quit)
        self.assertIn("已保存本地数据目录", message)
        config_path = repo_root / ".autobci" / "data_paths.json"
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record = config["tasks"]["rsvp_ship_image_only_v0"]
        self.assertEqual(record["dataset_root"], str(dataset.resolve()))
        self.assertEqual(record["dataset_name"], "RSVP图像数据")

    def test_data_command_prompt_then_next_input_saves_path(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        dataset = repo_root / "fixtures" / "RSVP图像数据"
        dataset.mkdir(parents=True)
        session_state: dict[str, object] = {}

        should_quit, message = handle_command(
            "/data",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("选择本地数据目录", message)
        self.assertEqual(session_state["selection_context"]["kind"], "data_path_input")

        should_quit, message = handle_command(
            str(dataset),
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )

        self.assertFalse(should_quit)
        self.assertIn("已保存本地数据目录", message)
        self.assertNotIn("selection_context", session_state)

    def test_program_plan_uses_configured_local_dataset_path(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        dataset = repo_root / "fixtures" / "RSVP图像数据"
        dataset.mkdir(parents=True)
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        handle_command(
            f"/data {dataset}",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )

        request = (
            "现在生成 Program：做纯图像 ship / not-ship 二分类，只用刚才配置的数据目录，"
            "不使用脑电，主指标 test_balanced_accuracy。"
        )
        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        pending = session_state["pending_action"]
        draft = pending["program_draft"]
        self.assertEqual(draft["program_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(draft["data_boundary"]["dataset_root"], str(dataset.resolve()))
        self.assertEqual(draft["data_boundary"]["dataset_name"], "RSVP图像数据")
        self.assertEqual(draft["data_boundary"]["local_config_source"], ".autobci/data_paths.json")

    def test_program_show_prefers_pending_program_draft_before_approval(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {
            "program_state": {
                "program_id": "old_frozen_v0",
                "version": "0.1",
                "status": "frozen",
                "task_type": "old_task",
                "primary_metric": "old_metric",
            },
            "autoresearch_status": {},
        }
        request = (
            "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
            "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
            "先形成 Program，不要启动 Executor。"
        )

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

            self.assertFalse(should_quit)
            self.assertIn("Program 草案", message)

            should_quit, message = handle_command(
                "/program show",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("当前 Program 草案（待确认）", message)
        self.assertIn("rsvp_ship_image_only_v0", message)
        self.assertIn("image_binary_classification", message)
        self.assertIn("Downloads/RSVP跨模态数据", message)
        self.assertNotIn("old_frozen_v0", message)

    def test_plan_mode_builds_image_only_program_plan_before_approval(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = (
            "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
            "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
            "先形成 Program，不要启动 Executor。"
        )

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/plan",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("Program 起草已开启", message)
            self.assertNotIn("ProgramMD", message)
            self.assertEqual(session_state["pending_action"]["plan_mode"], True)

            should_quit, message = handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("Program 已更新", message)
            self.assertIn("Program：", message)
            self.assertNotIn("ProgramMD", message)
            self.assertIn("forbidden_actions", message)
            pending = session_state["pending_action"]
            self.assertEqual(pending["plan_mode"], True)
            self.assertEqual(pending["plan_status"], "drafting")
            self.assertEqual(pending["revision"], 1)
            self.assertEqual(pending["program_draft"]["program_id"], "rsvp_ship_image_only_v0")
            self.assertEqual(pending["program_draft"]["research_goal"]["task_type"], "image_binary_classification")

            should_quit, message = handle_command(
                "/approve",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("选择“确认并开始运行”", message)
            self.assertEqual(session_state["pending_action"]["plan_mode"], True)

            should_quit, message = handle_command(
                "/program show",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("当前 Program", message)
            self.assertNotIn("ProgramMD", message)
            self.assertIn("revision: 1", message)
            self.assertIn("rsvp_ship_image_only_v0", message)

            should_quit, message = handle_command(
                "/plan accept",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("Program 已确认", message)
            self.assertNotIn("ProgramMD", message)
            self.assertIn("行动记录", message)
            self.assertIn("Program", message)
            self.assertIn("确认 Program", message)
            self.assertEqual(session_state["pending_action"]["plan_status"], "accepted")
            self.assertFalse(session_state["pending_action"]["plan_mode"])
            self.assertTrue(session_state["pending_action"]["requires_confirmation"])

            should_quit, message = handle_command(
                "/approve",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已冻结 Program：rsvp_ship_image_only_v0", message)
        self.assertNotIn("ProgramMD", message)
        self.assertIn("行动记录", message)
        self.assertIn("冻结计划", message)
        self.assertNotIn("pending_action", session_state)
        self.assertTrue((repo_root / "programs" / "rsvp_ship_image_only_v0" / "Program.md").exists())

    def test_plain_research_request_enters_program_plan_without_plan_command(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = (
            "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
            "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
            "先形成 Program，不要启动 Executor。"
        )

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        pending = session_state["pending_action"]
        self.assertEqual(pending["plan_mode"], True)
        self.assertEqual(pending["plan_status"], "drafting")
        self.assertEqual(pending["program_draft"]["program_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(pending["program_draft"]["research_goal"]["task_type"], "image_binary_classification")

    def test_new_clean_starts_default_program_plan(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/new clean",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        current = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
        manifest = json.loads(Path(current["path"]).read_text(encoding="utf-8"))
        self.assertFalse(should_quit)
        self.assertIn("可以直接描述研究任务", message)
        self.assertEqual(session_state["pending_action"]["plan_mode"], True)
        self.assertEqual(manifest["pending_action"]["plan_status"], "drafting")

    def test_next_action_numbers_accept_freeze_then_research_gate_uses_yes_no_details(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        self._write_director_fixture(repo_root)
        session_state: dict[str, object] = {}
        request = (
            "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
            "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
            "先形成 Program，不要启动 Executor。"
        )

        should_quit, message = handle_command(
            request,
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)

        should_quit, message = handle_command(
            "1",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("已确认并冻结 Program：rsvp_ship_image_only_v0", message)
        self.assertIn("Continue?", message)
        self.assertIn("[Y] Yes", message)

        should_quit, message = handle_command(
            "d",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("Details:", message)

        should_quit, message = handle_command(
            "y",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("研究闭环已推进", message)
        self.assertIn("行动记录", message)
        self.assertIn("选择研究方向", message)
        self.assertIn("结果复核", message)
        ledger = repo_root / "artifacts" / "research_loop" / "rsvp_ship_image_only_v0" / "ledger.jsonl"
        self.assertTrue(ledger.exists())
        payload = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(payload["task_id"], "rsvp_ship_image_only_v0")
        self.assertTrue(payload["track_id"])
        self.assertTrue(payload["judgment_chain"])

    def test_programmd_next_actions_are_run_revise_or_pause(self) -> None:
        from bci_autoresearch.product_shell.cli import build_intake_chat_view_model, handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "现在生成 Program：我想做纯图像 ship / not-ship 二分类，只用图像，不用脑电。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        view = build_intake_chat_view_model(
            fake_snapshot,
            pending_action=session_state["pending_action"],
            session_history=[],
        )
        labels = [item["label"] for item in view["next_actions"]]
        commands = [item["command"] for item in view["next_actions"]]
        self.assertEqual(labels, ["确认并开始运行", "补充修改意见", "暂不执行"])
        self.assertEqual(commands, ["/plan run", "/plan revise", "/plan exit"])
        self.assertNotIn("查看完整计划", labels)
        self.assertNotIn("重置计划", labels)

    def test_plan_revise_next_message_rewrites_programmd(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "现在生成 Program：我想做纯图像 ship / not-ship 二分类，只用图像，不用脑电。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "2",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("直接输入你想改的地方", message)
            self.assertEqual(session_state["selection_context"]["kind"], "program_revision")

            old_revision = int(session_state["pending_action"]["revision"])
            should_quit, message = handle_command(
                "把成功指标补充为 test_balanced_accuracy，同时强调不要读取脑电。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已按你的修改意见更新 Program", message)
        self.assertEqual(int(session_state["pending_action"]["revision"]), old_revision + 1)
        self.assertNotIn("selection_context", session_state)

    def test_numbered_next_actions_do_not_override_secondary_menus(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        self._write_director_fixture(repo_root)
        session_state: dict[str, object] = {}
        request = "我想做纯图像 ship / not-ship 二分类，只用图像，不用脑电。"

        handle_command(request, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)

        should_quit, message = handle_command(
            "/model",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("模型设置", message)
        should_quit, message = handle_command(
            "1",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("选择计划/对话模型使用的 Provider", message)
        self.assertNotIn("计划已确认", message)
        session_state.pop("selection_context", None)

        should_quit, message = handle_command(
            "/director",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("研究方向调度", message)
        should_quit, message = handle_command(
            "1",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("已生成研究方向队列", message)
        self.assertNotIn("计划已确认", message)
        session_state.pop("selection_context", None)

        should_quit, message = handle_command(
            "/switch",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("选择要切换的任务", message)
        should_quit, message = handle_command(
            "1",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("1.1", message)
        self.assertNotIn("计划已确认", message)

    def test_plan_cancel_exit_and_reset_manage_plan_state(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command("/plan", repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
            handle_command(
                "只用图像判断图片是不是船",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "/plan exit",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("已暂停 Program 起草", message)
            self.assertEqual(session_state["pending_action"]["plan_status"], "paused")
            self.assertFalse(session_state["pending_action"]["plan_mode"])

            should_quit, message = handle_command(
                "/plan",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("已恢复 Program 起草", message)
            self.assertEqual(session_state["pending_action"]["plan_status"], "drafting")
            self.assertTrue(session_state["pending_action"]["plan_mode"])

            should_quit, message = handle_command(
                "/plan reset",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertFalse(should_quit)
            self.assertIn("已重置 Program 计划", message)
            self.assertEqual(session_state["pending_action"]["revision"], 0)
            self.assertNotIn("program_draft", session_state["pending_action"])

            should_quit, message = handle_command(
                "/plan cancel",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已取消 Program 计划", message)
        self.assertNotIn("pending_action", session_state)

    def test_tui_test_mode_disables_real_intake_model_agent(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = "现在生成 Program：我想从零开始做一个纯图像任务，只用图像判断是不是船。"

        with (
            patch.dict(os.environ, {"AUTOBCI_TUI_TEST_MODE": "1"}, clear=False),
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch(
                "bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn",
                side_effect=AssertionError("TUI test mode must not call the real intake provider"),
            ),
        ):
            should_quit, message = cli.handle_command(
                request,
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=cli.should_use_tui_model_agent(),
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        self.assertEqual(session_state["pending_action"]["program_draft"]["program_id"], "rsvp_ship_image_only_v0")

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

    def test_model_intake_image_ship_task_overrides_eeg_drift(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        drifting_agent_output = {
            "tool_name": "reply",
            "message": "这是图像二分类任务。请说明数据来源：公开数据集还是自己采集脑电数据？",
            "normalized_request": "图像任务，分辨船只和非船",
            "reason": "用户描述了任务但数据来源不清。",
        }

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch("bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn", return_value=drifting_agent_output),
        ):
            should_quit, message = handle_command(
                "现在生成 Program：我们现在想要做一个图像任务，是分别船只和非船的。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
                use_model_agent=True,
            )

        self.assertFalse(should_quit)
        self.assertIn("Program 已更新", message)
        self.assertIn("rsvp_ship_image_only_v0", message)
        self.assertEqual(session_state["pending_action"]["program_draft"]["program_id"], "rsvp_ship_image_only_v0")
        self.assertNotIn("自己采集脑电数据", message)

    def test_intake_agent_uses_native_runtime_json_task(self) -> None:
        from bci_autoresearch.product_shell import cli

        calls: list[dict[str, object]] = []
        stub_runtime = self._make_intake_stub_runtime(calls)
        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[agents.intake]",
                    'provider = "openai"',
                    'model = "gpt-5.5"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with (
            patch.dict(os.environ, {"AUTOBCI_PROVIDER_CONFIG": str(config_path)}, clear=False),
            patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": stub_runtime}),
            patch("bci_autoresearch.product_shell.cli.subprocess.run") as subprocess_run,
        ):
            for key in [
                "AUTOBCI_INTAKE_PROVIDER",
                "AUTOBCI_INTAKE_MODEL",
                "AUTOBCI_DEFAULT_PROVIDER",
                "AUTOBCI_DEFAULT_MODEL",
            ]:
                os.environ.pop(key, None)
            output = cli.run_codex_intake_agent_turn(
                "你好",
                {"program_state": {}, "autoresearch_status": {}},
                repo_root=repo_root,
                timeout_seconds=3,
            )

        self.assertEqual(output["tool_name"], "reply")
        payload = calls[0]["payload"]
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertIn("JSON schema", payload["prompt"])
        self.assertIn("必须选择对应工具", payload["prompt"])
        self.assertEqual(Path(str(calls[0]["repo_root"])), repo_root)
        subprocess_run.assert_not_called()

    def test_intake_agent_without_model_config_fails_loudly(self) -> None:
        from bci_autoresearch.product_shell import cli

        calls: list[dict[str, object]] = []
        repo_root = self._make_temp_repo()
        cleared_env: dict[str, str] = {}
        env_keys = [
            "AUTOBCI_INTAKE_PROVIDER",
            "AUTOBCI_INTAKE_MODEL",
            "AUTOBCI_DEFAULT_PROVIDER",
            "AUTOBCI_DEFAULT_MODEL",
        ]
        try:
            for key in env_keys:
                if key in os.environ:
                    cleared_env[key] = os.environ.pop(key)
            os.environ["AUTOBCI_PROVIDER_CONFIG"] = str(repo_root / "missing-providers.toml")
            with patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": self._make_intake_stub_runtime(calls)}):
                with self.assertRaisesRegex(RuntimeError, "模型未配置"):
                    cli.run_codex_intake_agent_turn(
                        "你好",
                        {"program_state": {}, "autoresearch_status": {}},
                        repo_root=repo_root,
                        timeout_seconds=3,
                    )
        finally:
            os.environ.pop("AUTOBCI_PROVIDER_CONFIG", None)
            for key, value in cleared_env.items():
                os.environ[key] = value

        self.assertEqual(calls, [])

    def test_intake_agent_uses_provider_config_when_intake_env_is_unset(self) -> None:
        from bci_autoresearch.product_shell import cli

        calls: list[dict[str, object]] = []

        def fake_json_task(payload: dict[str, object], *, repo_root: Path | str | None = None) -> dict[str, object]:
            calls.append({"payload": payload, "repo_root": repo_root})
            return {
                "ok": True,
                "json": {
                    "tool_name": "reply",
                    "message": "我会用配置里的 GPT-5.5 做 Intake。",
                    "normalized_request": "你好",
                    "reason": "配置读取测试。",
                },
            }

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        config_path.write_text(
            '\n'.join(
                [
                    'default_provider = "openai"',
                    'default_model = "gpt-5.5"',
                    "",
                    "[providers.openai]",
                    'model = "gpt-5.5"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        fake_runtime = types.SimpleNamespace(run_json_task=fake_json_task)
        cleared_env: dict[str, str] = {}
        env_keys = [
            "AUTOBCI_INTAKE_PROVIDER",
            "AUTOBCI_INTAKE_MODEL",
            "AUTOBCI_DEFAULT_PROVIDER",
            "AUTOBCI_DEFAULT_MODEL",
        ]
        try:
            for key in env_keys:
                if key in os.environ:
                    cleared_env[key] = os.environ.pop(key)
            os.environ["AUTOBCI_PROVIDER_CONFIG"] = str(config_path)
            with patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": fake_runtime}):
                cli.run_codex_intake_agent_turn(
                    "你好",
                    {"program_state": {}, "autoresearch_status": {}},
                    repo_root=repo_root,
                    timeout_seconds=3,
                )
        finally:
            os.environ.pop("AUTOBCI_PROVIDER_CONFIG", None)
            for key, value in cleared_env.items():
                os.environ[key] = value

        payload = calls[0]["payload"]
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["model"], "gpt-5.5")

    def test_configured_mimo_intake_provider_exercises_common_json_contract(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[agents.intake]",
                    'provider = "xiaomi"',
                    'model = "mimo-v2-pro"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            patch.dict(os.environ, {"AUTOBCI_PROVIDER_CONFIG": str(config_path)}, clear=False),
            patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": self._make_intake_stub_runtime()}),
        ):
            greeting = cli.run_codex_intake_agent_turn(
                "你好",
                {"program_state": {}, "autoresearch_status": {}},
                repo_root=repo_root,
                timeout_seconds=3,
            )
            image_task = cli.run_codex_intake_agent_turn(
                "program 我从零开始做一个纯图像任务，只用图像，不用脑电，判断 ship / not-ship 二分类。",
                {"program_state": {}, "autoresearch_status": {}},
                repo_root=repo_root,
                timeout_seconds=3,
            )

        self.assertEqual(greeting["tool_name"], "reply")
        self.assertIn("任务契约", greeting["message"])
        self.assertEqual(image_task["tool_name"], "draft_program")
        self.assertIn("纯图像", image_task["message"])

    def test_intake_llm_smoke_covers_common_and_plan_scenarios_with_mimo_provider(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[agents.intake]",
                    'provider = "xiaomi"',
                    'model = "mimo-v2-pro"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            patch.dict(os.environ, {"AUTOBCI_PROVIDER_CONFIG": str(config_path)}, clear=False),
            patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": self._make_intake_stub_runtime()}),
        ):
            payload = cli.run_intake_llm_smoke(
                repo_root=repo_root,
                provider="mimo",
                model="mimo-v2-pro",
            )

        self.assertTrue(payload["ok"], json.dumps(payload, ensure_ascii=False, indent=2))
        names = {str(item["name"]) for item in payload["steps"]}
        self.assertGreaterEqual(len(names), 6)
        self.assertIn("json_greeting_contract", names)
        self.assertIn("json_status_contract", names)
        self.assertIn("plan_image_only_draft", names)
        self.assertIn("plan_accept", names)
        plan_steps = [item for item in payload["steps"] if item["name"] == "plan_image_only_draft"]
        self.assertEqual(plan_steps[0]["pending_program_id"], "rsvp_ship_image_only_v0")
        self.assertNotIn("没有及时返回", json.dumps(payload, ensure_ascii=False))

    def test_intake_llm_smoke_cli_json_uses_mimo_provider(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[agents.intake]",
                    'provider = "xiaomi"',
                    'model = "mimo-v2-pro"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        out = io.StringIO()
        with (
            patch.dict(os.environ, {"AUTOBCI_PROVIDER_CONFIG": str(config_path)}, clear=False),
            patch.dict(sys.modules, {"bci_autoresearch.agent_runtime": self._make_intake_stub_runtime()}),
            redirect_stdout(out),
        ):
            code = cli.main(
                [
                    "--repo-root",
                    str(repo_root),
                    "smoke",
                    "intake-llm",
                    "--provider",
                    "mimo",
                    "--model",
                    "mimo-v2-pro",
                    "--json",
                ]
            )

        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["provider"], "xiaomi")

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

    def test_model_cli_current_list_set_key_and_test(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        secrets_path = repo_root / "provider_secrets.toml"
        runner_path = repo_root / "pi_runner_stub.py"
        runner_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import json, sys",
                    "json.loads(sys.stdin.read())",
                    "print(json.dumps({'ok': True, 'json': {'ok': True}}))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        runner_path.chmod(0o755)
        env = {
            "AUTOBCI_PROVIDER_CONFIG": str(config_path),
            "AUTOBCI_PROVIDER_SECRETS": str(secrets_path),
            "AUTOBCI_PI_RUNNER": str(runner_path),
            "XIAOMI_API_KEY": "mimo-secret-for-test",
        }

        with patch.dict(os.environ, env, clear=False):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(
                    cli.main(
                        [
                            "--repo-root",
                            str(repo_root),
                            "model",
                            "set",
                            "--agent",
                            "intake",
                            "--provider",
                            "mimo",
                            "--model",
                            "mimo-v2-pro",
                        ]
                    ),
                    0,
                )
            self.assertIn("计划/对话模型", out.getvalue())
            self.assertIn("mimo-v2-pro", out.getvalue())

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "model", "current", "--agent", "intake", "--json"]), 0)
            current = json.loads(out.getvalue())
            self.assertEqual(current["provider"], "xiaomi")
            self.assertEqual(current["model"], "mimo-v2-pro")

            out = io.StringIO()
            with redirect_stdout(out), patch("getpass.getpass", return_value="mini-secret-should-not-print"):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "model", "key", "minimax"]), 0)
            self.assertIn("minimax", out.getvalue())
            self.assertNotIn("mini-secret-should-not-print", out.getvalue())

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "model", "test", "mimo", "--json"]), 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "model", "list", "--json"]), 0)
            model_list = json.loads(out.getvalue())
            self.assertIn("agents", model_list)
            self.assertIn("providers", model_list)

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
        self.assertIn("textual", payload)
        self.assertIn("pi_runtime", payload)
        self.assertIn("structure_sandbox_runner", payload)
        self.assertIn("data_paths", payload)
        self.assertTrue(payload["textual"]["ok"])
        self.assertTrue(payload["pi_runtime"]["runner_file_exists"])
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

    def test_linux_doctor_command_prints_readiness(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        with (
            patch("bci_autoresearch.product_shell.cli.is_dashboard_running", return_value=False),
            patch("bci_autoresearch.product_shell.cli.shutil.which", return_value="/usr/bin/tool"),
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["--repo-root", str(repo_root), "linux", "doctor"]), 0)

        text = out.getvalue()
        self.assertIn("Linux readiness", text)
        self.assertIn("venv python", text)

    def test_model_intake_agent_timeout_reports_failure_without_program_draft(self) -> None:
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
        self.assertIn("计划/对话模型调用失败", message)
        self.assertIn("model timed out", message)
        self.assertIn("没有生成 Program", message)
        self.assertNotIn("pending_action", session_state)

    def test_model_intake_agent_invalid_json_task_reports_failure_without_program_draft(self) -> None:
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
        self.assertIn("计划/对话模型调用失败", message)
        self.assertIn("JSON 缺少字段", message)
        self.assertIn("没有生成 Program", message)
        self.assertNotIn("pending_action", session_state)

    def test_model_intake_agent_contract_error_is_not_hidden(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch(
                "bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn",
                side_effect=RuntimeError("计划/对话模型 JSON 缺少字段：tool_name"),
            ),
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
        self.assertIn("计划/对话模型调用失败", message)
        self.assertIn("计划/对话模型 JSON 缺少字段", message)
        self.assertIn("不会调用本地替代逻辑", message)
        self.assertNotIn("pending_action", session_state)

    def test_model_timeout_status_question_does_not_invent_boundary_answer(self) -> None:
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
        self.assertIn("计划/对话模型调用失败", message)
        self.assertIn("model timed out", message)
        self.assertNotIn("旧 AutoResearch", message)
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
        self.assertIn("新的计划对话", message)
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
        self.assertEqual(second_manifest["pending_action"]["plan_status"], "drafting")
        self.assertEqual(session_state["pending_action"]["plan_mode"], True)

    def test_new_clean_does_not_carry_stale_global_program_into_workspace(self) -> None:
        from bci_autoresearch.control_plane import build_status_snapshot, get_control_plane_paths
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        self._write_frozen_crossmodal_program(repo_root)
        session_state: dict[str, object] = {}
        paths = get_control_plane_paths(repo_root)

        should_quit, message = cli.handle_command(
            "/new clean",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        snapshot = cli._attach_experiment_state(build_status_snapshot(paths), paths, session_state)
        rendered = cli.build_tui_screen(snapshot, session_history=[], pending_action=None)
        manifest_path = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text())["path"]
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertFalse(should_quit)
        self.assertIn("New Clean", message)
        self.assertEqual(manifest["program_id"], "")
        self.assertEqual(manifest["program_status"], "not_started")
        self.assertNotIn("Program Frozen", rendered)
        self.assertNotIn("rsvp_ship_crossmodal_v0", rendered)

    def test_pending_program_draft_manifest_overrides_stale_global_program(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        self._write_frozen_crossmodal_program(repo_root)
        session_state: dict[str, object] = {}

        handle_command(
            "/new clean",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        handle_command(
            "现在生成 Program：我从零开始做一个纯图像任务，只用图像，不用脑电，判断图片是不是船，ship / not-ship 二分类，主指标 test_balanced_accuracy。",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        manifest_path = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text())["path"]
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["program_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(manifest["program_status"], "draft")
        self.assertEqual(manifest["pending_action"]["program_draft"]["research_goal"]["task_type"], "image_binary_classification")

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
            self.assertEqual(new_manifest["pending_action"]["plan_status"], "drafting")
            self.assertEqual(session_state["pending_action"]["plan_mode"], True)

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
        self.assertIn("Program:draft", rendered)

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
        self.assertIn("Topic:", message)
        self.assertIn("Attempt:", message)

    def test_lifecycle_store_migrates_old_project_db_to_topic_attempt_schema(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.lifecycle import ensure_lifecycle_store

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        db_path = Path(paths.monitor_dir) / "sessions.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                create table projects (
                  project_id text primary key,
                  title text not null,
                  status text not null,
                  created_at text not null,
                  updated_at text not null,
                  archived_at text not null default '',
                  parent_project_id text not null default '',
                  current_session_id text not null default '',
                  current_program_id text not null default '',
                  program_status text not null default 'not_started',
                  active_run_id text not null default '',
                  source_snapshot_id text not null default '',
                  pending_action_json text not null default 'null',
                  run_ids_json text not null default '[]',
                  artifact_refs_json text not null default '[]',
                  manifest_path text not null default ''
                );
                insert into projects (
                  project_id, title, status, created_at, updated_at
                ) values (
                  'proj-old', '旧任务', 'active', '2026-05-10T00:00:00Z', '2026-05-10T00:00:00Z'
                );
                """
            )

        ensure_lifecycle_store(paths)

        with sqlite3.connect(db_path) as conn:
            topic_table = conn.execute(
                "select name from sqlite_master where type = 'table' and name = 'topics'"
            ).fetchone()
            project_columns = {row[1] for row in conn.execute("pragma table_info(projects)").fetchall()}

        self.assertIsNotNone(topic_table)
        self.assertTrue(
            {
                "topic_id",
                "attempt_index",
                "attempt_title",
                "task_fingerprint",
                "debug_flag",
                "title_source",
                "tags_json",
            }.issubset(project_columns)
        )

    def test_pure_image_ship_title_groups_repeated_attempts_under_same_topic(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command
        from bci_autoresearch.product_shell.lifecycle import list_projects, list_topics
        from bci_autoresearch.control_plane import get_control_plane_paths

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = (
            "现在生成 Program：我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
            "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类。"
        )

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(request, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
            handle_command("/new clean", repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
            handle_command(request, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)

        projects = [item for item in list_projects(paths) if item.get("task_fingerprint")]
        topics = list_topics(paths)
        self.assertEqual(len({item["topic_id"] for item in projects}), 1)
        self.assertEqual(topics[0]["topic_title"], "纯图像船只二分类")
        self.assertEqual(sorted(item["attempt_index"] for item in projects), [1, 2])
        self.assertTrue(all(str(item["attempt_title"]).startswith("纯图像船只二分类尝试 #") for item in projects))
        self.assertTrue(all(item["title_source"] == "fallback_program" for item in projects))

    def test_cross_modal_prompt_uses_different_topic_from_image_only(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command
        from bci_autoresearch.product_shell.lifecycle import list_projects
        from bci_autoresearch.control_plane import get_control_plane_paths

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        image_request = "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。"
        cross_modal_request = "现在生成 Program：我想比较图像识别出船和用脑电识别出船，看看哪种方式效果更好。"

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(image_request, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
            handle_command("/new clean", repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
            handle_command(cross_modal_request, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)

        projects = [item for item in list_projects(paths) if item.get("task_fingerprint")]
        self.assertEqual(len({item["topic_id"] for item in projects}), 2)
        self.assertEqual(len({item["task_fingerprint"] for item in projects}), 2)

    def test_switch_command_lists_numbered_projects_and_stores_selection_context(self) -> None:
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
                "/switch",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("选择要切换的任务", message)
        self.assertIn("1.", message)
        selection_context = session_state.get("selection_context")
        self.assertIsInstance(selection_context, dict)
        self.assertEqual(selection_context["kind"], "topic_switch")
        self.assertGreaterEqual(len(selection_context["topics"]), 1)

    def test_project_switch_selection_resumes_numbered_project_and_clears_context(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            first_project_id = str(session_state["experiment_id"])
            handle_command(
                "/new clean",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            handle_command(
                "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            self.assertNotEqual(str(session_state["experiment_id"]), first_project_id)
            handle_command(
                "/switch",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "1.2",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已切换到任务 1.2", message)
        self.assertEqual(str(session_state["experiment_id"]), first_project_id)
        self.assertNotIn("selection_context", session_state)

    def test_project_switch_topic_number_expands_attempt_list(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            handle_command(
                "/switch",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "1",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("纯图像船只二分类", message)
        self.assertIn("1.1", message)
        self.assertEqual(session_state["selection_context"]["kind"], "topic_attempt_switch")

    def test_project_switch_selection_rejects_out_of_range_number_and_keeps_context(self) -> None:
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
                "/switch",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "9",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("没有第 9 个任务", message)
        self.assertIn("selection_context", session_state)

    def test_switch_default_hides_debug_attempts_but_debug_flag_shows_them(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
        request = "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。"

        with (
            patch.dict(os.environ, {"AUTOBCI_TUI_TEST_MODE": "1"}, clear=False),
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
        ):
            handle_command(request, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
            should_quit, default_message = handle_command(
                "/switch",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            debug_should_quit, debug_message = handle_command(
                "/switch --debug",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            all_should_quit, all_message = handle_command(
                "/switch all",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertFalse(debug_should_quit)
        self.assertFalse(all_should_quit)
        self.assertIn("当前还没有可切换的任务", default_message)
        self.assertIn("纯图像船只二分类", debug_message)
        self.assertIn("debug", debug_message)
        self.assertIn("纯图像船只二分类", all_message)

    def test_rename_updates_current_attempt_and_topic_titles(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.cli import handle_command
        from bci_autoresearch.product_shell.lifecycle import get_current_project, list_topics

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, attempt_message = handle_command(
                "/rename 手动调试尝试",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            topic_should_quit, topic_message = handle_command(
                "/rename topic 手动纯图像主题",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertFalse(topic_should_quit)
        self.assertIn("已重命名当前尝试", attempt_message)
        self.assertIn("已重命名当前 Topic", topic_message)
        self.assertEqual(get_current_project(paths)["attempt_title"], "手动调试尝试")
        self.assertEqual(list_topics(paths)[0]["topic_title"], "手动纯图像主题")

    def test_archive_topic_archives_topic_and_attempts(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.cli import handle_command
        from bci_autoresearch.product_shell.lifecycle import list_projects, list_topics

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "现在生成 Program：我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "/archive topic",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已归档 Topic", message)
        self.assertEqual(list_topics(paths)[0]["status"], "archived")
        self.assertTrue(all(item["status"] == "archived" for item in list_projects(paths)))

    def test_switch_command_handles_empty_project_list(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/switch",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("当前还没有可切换的任务", message)
        self.assertNotIn("selection_context", session_state)

    def test_chinese_switch_phrase_opens_project_switcher(self) -> None:
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
                "切换线程",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("选择要切换的任务", message)
        self.assertEqual(session_state["selection_context"]["kind"], "topic_switch")

    def test_chinese_new_session_phrase_starts_new_clean_workspace(self) -> None:
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
            first_project_id = str(session_state["experiment_id"])
            should_quit, message = handle_command(
                "新起一个 session",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已开始新的实验工作区", message)
        self.assertNotEqual(str(session_state["experiment_id"]), first_project_id)

    def test_chinese_snapshot_phrase_saves_current_state(self) -> None:
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
                "保存当前状态",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已保存快照", message)

    def test_model_command_opens_numbered_menu_and_stores_selection_context(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/model",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("模型设置", message)
        self.assertIn("当前计划/对话模型", message)
        self.assertIn("1. 切换当前模型", message)
        self.assertNotIn("Judge", message)
        self.assertNotIn("Guard", message)
        self.assertNotIn("Research", message)
        self.assertEqual(session_state["selection_context"]["kind"], "model_menu")

    def test_initial_model_setup_opens_when_intake_provider_key_is_missing(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        secrets_path = repo_root / "provider_secrets.toml"
        session_state: dict[str, object] = {}
        env = {
            "AUTOBCI_PROVIDER_CONFIG": str(config_path),
            "AUTOBCI_PROVIDER_SECRETS": str(secrets_path),
        }

        with patch.dict(os.environ, env, clear=True):
            message = cli._maybe_open_initial_model_setup(session_state)

        self.assertIn("首次配置", message)
        self.assertIn("没有内置模型 key", message)
        self.assertIn("当前计划/对话模型", message)
        self.assertIn("1. 配置 Provider API key", message)
        self.assertEqual(session_state["selection_context"]["kind"], "model_initial_setup")

    def test_initial_model_setup_key_path_prepares_hidden_input_and_auto_activation(self) -> None:
        from bci_autoresearch.product_shell import cli

        repo_root = self._make_temp_repo()
        config_path = repo_root / "providers.toml"
        secrets_path = repo_root / "provider_secrets.toml"
        session_state: dict[str, object] = {}
        env = {
            "AUTOBCI_PROVIDER_CONFIG": str(config_path),
            "AUTOBCI_PROVIDER_SECRETS": str(secrets_path),
        }

        with patch.dict(os.environ, env, clear=True):
            cli._maybe_open_initial_model_setup(session_state)
            message = cli._handle_model_selection(session_state, 1)
            self.assertIn("选择要配置 API key", message)
            self.assertTrue(session_state["selection_context"]["initial_setup"])
            message = cli._handle_model_selection(session_state, 3)

        self.assertIn("隐藏输入", message)
        secret_input = session_state["secret_input"]
        self.assertEqual(secret_input["provider"], "xiaomi")
        self.assertEqual(secret_input["after_save_agent"], "intake")
        self.assertEqual(secret_input["after_save_model"], "mimo-v2-pro")

    def test_initial_model_secret_save_tests_and_sets_intake(self) -> None:
        from bci_autoresearch.product_shell import cli

        calls: list[tuple[object, ...]] = []

        def fake_provider_call(names: tuple[str, ...], *args: object, **kwargs: object) -> dict[str, object]:
            calls.append((names, *args, kwargs))
            if "write_provider_secret" in names:
                return {"ok": True, "provider": args[0], "key_saved": True}
            if "set_agent_model" in names or "set_agent_provider_model" in names:
                return {"ok": True, "agent": args[0], "provider": args[1], "model": kwargs.get("model"), "live": True}
            raise AssertionError(f"unexpected provider call: {names}")

        with (
            patch("bci_autoresearch.product_shell.cli._provider_call", side_effect=fake_provider_call),
            patch(
                "bci_autoresearch.product_shell.cli._provider_test",
                return_value={"ok": True, "provider": "xiaomi", "model": "mimo-v2-pro"},
            ),
        ):
            message = cli._save_provider_secret_from_input(
                "xiaomi",
                "sk-secret-for-test",
                after_save_agent="intake",
                after_save_model="mimo-v2-pro",
        )

        self.assertIn("已保存 xiaomi API key", message)
        self.assertIn("计划/对话模型已设置为 xiaomi / mimo-v2-pro", message)
        self.assertIn("行动记录", message)
        self.assertIn("测试 Provider", message)
        self.assertIn("切换模型", message)
        self.assertTrue(any("write_provider_secret" in call[0] for call in calls))
        self.assertTrue(any("set_agent_model" in call[0] or "set_agent_provider_model" in call[0] for call in calls))

    def test_model_menu_selection_enters_provider_picker_for_current_model(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "/model",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "1",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
        )

        self.assertFalse(should_quit)
        self.assertIn("选择计划/对话模型使用的 Provider", message)
        self.assertNotIn("选择要配置的研究模块", message)
        self.assertNotIn("Judge", message)
        self.assertNotIn("Guard", message)
        self.assertNotIn("Research", message)
        self.assertEqual(session_state["selection_context"]["kind"], "model_provider_select")
        self.assertEqual(session_state["selection_context"]["agent"], "intake")

    def test_model_plain_command_opens_menu(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "model",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("模型设置", message)
        self.assertEqual(session_state["selection_context"]["kind"], "model_menu")

    def test_model_selection_missing_key_prepares_secret_input_before_test(self) -> None:
        from bci_autoresearch.product_shell import cli

        session_state: dict[str, object] = {
            "selection_context": {
                "kind": "model_model_select",
                "agent": "intake",
                "provider": "xiaomi",
                "options": [{"model": "mimo-v2-pro"}],
            }
        }

        with patch(
            "bci_autoresearch.product_shell.cli._model_provider_rows",
            return_value=[
                {
                    "name": "xiaomi",
                    "display_name": "Xiaomi MiMo",
                    "model": "mimo-v2-pro",
                    "default_model": "mimo-v2-pro",
                    "api_key_env": "XIAOMI_API_KEY",
                    "missing_api_key_env": "XIAOMI_API_KEY",
                    "ready": False,
                }
            ],
        ):
            message = cli._handle_model_selection(session_state, 1)

        self.assertIn("隐藏输入", message)
        self.assertNotIn("selection_context", session_state)
        self.assertEqual(session_state["secret_input"]["provider"], "xiaomi")
        self.assertEqual(session_state["secret_input"]["after_save_agent"], "intake")
        self.assertEqual(session_state["secret_input"]["after_save_model"], "mimo-v2-pro")

    def test_model_menu_rejects_out_of_range_number_and_keeps_context(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "/model",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "9",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("没有第 9 个选项", message)
        self.assertEqual(session_state["selection_context"]["kind"], "model_menu")

    def test_director_command_opens_numbered_menu_and_stores_selection_context(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/director",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("研究方向调度", message)
        self.assertIn("1. 生成研究队列", message)
        self.assertNotIn("Director 调试", message)
        self.assertEqual(session_state["selection_context"]["kind"], "director_menu")

    def test_director_menu_selection_generates_plan_and_clears_context(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        self._write_director_fixture(repo_root)
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "/director",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "1",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("已生成研究方向队列", message)
        self.assertIn("10+ 个方向", message)
        self.assertIn("行动记录", message)
        self.assertIn("Web Research", message)
        self.assertIn("生成研究方向队列", message)
        self.assertTrue((repo_root / "artifacts" / "monitor" / "director_plans" / "latest.json").exists())
        self.assertNotIn("selection_context", session_state)

    def test_remote_command_starts_current_session_bridge_and_accepts_http_message(self) -> None:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {"session_id": "intake-test", "experiment_id": "exp-test"}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/remote --host 127.0.0.1 --port 0",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("Remote 已开启", message)
        self.assertIn("current-session remote", message)
        bridge = session_state["_remote_bridge"]
        token = str(getattr(bridge, "token"))
        port = int(getattr(bridge, "bind_port"))
        runtime_path = repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json"
        runtime_text = runtime_path.read_text(encoding="utf-8")
        self.assertIn("current_tui_session", runtime_text)
        self.assertNotIn(token, runtime_text)

        try:
            unauthorized = Request(
                f"http://127.0.0.1:{port}/message",
                data=json.dumps({"text": "/help"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(unauthorized, timeout=5)
            self.assertEqual(raised.exception.code, 401)

            request = Request(
                f"http://127.0.0.1:{port}/message?token={token}",
                data=json.dumps({"text": "/help", "sender": "phone"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            reply = json.loads(urlopen(request, timeout=5).read().decode("utf-8"))
            self.assertTrue(reply["ok"])
            self.assertIn("常用命令", reply["reply"])

            events = json.loads(urlopen(f"http://127.0.0.1:{port}/events?token={token}", timeout=5).read().decode("utf-8"))
            self.assertTrue(events["ok"])
            self.assertEqual(events["inbox"][-1]["sender"], "phone")
            self.assertIn("常用命令", events["outbox"][-1]["text"])
        finally:
            handle_command(
                "/remote stop",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

    def test_director_menu_rejects_out_of_range_number_and_keeps_context(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "/director",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "9",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("没有第 9 个选项", message)
        self.assertEqual(session_state["selection_context"]["kind"], "director_menu")

    def test_director_latest_without_plan_is_nonfatal(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/director latest",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("还没有研究方向队列", message)

    def test_model_key_command_prepares_secret_input_without_transcript_turn(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.cli import handle_command, read_current_intake_history

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/model key minimax",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("隐藏输入", message)
        self.assertEqual(session_state["secret_input"]["kind"], "provider_api_key")
        self.assertEqual(session_state["secret_input"]["provider"], "minimax")
        self.assertEqual(read_current_intake_history(get_control_plane_paths(repo_root)), [])

    def test_model_key_mimo_alias_prepares_xiaomi_secret_input(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "/model key mimo",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("隐藏输入", message)
        self.assertEqual(session_state["secret_input"]["kind"], "provider_api_key")
        self.assertEqual(session_state["secret_input"]["provider"], "xiaomi")

    def test_model_provider_picker_lists_pi_backed_mimo(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            handle_command(
                "/model",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            handle_command(
                "1",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )
            should_quit, message = handle_command(
                "3",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("xiaomi", message)
        self.assertIn("MiMo", message)
        self.assertIn("mimo-v2-pro", message)

    def test_chinese_model_phrase_opens_model_menu(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            should_quit, message = handle_command(
                "切换模型",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertFalse(should_quit)
        self.assertIn("模型设置", message)
        self.assertEqual(session_state["selection_context"]["kind"], "model_menu")

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
        self.assertIn("Program:draft", rendered)
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
        self.assertIn("Phase:计划中", rendered)
        self.assertIn("Program", rendered)
        self.assertIn("描述你的研究任务", rendered)
        self.assertIn("› 直接描述任务，或输入 / 查看高级命令", rendered)
        self.assertNotIn("Program / Amendment", rendered)
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
        self.assertIn("Program", rendered)
        self.assertIn("描述你的研究任务", rendered)
        self.assertIn("dashboard 已打开", rendered)
        self.assertIn("› 直接描述任务，或输入 / 查看高级命令", rendered)
        self.assertNotIn("Program / Amendment", rendered)
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
                last_message="常用命令：new | run | model | tasks | dashboard | remote",
                last_command="help",
            )
        )
        rendered = console.export_text()

        self.assertIn("› help", rendered)
        self.assertIn("常用命令：new | run | model", rendered)
        self.assertIn("tasks", rendered)
        self.assertIn("remote", rendered)
        self.assertNotIn("help ·", rendered)
        self.assertNotIn("quit ·", rendered)
        self.assertNotIn("report latest", rendered)
        self.assertNotIn("program show", rendered)

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
            output_history=["AutoBCI> help", "常用命令：new | data | run | model | theme | tasks | dashboard | remote"],
            ui_tick=1,
        )

        self.assertIn("AutoBCI  Phase:", model["header_text"])
        self.assertIn("Program:", model["header_text"])
        self.assertEqual(model["run_status"], "live")
        self.assertIn("conversation_items", model)
        self.assertIn("system_event_items", model)
        self.assertIn("program", model)
        self.assertIn("status_rule", model)
        self.assertIn("system_trail", model)
        self.assertEqual(model["commands"], ["/new", "/data", "/run", "/model", "/theme", "/tasks", "/dashboard", "/remote"])
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
                    "next_step": "等待方向选择重新排序 formal 队列。",
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

        self.assertTrue(any(item["title"] == "方向选择 -> 执行沙盒" for item in items))
        self.assertTrue(any(item["title"] == "执行沙盒 -> 方向选择" for item in items))

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
                "status": "verified",
                "publish_ready": True,
            },
        }

        text = "".join(fragment for _style, fragment in _pt_header_fragments(snapshot, boot_mode=False))
        self.assertIn("Framework Benchmark", text)
        self.assertIn("总迭代 962", text)
        self.assertIn("突破率 1.1%", text)
        self.assertIn("吞吐 5.5/h", text)

    def test_header_fragments_hide_unverified_framework_benchmark_banner(self) -> None:
        from bci_autoresearch.product_shell.cli import _pt_header_fragments

        snapshot = {
            "stage": "smoke",
            "dashboard_url": "http://127.0.0.1:8878/",
            "framework_benchmark": {
                "total_iterations": 1160,
                "breakthrough_rate": 0.009,
                "cost_per_breakthrough": 111.1,
                "diversity_index": 0.72,
                "iterations_per_hour": 6.6,
            },
        }

        text = "".join(fragment for _style, fragment in _pt_header_fragments(snapshot, boot_mode=False))

        self.assertNotIn("Framework Benchmark", text)
        self.assertNotIn("总迭代 1160", text)
        self.assertNotIn("突破率", text)

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
                "status": "verified",
                "publish_ready": True,
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
        self.assertIn("audit_refs", row)
        self.assertNotIn("conversation_transcript", row)

        audit_rows = self._read_jsonl(repo_root / "artifacts" / "monitor" / "audit" / "judgment_chain.jsonl")
        self.assertEqual(len(audit_rows), 1)
        audit = audit_rows[0]
        self.assertEqual(audit["schema_version"], "autobci_audit_judgment_chain_v1")
        self.assertEqual(audit["session_id"], row["session_id"])
        self.assertEqual(audit["turn_id"], row["turn_id"])
        self.assertEqual(audit["decision"]["intent_kind"], "draft_proposal")
        self.assertEqual(audit["reasoning_visibility"]["raw_cot_visible"], False)
        self.assertEqual(audit["reasoning_visibility"]["raw_cot_saved"], False)
        self.assertNotIn("raw_chain_of_thought", json.dumps(audit, ensure_ascii=False))
        self.assertTrue((repo_root / "artifacts" / "monitor" / "audit" / "sessions" / f"{row['session_id']}.md").exists())

    def test_reasoning_raw_mode_saves_provider_returned_reasoning_only(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}
        fake_snapshot = {"program_state": {}, "autoresearch_status": {}}

        with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
            _, mode_message = handle_command(
                "/reasoning raw",
                repo_root=repo_root,
                host="127.0.0.1",
                port=8878,
                session_state=session_state,
            )

        self.assertEqual(session_state["reasoning_mode"], "raw")
        self.assertIn("provider 明确返回", mode_message)

        with (
            patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
            patch(
                "bci_autoresearch.product_shell.cli.run_codex_intake_agent_turn",
                return_value={
                    "tool_name": "reply",
                    "message": "收到。",
                    "normalized_request": "你好",
                    "reason": "寒暄。",
                    "raw_reasoning": "raw provider reasoning visible for debug",
                },
            ),
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
        self.assertIn("收到", message)
        audit_rows = self._read_jsonl(repo_root / "artifacts" / "monitor" / "audit" / "judgment_chain.jsonl")
        self.assertEqual(audit_rows[-1]["reasoning_visibility"]["mode"], "raw")
        self.assertEqual(audit_rows[-1]["reasoning_visibility"]["raw_cot_visible"], True)
        self.assertEqual(audit_rows[-1]["reasoning_visibility"]["raw_cot_saved"], True)
        self.assertIn("raw provider reasoning", audit_rows[-1]["reasoning_visibility"]["raw_cot_excerpt"])

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
        self.assertIn("AutoBCI  Phase", rendered)
        self.assertIn("Program", rendered)
        self.assertIn("描述你的研究任务", rendered)
        self.assertIn("AutoBCI 已退出。", rendered)
        self.assertNotIn("Program / Amendment", rendered)

    def test_pyproject_declares_autobci_console_script(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "autobci"', pyproject)
        self.assertIn("Local auditable research-loop harness alpha", pyproject)
        self.assertIn('autobci = "bci_autoresearch.product_shell.cli:main"', pyproject)
        self.assertIn("rich>=13.7", pyproject)
        self.assertIn("prompt_toolkit>=3.0", pyproject)
        self.assertIn("textual>=8.2", pyproject)

    def test_install_scripts_and_gitignore_cover_github_alpha_local_state(self) -> None:
        install = ROOT / "scripts" / "install.sh"
        linux = ROOT / "scripts" / "install_linux.sh"
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertTrue(install.exists())
        self.assertIn("python -m bci_autoresearch.product_shell.cli doctor --json", install.read_text(encoding="utf-8"))
        self.assertIn("scripts/install.sh", linux.read_text(encoding="utf-8"))
        for pattern in [
            ".autobci/",
            "artifacts/",
            "data/",
            ".venv/",
            "provider_secrets.toml",
            "*.pyc",
        ]:
            self.assertIn(pattern, gitignore)

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
        self.assertIn("textual>=8.2", requirements)

    def test_textual_tui_uses_blueprint_theme_palette(self) -> None:
        from bci_autoresearch.product_shell.textual_tui import AutoBciTextualApp, TEXTUAL_BLUEPRINT_THEME, TEXTUAL_THEMES

        self.assertEqual(TEXTUAL_BLUEPRINT_THEME["base"], "#06192d")
        self.assertEqual(TEXTUAL_BLUEPRINT_THEME["panel"], "#0d3d68")
        self.assertEqual(TEXTUAL_BLUEPRINT_THEME["accent"], "#8ee8ff")
        self.assertEqual(TEXTUAL_BLUEPRINT_THEME["ok"], "#8cffc4")
        self.assertEqual(TEXTUAL_BLUEPRINT_THEME["risk"], "#ffd166")
        self.assertEqual(TEXTUAL_THEMES["3"]["name"], "Blueprint")
        self.assertEqual(len(TEXTUAL_THEMES), 5)
        self.assertIn("name", TEXTUAL_THEMES["1"])
        self.assertNotEqual(TEXTUAL_THEMES["1"]["base"], TEXTUAL_THEMES["3"]["base"])
        self.assertNotEqual(TEXTUAL_THEMES["5"]["accent"], TEXTUAL_THEMES["3"]["accent"])
        css = AutoBciTextualApp.CSS
        for key in ("base", "surface", "top", "panel", "tool", "line", "text", "muted", "accent"):
            color = TEXTUAL_BLUEPRINT_THEME[key]
            self.assertIn(color, css)

    def test_textual_slash_menu_matches_primary_commands(self) -> None:
        from bci_autoresearch.product_shell.textual_tui import slash_menu_matches

        self.assertEqual(slash_menu_matches(""), [])
        self.assertEqual(slash_menu_matches("你好"), [])
        self.assertIn("/data", slash_menu_matches("/"))
        self.assertIn("/model", slash_menu_matches("/"))
        self.assertIn("/theme", slash_menu_matches("/"))
        self.assertEqual(slash_menu_matches("/d"), ["/data", "/dashboard"])
        self.assertEqual(slash_menu_matches("/th"), ["/theme"])
        self.assertEqual(slash_menu_matches("/h"), [])
        self.assertEqual(slash_menu_matches("/s"), [])
        self.assertEqual(slash_menu_matches("/new"), ["/new"])

    def test_help_message_prioritizes_primary_commands(self) -> None:
        from bci_autoresearch.product_shell.chat_actions import build_help_message

        message = build_help_message()
        self.assertIn("常用命令：new | data | run | model | theme | tasks | dashboard | remote", message)
        self.assertIn("高级命令：program show | status | help | quit | plan show | director | snapshot | fork | archive | resume", message)
        self.assertNotIn("聊天动作", message)
        self.assertNotIn("生命周期：new clean | switch | continue | snapshot", message)

    def test_slash_command_completer_shows_simplified_primary_menu(self) -> None:
        from prompt_toolkit.completion import Completer
        from prompt_toolkit.document import Document

        from bci_autoresearch.product_shell.cli import SLASH_COMMANDS, SLASH_MENU_COMMANDS, build_slash_command_completer

        advanced_supported = {
            "/new",
            "/data",
            "/new clean",
            "/plan",
            "/model",
            "/theme",
            "/reasoning",
            "/director",
            "/run",
            "/remote",
            "/tasks",
            "/switch",
            "/continue",
            "/projects",
            "/resume",
            "/snapshot",
            "/fork",
            "/archive",
            "/rename",
            "/title regenerate",
            "/clear",
            "/reset current run",
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
        primary = {
            "/new",
            "/data",
            "/run",
            "/model",
            "/theme",
            "/tasks",
            "/dashboard",
            "/remote",
        }
        self.assertTrue(advanced_supported.issubset(set(SLASH_COMMANDS)))
        self.assertEqual(set(SLASH_MENU_COMMANDS), primary)
        self.assertLessEqual(len(SLASH_MENU_COMMANDS), 8)

        completer = build_slash_command_completer()
        self.assertIsNotNone(completer)
        self.assertIsInstance(completer, Completer)
        root_items = {item.text for item in completer.get_completions(Document("/", cursor_position=1), Mock())}
        self.assertEqual(root_items, primary)
        self.assertNotIn("/snapshot", root_items)
        self.assertNotIn("/archive", root_items)
        self.assertNotIn("/program show", root_items)
        self.assertNotIn("/approve", root_items)
        d_items = {item.text for item in completer.get_completions(Document("/d", cursor_position=2), Mock())}
        self.assertEqual(d_items, {"/data", "/dashboard"})
        data_items = list(completer.get_completions(Document("/data", cursor_position=5), Mock()))
        self.assertEqual([item.text for item in data_items], ["/data"])
        self.assertIn("数据", data_items[0].display_meta_text)
        dashboard_items = [item.text for item in completer.get_completions(Document("/da", cursor_position=3), Mock())]
        self.assertEqual(dashboard_items, ["/data", "/dashboard"])
        dashboard_items = [item.text for item in completer.get_completions(Document("/das", cursor_position=4), Mock())]
        self.assertEqual(dashboard_items, ["/dashboard"])
        model_items = list(completer.get_completions(Document("/model", cursor_position=6), Mock()))
        self.assertEqual([item.text for item in model_items], ["/model"])
        self.assertIn("模型", model_items[0].display_meta_text)
        run_items = list(completer.get_completions(Document("/run", cursor_position=4), Mock()))
        self.assertEqual([item.text for item in run_items], ["/run"])
        self.assertIn("研究", run_items[0].display_meta_text)
        tasks_items = list(completer.get_completions(Document("/tasks", cursor_position=6), Mock()))
        self.assertEqual([item.text for item in tasks_items], ["/tasks"])
        self.assertIn("切换", tasks_items[0].display_meta_text)
        theme_items = list(completer.get_completions(Document("/theme", cursor_position=6), Mock()))
        self.assertEqual([item.text for item in theme_items], ["/theme"])
        self.assertIn("配色", theme_items[0].display_meta_text)
        s_items = {item.text for item in completer.get_completions(Document("/s", cursor_position=2), Mock())}
        self.assertEqual(s_items, set())
        new_items = {item.text for item in completer.get_completions(Document("/new", cursor_position=4), Mock())}
        self.assertEqual(new_items, {"/new"})
        remote_items = {item.text for item in completer.get_completions(Document("/r", cursor_position=2), Mock())}
        self.assertEqual(remote_items, {"/run", "/remote"})
        program_completions = list(completer.get_completions(Document("/program ", cursor_position=9), Mock()))
        self.assertEqual(program_completions, [])
        plain_items = list(completer.get_completions(Document("dashboard", cursor_position=9), Mock()))
        self.assertEqual(plain_items, [])

    def test_prompt_toolkit_tui_attaches_slash_completer_to_input_area(self) -> None:
        from prompt_toolkit.layout.containers import FloatContainer
        from prompt_toolkit.layout.menus import CompletionsMenu

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
            area = original_text_area(*args, **kwargs)
            captured["input_area"] = area
            return area

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
        self.assertTrue(text_area_kwargs["multiline"])
        self.assertTrue(text_area_kwargs["wrap_lines"])
        height = text_area_kwargs["height"]
        self.assertEqual(getattr(height, "min", None), 1)
        self.assertEqual(getattr(height, "max", None), 8)
        input_area = captured["input_area"]
        self.assertTrue(callable(input_area.window.height))
        layout = captured["kwargs"]["layout"]
        self.assertIsInstance(layout.container, FloatContainer)
        completion_floats = [
            item
            for item in layout.container.floats
            if getattr(item.content, "completion_frame", False)
            and isinstance(getattr(item.content, "completion_menu", None), CompletionsMenu)
        ]
        self.assertEqual(len(completion_floats), 1)
        completion_float = completion_floats[0]
        self.assertEqual(completion_float.left, 0)
        self.assertEqual(completion_float.right, 0)
        self.assertEqual(completion_float.bottom, 3)
        self.assertFalse(completion_float.ycursor)
        completion_menu = completion_float.content.completion_menu
        self.assertFalse(completion_menu.content.dont_extend_width())

    def test_prompt_toolkit_style_includes_global_background_for_padding(self) -> None:
        from bci_autoresearch.product_shell.cli import _build_prompt_toolkit_style, PALETTE

        style = _build_prompt_toolkit_style()
        self.assertIsNotNone(style)
        rules = getattr(style, "style_rules", [])
        self.assertIn(("app", f"bg:{PALETTE['background']} {PALETTE['text']}"), rules)

    def test_transcript_fragments_color_user_and_agent_messages_differently(self) -> None:
        from bci_autoresearch.product_shell.cli import _build_prompt_toolkit_style, _pt_transcript_fragments, PALETTE

        fragments = _pt_transcript_fragments(
            {
                "transcript_rows": [
                    {"role": "user", "text": "你好"},
                    {"role": "intake", "text": "收到，我会整理研究计划。"},
                ],
                "system_trail": {"show_default": False},
            }
        )

        self.assertIn(("class:message.user.text", "你好"), fragments)
        self.assertIn(("class:message.agent.text", "收到，我会整理研究计划。"), fragments)
        style = _build_prompt_toolkit_style()
        rules = dict(getattr(style, "style_rules", []))
        self.assertEqual(rules["message.user.text"], PALETTE["user_text"])
        self.assertEqual(rules["message.agent.text"], PALETTE["agent_text"])
        self.assertNotEqual(rules["message.user.text"], rules["message.agent.text"])

    def test_run_tui_prefers_textual_path_by_default_when_available(self) -> None:
        from bci_autoresearch.product_shell import cli

        with (
            patch.dict(os.environ, {"AUTOBCI_TUI_ENGINE": "auto"}, clear=False),
            patch.object(cli, "_textual_available", return_value=True),
            patch.object(cli, "PROMPT_TOOLKIT_AVAILABLE", True),
            patch.object(cli.sys.stdout, "isatty", return_value=True),
            patch.object(cli, "_run_textual_tui", return_value=6) as textual_runner,
            patch.object(cli, "_run_prompt_toolkit_tui", return_value=7) as pt_runner,
            patch.object(cli, "_run_rich_tui", return_value=8) as rich_runner,
        ):
            exit_code = cli.run_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
            )

        self.assertEqual(exit_code, 6)
        textual_runner.assert_called_once()
        pt_runner.assert_not_called()
        rich_runner.assert_not_called()

    def test_run_tui_can_force_prompt_toolkit_legacy_engine(self) -> None:
        from bci_autoresearch.product_shell import cli

        with (
            patch.dict(os.environ, {"AUTOBCI_TUI_ENGINE": "prompt_toolkit"}, clear=False),
            patch.object(cli, "_textual_available", return_value=True),
            patch.object(cli, "PROMPT_TOOLKIT_AVAILABLE", True),
            patch.object(cli.sys.stdout, "isatty", return_value=True),
            patch.object(cli, "_run_textual_tui", return_value=6) as textual_runner,
            patch.object(cli, "_run_prompt_toolkit_tui", return_value=7) as pt_runner,
            patch.object(cli, "_run_rich_tui", return_value=8) as rich_runner,
        ):
            exit_code = cli.run_tui(
                repo_root=ROOT,
                host="127.0.0.1",
                port=8878,
            )

        self.assertEqual(exit_code, 7)
        textual_runner.assert_not_called()
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
        self.assertTrue(profile["mouse_support"])
        self.assertFalse(profile["animate_ui"])
        self.assertTrue(profile["defer_repaint_while_typing"])
        self.assertIsNone(profile["cursor"])

        with patch.dict(cli.os.environ, {"AUTOBCI_DISABLE_MOUSE": "1"}, clear=False):
            self.assertFalse(cli._terminal_runtime_profile()["mouse_support"])

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
        self.assertTrue(captured["kwargs"]["mouse_support"])
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
        self.assertIn(("up",), keys)
        self.assertIn(("down",), keys)
        self.assertIn(("escape", "c-m"), keys)
        self.assertIn(("c-j",), keys)

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
