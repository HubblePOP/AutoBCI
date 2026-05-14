from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.support.tui_harness import TuiSession


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _make_repo_with_director_state(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    monitor_dir = repo_root / "artifacts" / "monitor"
    monitor_dir.mkdir(parents=True)
    handoff_dir = repo_root / "memory" / "docs" / "dev_pack_2026_04_20" / "08_LOCAL_AGENT_HANDOFF"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "DIRECTOR_AGENT.md").write_text(
        "# 研究方向队列生成器\n\n只生成研究方向队列，不启动执行沙盒，不写正式执行 manifest。\n",
        encoding="utf-8",
    )
    (monitor_dir / "rsvp_ship_image_autoresearch_latest.json").write_text(
        json.dumps(
            {
                "run_id": "rsvp-ship-image-pty-test",
                "created_at": "2026-05-10T01:04:32Z",
                "program_id": "rsvp_ship_image_only_v0",
                "dataset_name": "Downloads/RSVP跨模态数据",
                "status": "completed_image_only",
                "target_mode": "rsvp_ship_image_classification",
                "primary_metric": "test_balanced_accuracy",
                "benchmark_primary_score": 0.8886,
                "test_primary_metric": 0.8696,
                "no_cross_modal_claim": True,
                "eeg_status": "not_used_image_only_debug",
                "selected_model": {
                    "model_id": "image_logistic_baseline",
                    "feature_view": "grayscale_32x32_flat",
                    "algorithm": "numpy weighted logistic regression",
                },
                "split": {"train": 700, "validation": 150, "test": 150},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return repo_root


def _seed_switch_attempts(repo_root: Path) -> None:
    from bci_autoresearch.product_shell.cli import handle_command

    session_state: dict[str, object] = {}
    fake_snapshot = {"program_state": {}, "autoresearch_status": {}}
    prompt = "我想做纯图像任务，只用图像判断图片是不是船，ship / not-ship 二分类。"
    generate = "可以了，按现在聊的版本生成 Program markdown。"

    with patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot):
        handle_command(prompt, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
        handle_command(generate, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
        handle_command("/new clean", repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
        handle_command(prompt, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)
        handle_command(generate, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=session_state)

    debug_state: dict[str, object] = {}
    with (
        patch.dict("os.environ", {"AUTOBCI_TUI_TEST_MODE": "1"}, clear=False),
        patch("bci_autoresearch.product_shell.cli.build_status_snapshot", return_value=fake_snapshot),
    ):
        handle_command("/new clean", repo_root=repo_root, host="127.0.0.1", port=8878, session_state=debug_state)
        handle_command(prompt, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=debug_state)
        handle_command(generate, repo_root=repo_root, host="127.0.0.1", port=8878, session_state=debug_state)


def test_slash_menu_is_visible_and_stays_inside_terminal_frame(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        assert "Framework Benchmark" not in tui.screen_text()
        startup_screen = tui.screen_text()
        assert "Intake:" not in startup_screen
        assert "Director:" not in startup_screen
        assert "Executor:" not in startup_screen
        assert "Judge:" not in startup_screen
        assert "Guard:" not in startup_screen
        assert "Intake Agent" not in startup_screen
        tui.send_text("/")
        tui.wait_for_text("/run")
        tui.assert_visible("/model")
        tui.assert_visible("/theme")
        tui.assert_visible("/tasks")
        tui.assert_visible("/remote")
        tui.assert_visible("切换模型")
        tui.assert_visible("修改配色")
        screen = tui.screen_text()
        assert "/director" not in screen
        assert "/status" not in screen
        assert "/help" not in screen
        assert "/quit" not in screen
        assert "/snapshot" not in screen
        assert "/archive" not in screen
        assert "/program show" not in screen
        assert "┌└" not in screen
        assert "└│" not in screen

        for line in screen.splitlines():
            assert _display_width(line.rstrip()) <= tui.cols


def test_empty_default_plan_card_is_hidden_on_startup(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        screen = tui.screen_text()
        assert "研究计划流" not in screen
        assert "研究计划 / Program" not in screen
        assert "尚未形成 Program 草案" not in screen
        assert "缺失字段：研究目标、任务类型、成功指标" not in screen
        assert "1. 确认并开始运行" not in screen


def test_textual_startup_hides_stale_quit_events(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    events_path = repo_root / "artifacts" / "monitor" / "control_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "recorded_at": f"2026-05-12T06:0{index}:00Z",
                    "action": "quit",
                    "message": "AutoBCI 已退出。",
                    "ok": True,
                },
                ensure_ascii=False,
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        screen = tui.screen_text()
        assert "quit:" not in screen
        assert "AutoBCI 已退出" not in screen
        tui.assert_no_crash()


def test_long_markdown_reply_is_not_truncated_or_shown_with_raw_stars(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    sessions_dir = repo_root / "artifacts" / "monitor" / "intake_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    history_path = sessions_dir / "intake-markdown-test.jsonl"
    long_reply = "\n".join(
        [
            "第一段：**关键判断** 是先做纯图像 ship / not-ship 二分类。",
            "第二段：**数据来源** 是本地任务目录，不在这里改 data/raw。",
            "第三段：**指标** 是 test_balanced_accuracy，同时保留混淆矩阵。",
            "第四段：**风险** 是背景捷径、重复样本和 lucky split。",
            "第五段：**下一步** 是冻结 Program 后进入研究闭环。",
        ]
    )
    history_path.write_text(
        json.dumps(
            {
                "created_at": "2026-05-12T01:00:00Z",
                "role": "intake",
                "text": long_reply,
                "intent_kind": "draft_program",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (sessions_dir / "current.json").write_text(
        json.dumps({"session_id": "intake-markdown-test", "path": str(history_path), "updated_at": "2026-05-12T01:00:01Z"}),
        encoding="utf-8",
    )

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("第五段")
        screen = tui.screen_text()
        assert "关键判断" in screen
        assert "**" not in screen
        assert "第五段" in screen
        assert "下一步" in screen
        assert "…" not in screen
        tui.assert_no_crash()


def test_first_launch_without_model_key_shows_setup_menu(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    provider_envs = [
        "OPENAI_API_KEY",
        "MINIMAX_API_KEY",
        "XIAOMI_API_KEY",
        "DEEPSEEK_API_KEY",
        "KIMI_API_KEY",
        "ZAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    with TuiSession(repo_root=repo_root, cols=132, rows=34, tui_test_mode=False, remove_env=provider_envs) as tui:
        tui.wait_for_text("首次配置")
        tui.assert_visible("没有内置模型 key")
        tui.assert_visible("1. 配置 Provider API key")
        tui.submit("1")
        tui.wait_for_text("选择要配置 API key")
        tui.submit("3")
        tui.wait_for_text("xiaomi API key")
        tui.assert_no_crash()


def test_assistant_replies_use_bullet_without_role_prefix(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("你好")
        tui.wait_for_text("· 你好")
        screen = tui.screen_text()
        assert "研究计划流" not in screen
        assert "› 你好" in screen


def test_textual_tui_starts_as_modern_default_engine_and_accepts_input(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI")
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.wait_for_text("Enter 发送")
        assert "研究审计" not in tui.screen_text()
        tui.submit("你好")
        tui.wait_for_text("你好")
        tui.wait_for_text("AutoBCI")
        tui.assert_no_crash()


def test_textual_slash_picker_navigates_and_executes_with_enter(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.send_text("/")
        tui.wait_for_text("常用命令")
        tui.assert_visible("> /new")
        tui.assert_visible("/run")
        tui.assert_visible("↑↓ 选择")
        tui.assert_no_crash()

        tui.send_key("down")
        tui.wait_for_text("> /data")
        tui.assert_no_crash()
        tui.send_key("up")
        tui.wait_for_text("> /new")
        tui.assert_no_crash()
        tui.clear_input()
        tui.assert_no_crash()

    repo_root = _make_repo_with_director_state(tmp_path / "help")
    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.send_text("/m")
        tui.wait_for_text("> /model")
        tui.send_key("enter")
        tui.wait_for_text("模型设置")
        tui.assert_no_crash()


def test_textual_data_path_prompt_accepts_absolute_path(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    dataset = tmp_path / "RSVP图像数据"
    dataset.mkdir()

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit("/data")
        tui.wait_for_text("选择本地数据目录")
        tui.submit(str(dataset))
        tui.wait_for_text("已保存本地数据目录")
        tui.assert_no_crash()

    config_path = repo_root / ".autobci" / "data_paths.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["tasks"]["rsvp_ship_image_only_v0"]["dataset_root"] == str(dataset.resolve())


def test_textual_theme_command_lists_and_switches_palette(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit("/theme")
        tui.wait_for_text("选择配色")
        tui.assert_visible("> 1. Graphite")
        tui.assert_visible("3. Blueprint")
        tui.assert_no_crash()

        tui.send_keys(["down", "enter"])
        tui.wait_for_text("配色已切换：2")
        tui.assert_no_crash()

        tui.submit("/theme")
        tui.wait_for_text("选择配色")
        tui.send_text("4")
        tui.wait_for_text("配色已切换：4")
        tui.assert_no_crash()

    repo_root = _make_repo_with_director_state(tmp_path / "direct")
    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit("/theme 5")
        tui.wait_for_text("配色已切换：5")
        tui.assert_no_crash()


def test_textual_escape_clears_stale_model_picker_before_theme_switch(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=36,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit("model")
        tui.wait_for_text("模型设置")
        tui.send_key("enter")
        tui.wait_for_text("选择计划/对话模型使用的 Provider")
        tui.send_keys(["down", "down", "enter"])
        tui.wait_for_text("选择 xiaomi (Xiaomi MiMo) 的模型")
        tui.assert_visible("> 1. mimo-v2-pro")

        tui.send_key("escape")
        tui.assert_no_crash()
        tui.submit("/theme")
        tui.wait_for_text("选择配色")
        tui.send_text("1")
        tui.wait_for_text("配色已切换：1")
        tui.assert_no_crash()
        screen = tui.screen_text()
        assert "选择模型  ↑↓" not in screen
        assert "> 1. mimo-v2-pro" not in screen


def test_textual_tui_uses_single_column_feed_without_message_boxes(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        screen = tui.screen_text()
        assert "研究审计" not in screen
        assert "╭" not in screen
        assert "╰" not in screen
        assert "→ 下一步" not in screen
        tui.assert_no_crash()


def test_textual_short_conversation_is_anchored_near_composer(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        screen = tui.wait_for_text("已接入当前研究态")
        lines = screen.splitlines()
        message_line = next(index for index, line in enumerate(lines) if "已接入当前研究态" in line)
        assert message_line >= 18
        assert "已接入当前研究态" not in "\n".join(lines[:12])
        tui.assert_no_crash()


def test_textual_conversation_tail_stacks_from_bottom(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=110,
        rows=28,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit("第一条唯一消息")
        tui.wait_for_text("第一条唯一消息")
        tui.submit("第二条唯一消息")
        tui.wait_for_text("第二条唯一消息")
        tui.assert_no_crash()

        lines = tui.screen_text().splitlines()

        def line_index(text: str) -> int:
            for index, line in enumerate(lines):
                if text in line:
                    return index
            raise AssertionError(f"missing visible text: {text}")

        first = line_index("第一条唯一消息")
        second = line_index("第二条唯一消息")
        assert first < second
        assert line_index("已接入当前研究态") < first
        assert line_index("最近工具 / 判断") < first


def test_textual_next_actions_are_picker_items_not_inline_numbers(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    discussion = (
        "我想从零开始做一个纯图像任务，只用图像，不用脑电。"
        "目标是判断图片是不是船，做 ship / not-ship 二分类，主指标 test_balanced_accuracy。"
    )

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=36,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit(discussion)
        tui.wait_for_text("还没有生成 Program")
        screen = tui.screen_text()
        assert "研究计划 / Program" not in screen
        assert "下一步动作" not in screen
        tui.assert_no_crash()

        tui.submit("可以了，按现在聊的版本生成 Program markdown。")
        tui.wait_for_text("Program：")
        tui.wait_for_text("禁止动作")
        tui.wait_for_text("下一步动作")
        tui.assert_visible("> 1. 确认并开始运行")
        screen = tui.screen_text()
        assert "→ 下一步" not in screen
        assert "下一步1." not in screen
        tui.assert_no_crash()

        tui.send_key("down")
        tui.wait_for_text("> 2. 补充修改意见")
        tui.send_key("enter")
        tui.wait_for_text("直接输入你想改的地方")
        tui.assert_no_crash()


def test_textual_tui_renders_ascii_diagram_in_main_feed(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    sessions_dir = repo_root / "artifacts" / "monitor" / "intake_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    history_path = sessions_dir / "intake-ascii-diagram-test.jsonl"
    reply = "\n".join(
        [
            "研究闭环示意：",
            "```text",
            "Program -> 研究方向 -> 执行沙盒 -> 固定评估 -> 结果复核 -> ledger",
            "    ^                                              |",
            "    +---------------- ledger / evidence <----------+",
            "```",
        ]
    )
    history_path.write_text(
        json.dumps(
            {
                "created_at": "2026-05-12T03:00:00Z",
                "role": "intake",
                "text": reply,
                "intent_kind": "reply",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (sessions_dir / "current.json").write_text(
        json.dumps({"session_id": "intake-ascii-diagram-test", "path": str(history_path), "updated_at": "2026-05-12T03:00:01Z"}),
        encoding="utf-8",
    )

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=34,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("Program -> 研究方向")
        tui.assert_visible("ledger / evidence")
        screen = tui.screen_text()
        assert "研究审计" not in screen
        assert "╭" not in screen
        assert "╰" not in screen
        tui.assert_no_crash()


def test_command_results_render_as_tool_box(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/status")
        tui.wait_for_text("工具调用")
        tui.assert_visible("AutoBci 内置控制面")
        screen = tui.screen_text()
        assert "研究计划流" not in screen
        tui.assert_no_crash()


def test_reasoning_raw_mode_marks_unavailable_cot_without_crash(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/reasoning raw")
        tui.wait_for_text("推理调试已切到 raw")
        tui.submit("你好")
        tui.wait_for_text("provider 未返回原始 CoT")
        tui.assert_no_crash()


def test_slash_completion_up_down_keys_do_not_crash(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.send_text("/t")
        tui.wait_for_text("/tasks")
        tui.send_key("up")
        tui.send_key("down")
        tui.assert_no_crash()
        screen = tui.screen_text()
        raw = "".join(tui.raw_chunks)

    assert "Unhandled exception" not in raw
    assert "Press ENTER to continue" not in screen


def test_all_slash_prefixes_navigate_without_crash(tmp_path: Path) -> None:
    prefixes = {
        "/": "/run",
        "/n": "/new",
        "/t": "/tasks",
        "/th": "/theme",
        "/m": "/model",
        "/d": "/dashboard",
        "/r": "/remote",
    }
    for prefix, expected in prefixes.items():
        repo_root = _make_repo_with_director_state(tmp_path / prefix.replace("/", "root").replace(" ", "_"))
        with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
            tui.wait_for_text("描述你的研究任务")
            tui.send_text(prefix)
            tui.wait_for_text(expected)
            tui.send_keys(["up", "down", "enter"])
            tui.assert_no_crash()
            tui.expect_alive()


def test_all_root_commands_smoke_without_crash(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    commands = [
        "/new",
        "/data",
        "/data show",
        "/new clean",
        "/plan",
        "/model",
        "/theme",
        "/reasoning",
        "/director",
        "/run status",
        "/research status",
        "/remote --host 127.0.0.1 --port 0",
        "/remote stop",
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
    ]

    with TuiSession(repo_root=repo_root, cols=132, rows=36) as tui:
        tui.wait_for_text("描述你的研究任务")
        for command in commands:
            tui.submit(command)
            tui.assert_no_crash()
            tui.expect_alive()
        tui.wait_for_text("常用命令")
        tui.submit("/quit")
        tui.assert_no_crash()
        tui.wait_for_text("AutoBCI 已退出")


def test_model_menu_navigation_matrix_without_real_provider(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=36) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/model")
        tui.wait_for_text("模型设置")
        tui.assert_visible("当前计划/对话模型")
        screen = tui.screen_text()
        assert "Judge" not in screen
        assert "Guard" not in screen
        assert "Research" not in screen
        tui.send_keys(["up", "down"])
        tui.assert_no_crash()
        tui.submit("9")
        tui.wait_for_text("没有第 9 个选项")
        tui.assert_no_crash()
        tui.submit("1")
        tui.wait_for_text("选择计划/对话模型使用的 Provider")
        tui.send_keys(["up", "down"])
        tui.assert_no_crash()
        tui.submit("9")
        tui.wait_for_text("没有第 9 个选项")
        tui.submit("3")
        tui.wait_for_text("选择 xiaomi (Xiaomi MiMo) 的模型")
        tui.submit("1")
        tui.wait_for_text("已准备为 xiaomi 保存 API key")
        tui.assert_visible("xiaomi API key")
        tui.assert_no_crash()


def test_textual_model_picker_uses_nested_menu_until_key_input(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(
        repo_root=repo_root,
        cols=132,
        rows=36,
        extra_env={"AUTOBCI_TUI_ENGINE": "textual"},
    ) as tui:
        tui.wait_for_text("AutoBCI · 研究控制台")
        tui.submit("model")
        tui.wait_for_text("模型设置")
        tui.assert_visible("> 1")
        tui.assert_visible("切换当前模型")
        tui.assert_no_crash()

        tui.send_key("enter")
        tui.wait_for_text("选择计划/对话模型使用的 Provider")
        tui.assert_visible("> 1")
        screen = tui.screen_text()
        assert "选择要配置的研究模块" not in screen
        assert "Judge" not in screen
        assert "Guard" not in screen
        assert "Research" not in screen
        tui.assert_no_crash()

        tui.assert_visible("openai")
        tui.assert_visible("xiaomi")
        tui.send_keys(["down", "down", "enter"])
        tui.wait_for_text("选择 xiaomi (Xiaomi MiMo) 的模型")
        tui.assert_visible("mimo-v2-pro")
        tui.assert_no_crash()

        tui.send_key("enter")
        tui.wait_for_text("已准备为 xiaomi 保存 API key")
        tui.assert_visible("粘贴 xiaomi API key")
        tui.assert_no_crash()


def test_director_menu_navigation_matrix_without_executor(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=38, timeout=12.0) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/director")
        tui.wait_for_text("研究方向调度")
        tui.send_keys(["up", "down"])
        tui.assert_no_crash()
        tui.submit("9")
        tui.wait_for_text("没有第 9 个选项")
        tui.submit("/director")
        tui.submit("1")
        tui.wait_for_text("已生成研究方向队列", timeout=15.0)
        tui.assert_no_crash()
        tui.submit("/director")
        tui.submit("2")
        tui.wait_for_text("研究方向队列")
        tui.submit("/director")
        tui.submit("3")
        tui.wait_for_text("研究证据包")
        tui.submit("/director")
        tui.submit("4")
        tui.wait_for_text("已退出研究方向调度")
        tui.assert_no_crash()


def test_switch_topic_attempt_navigation_matrix(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    _seed_switch_attempts(repo_root)

    with TuiSession(repo_root=repo_root, cols=132, rows=38) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/switch")
        tui.wait_for_text("纯图像船只二分类")
        tui.send_keys(["up", "down"])
        tui.assert_no_crash()
        tui.submit("9")
        tui.wait_for_text("没有第 9 个任务")
        tui.submit("/switch")
        tui.submit("1")
        tui.wait_for_text("1.1")
        tui.submit("1.1")
        tui.wait_for_text("已切换到任务 1.1")
        tui.submit("/switch all")
        tui.wait_for_text("debug")
        tui.submit("/switch --debug")
        tui.wait_for_text("debug")
        tui.assert_no_crash()


def test_input_history_completion_and_multiline_keys_do_not_conflict(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.send_text("第一行")
        tui.send_key("alt-enter")
        tui.send_text("第二行")
        tui.assert_visible("第一行")
        tui.assert_visible("第二行")
        tui.clear_input()
        tui.send_text("abc")
        tui.send_keys(["up", "down"])
        tui.assert_no_crash()
        tui.clear_input()
        tui.send_text("/t")
        tui.wait_for_text("/tasks")
        tui.send_keys(["up", "down"])
        tui.assert_no_crash()


def test_long_chinese_paste_is_visible_before_submit(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    prompt = (
        "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
        "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
        "先形成 Program，不要启动 Executor，也不要开始正式 AutoResearch。"
    )

    with TuiSession(repo_root=repo_root, cols=132, rows=34) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.send_text(prompt)
        tui.wait_for_text("纯图像任务")
        tui.assert_visible("不用脑电")
        tui.assert_visible("not-ship")


def test_conversation_keeps_prior_messages_when_space_available(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=120, rows=42, timeout=10.0) as tui:
        tui.wait_for_text("描述你的研究任务")
        for index in range(1, 5):
            text = f"第{index}条消息"
            tui.submit(text)
            screen = tui.wait_for_text(text)
            tui.assert_no_crash()

        screen = tui.screen_text()
        assert screen.rfind("第4条消息") > screen.rfind("第3条消息")
        assert "第1条消息" in screen
        assert "第4条消息" in screen


def test_conversation_history_can_scroll_back_after_many_messages(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=120, rows=22, timeout=10.0) as tui:
        tui.wait_for_text("描述你的研究任务")
        for index in range(1, 11):
            text = f"滚动测试{index:02d}"
            tui.submit(text)
            tui.wait_for_text(text)
            tui.assert_no_crash()

        screen = tui.screen_text()
        assert "滚动测试10" in screen
        assert "滚动测试01" not in screen
        tui.send_keys(["pageup", "pageup", "pageup", "pageup", "pageup", "pageup"])
        tui.wait_for_text("滚动测试01")
        tui.assert_no_crash()


def test_conversation_history_mouse_wheel_scrolls_back(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=110, rows=20, timeout=10.0) as tui:
        tui.wait_for_text("描述你的研究任务")
        for index in range(1, 9):
            text = f"鼠标滚动测试{index:02d}"
            tui.submit(text)
            tui.wait_for_text(text)
            tui.assert_no_crash()

        screen = tui.screen_text()
        assert "鼠标滚动测试08" in screen
        assert "鼠标滚动测试01" not in screen
        tui.send_mouse_wheel_up(count=16)
        tui.wait_for_text("鼠标滚动测试01")
        tui.send_text("焦点恢复测试")
        tui.assert_visible("焦点恢复测试")
        tui.send_key("enter")
        tui.wait_for_text("焦点恢复测试")
        tui.assert_no_crash()


def test_clicking_conversation_does_not_steal_typing_focus(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)

    with TuiSession(repo_root=repo_root, cols=110, rows=20, timeout=10.0) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("点击焦点准备")
        tui.wait_for_text("点击焦点准备")
        tui.send_mouse_click(col=20, row=8)
        tui.send_text("点击后仍可输入")
        tui.assert_visible("点击后仍可输入")
        tui.send_key("enter")
        tui.wait_for_text("点击后仍可输入")
        tui.assert_no_crash()


def test_intake_to_research_loop_generates_image_only_step_without_executor(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    prompt = (
        "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
        "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
        "先形成 Program，不要启动 Executor。"
    )

    with TuiSession(repo_root=repo_root, cols=132, rows=36) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/new clean")
        tui.wait_for_text("已全新开始 New Clean")
        tui.submit(prompt)
        tui.wait_for_text("研究计划 / Program")
        tui.assert_visible("rsvp_ship_image_only_v0")
        tui.assert_visible("image_binary_classification")
        tui.submit("1")
        tui.wait_for_text("已确认并冻结 Program")
        tui.wait_for_text("Continue?")
        tui.assert_visible("[Y] Yes")
        tui.assert_visible("[D] Details")
        tui.submit("d")
        tui.wait_for_text("Details:")
        tui.assert_visible("track_id")
        tui.submit("y")
        tui.wait_for_text("判断：succeeded", timeout=15.0)
        tui.assert_visible("行动记录")
        tui.assert_visible("选择研究方向")
        tui.assert_visible("结果复核")
        tui.send_keys(["pageup", "pagedown"])
        tui.assert_no_crash()

    ledger = repo_root / "artifacts" / "research_loop" / "rsvp_ship_image_only_v0" / "ledger.jsonl"
    payload = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["task_id"] == "rsvp_ship_image_only_v0"
    assert payload["track_id"]
    assert payload["direction"]
    assert payload["safety"]["raw_data_touched"] is False
    assert payload["safety"]["formal_manifest_written"] is False
    assert payload["judgment_chain"]
    assert payload["rules_checked"]
    assert isinstance(payload["risk_flags"], list)
    events = repo_root / "artifacts" / "research_loop" / "rsvp_ship_image_only_v0" / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["event_type"] == "human_gate_waiting" for row in rows)
    assert any(row["event_type"] == "judge_decision" for row in rows)

    current = json.loads((repo_root / "artifacts" / "monitor" / "experiments" / "current.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(current["path"]).read_text(encoding="utf-8"))
    assert manifest["debug_flag"] is True


def test_plan_mode_program_plan_to_director_without_executor(tmp_path: Path) -> None:
    repo_root = _make_repo_with_director_state(tmp_path)
    prompt = (
        "我想从零开始做一个纯图像任务。数据在下载文件夹里的 RSVP 跨模态数据里；"
        "这一轮只用图像，不用脑电。目标是判断图片是不是船，做 ship / not-ship 二分类，"
        "先形成 Program，不要启动 Executor。"
    )

    with TuiSession(repo_root=repo_root, cols=132, rows=38) as tui:
        tui.wait_for_text("描述你的研究任务")
        tui.submit("/new clean")
        tui.wait_for_text("已全新开始 New Clean")
        tui.submit("/plan")
        tui.wait_for_text("Program 起草")
        tui.assert_no_crash()
        tui.send_text(prompt)
        tui.wait_for_text("纯图像任务")
        tui.assert_visible("不用脑电")
        tui.send_key("enter")
        tui.wait_for_text("研究计划 / Program")
        tui.assert_visible("rsvp_ship_image_only_v0")
        tui.submit("/plan show")
        tui.wait_for_text("Program.md")
        tui.assert_visible("image_binary_classification")
        tui.submit("/approve")
        tui.wait_for_text("选择“确认并开始运行”")
        tui.assert_no_crash()
        tui.submit("/plan accept")
        tui.wait_for_text("Program 已确认")
        tui.submit("/approve")
        tui.wait_for_text("已冻结 Program")
        tui.submit("/director")
        tui.wait_for_text("生成研究队列")
        tui.submit("1")
        tui.wait_for_text("研究方向队列：", timeout=15.0)
        tui.assert_visible("10. [候选]")
        tui.assert_no_crash()

    latest = repo_root / "artifacts" / "monitor" / "director_plans" / "latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["program_id"] == "rsvp_ship_image_only_v0"
    assert payload["safety"]["executor_started"] is False
    assert {track["input_mode"] for track in payload["tracks"]} == {"image_only"}
