from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pexpect
import pyte


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TuiSession:
    def __init__(
        self,
        *,
        repo_root: Path,
        cols: int = 132,
        rows: int = 34,
        timeout: float = 8.0,
        python_executable: str | None = None,
        tui_test_mode: bool = True,
        extra_env: dict[str, str] | None = None,
        remove_env: Iterable[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.cols = cols
        self.rows = rows
        self.timeout = timeout
        self.python_executable = python_executable or sys.executable
        self.tui_test_mode = tui_test_mode
        self.extra_env = dict(extra_env or {})
        self.remove_env = tuple(remove_env or ())
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.child: pexpect.spawn | None = None
        self.raw_chunks: list[str] = []
        self.actions: list[str] = []

    def __enter__(self) -> TuiSession:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"
        if self.tui_test_mode:
            env["AUTOBCI_TUI_TEST_MODE"] = "1"
        else:
            env.pop("AUTOBCI_TUI_TEST_MODE", None)
        env.setdefault("AUTOBCI_TUI_ENGINE", "prompt_toolkit")
        isolated_config_dir = self.repo_root / ".autobci" / "tui-test"
        isolated_config_dir.mkdir(parents=True, exist_ok=True)
        env["AUTOBCI_PROVIDER_CONFIG"] = str(isolated_config_dir / "providers.toml")
        env["AUTOBCI_PROVIDER_SECRETS"] = str(isolated_config_dir / "provider_secrets.toml")
        for name in self.remove_env:
            env.pop(name, None)
        env.update(self.extra_env)
        env["PYTHONUNBUFFERED"] = "1"
        env["TERM"] = env.get("TERM") or "xterm-256color"
        env["COLUMNS"] = str(self.cols)
        env["LINES"] = str(self.rows)
        self.child = pexpect.spawn(
            self.python_executable,
            ["-m", "bci_autoresearch.product_shell.cli", "--repo-root", str(self.repo_root)],
            cwd=str(ROOT),
            env=env,
            encoding="utf-8",
            codec_errors="replace",
            dimensions=(self.rows, self.cols),
            timeout=self.timeout,
            maxread=8192,
        )
        self._drain_until_quiet(timeout=0.25)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        child = self.child
        if child is None:
            return
        if child.isalive():
            child.sendcontrol("c")
            try:
                child.expect(pexpect.EOF, timeout=1.0)
            except pexpect.TIMEOUT:
                child.terminate(force=True)
        self.child = None

    def send_text(self, text: str) -> None:
        self.actions.append(f"text:{text[:80]}")
        self._require_child().send(text)
        self._drain_until_quiet(timeout=0.08)

    def send_key(self, key: str) -> None:
        sequences = {
            "enter": "\r",
            "alt-enter": "\x1b\r",
            "ctrl-j": "\n",
            "up": "\x1b[A",
            "down": "\x1b[B",
            "left": "\x1b[D",
            "right": "\x1b[C",
            "pageup": "\x1b[5~",
            "pagedown": "\x1b[6~",
            "ctrl-u": "\x15",
            "escape": "\x1b",
        }
        try:
            sequence = sequences[key]
        except KeyError as exc:
            raise ValueError(f"unknown key: {key}") from exc
        self.actions.append(f"key:{key}")
        self._require_child().send(sequence)
        self._drain_until_quiet(timeout=0.08)

    def send_keys(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.send_key(key)

    def send_mouse_wheel_up(self, *, col: int = 10, row: int = 8, count: int = 1) -> None:
        for _ in range(count):
            self.actions.append(f"mouse-wheel-up:{col},{row}")
            self._require_child().send(f"\x1b[<64;{col};{row}M")
            self._drain_until_quiet(timeout=0.05)

    def send_mouse_wheel_down(self, *, col: int = 10, row: int = 8, count: int = 1) -> None:
        for _ in range(count):
            self.actions.append(f"mouse-wheel-down:{col},{row}")
            self._require_child().send(f"\x1b[<65;{col};{row}M")
            self._drain_until_quiet(timeout=0.05)

    def send_mouse_click(self, *, col: int = 10, row: int = 8) -> None:
        self.actions.append(f"mouse-click:{col},{row}")
        self._require_child().send(f"\x1b[<0;{col};{row}M\x1b[<0;{col};{row}m")
        self._drain_until_quiet(timeout=0.05)

    def clear_input(self) -> None:
        for _ in range(4):
            self.send_key("ctrl-u")
        self._drain_until_quiet(timeout=0.05)

    def submit(self, text: str) -> None:
        self.send_text(text)
        self.send_key("enter")

    def expect_alive(self) -> None:
        child = self._require_child()
        if not child.isalive():
            raise AssertionError(self._format_failure("Expected TUI process to still be alive"))

    def assert_no_crash(self) -> None:
        current = self.screen_text()
        raw = "".join(self.raw_chunks)
        crash_markers = (
            "Unhandled exception",
            "Traceback",
            "AssertionError",
            "Press ENTER to continue",
        )
        for marker in crash_markers:
            if marker in current or marker in raw:
                raise AssertionError(self._format_failure(f"Unexpected TUI crash marker {marker!r}"))

    def screen_text(self) -> str:
        self._drain_available(timeout=0.02)
        return "\n".join(self.screen.display)

    def wait_for_text(self, text: str, *, timeout: float | None = None) -> str:
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            current = self.screen_text()
            if text in current:
                return current
            time.sleep(0.05)
        raise AssertionError(self._format_failure(f"Timed out waiting for {text!r}"))

    def assert_visible(self, text: str) -> None:
        current = self.screen_text()
        if text not in current:
            raise AssertionError(self._format_failure(f"Expected visible text {text!r}"))

    def assert_all_visible(self, items: Iterable[str]) -> None:
        for item in items:
            self.assert_visible(item)

    def _require_child(self) -> pexpect.spawn:
        if self.child is None:
            raise RuntimeError("TuiSession is not running")
        return self.child

    def _drain_until_quiet(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            before = len(self.raw_chunks)
            self._drain_available(timeout=0.02)
            if len(self.raw_chunks) == before:
                time.sleep(0.02)

    def _drain_available(self, *, timeout: float) -> None:
        child = self._require_child()
        while True:
            try:
                chunk = child.read_nonblocking(size=8192, timeout=timeout)
            except pexpect.TIMEOUT:
                return
            except pexpect.EOF:
                return
            if not chunk:
                return
            self.raw_chunks.append(chunk)
            self.stream.feed(chunk)
            timeout = 0

    def _format_failure(self, message: str) -> str:
        raw_tail = "".join(self.raw_chunks[-5:])
        return "\n".join(
            [
                message,
                "",
                "----- screen -----",
                self.screen_text(),
                "",
                "----- raw tail -----",
                raw_tail,
                "",
                "----- recent actions -----",
                "\n".join(self.actions[-20:]),
            ]
        )
