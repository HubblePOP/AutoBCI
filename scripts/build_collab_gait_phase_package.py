from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import textwrap
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts" / "gait_phase_labeler_collab.py"
PLOT_SCRIPT = ROOT / "scripts" / "plot_collab_gait_phase_head_mid_tail.py"
QC_SCRIPT = ROOT / "scripts" / "plot_collab_gait_phase_step_duration_qc.py"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "share" / "gait_phase_labeler_package"
DEFAULT_ZIP_PATH = ROOT / "artifacts" / "share" / "gait_phase_labeler_package.zip"

SELECTED_REFERENCE_INPUTS = [
    ("walk_20240701_20km_01", Path("/Volumes/Elements/bci/处理后的关节数据/20240701/walk_20km_01.xlsx")),
    ("walk_20240717_20km_03", Path("/Volumes/Elements/bci/处理后的关节数据/20240717/walk_20km_03.xlsx")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a shareable gait-phase labeler package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    return parser.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_readme() -> str:
    return textwrap.dedent(
        """\
        # Gait Phase Labeler Package

        这个包用于在 Vicon `.xlsx` 运动捕捉文件上生成步态二分类参考标签。

        输出内容和我们当前主线保持一致：
        - `reference_labels.jsonl`
        - `summary.json`

        当前规则：
        - 输入：`RHTOE_z`、`RFTOE_z`
        - 先做 `75ms` 平滑
        - 再做滞回阈值切分：`high_q=0.70`，`low_q=0.35`
        - `1 = 摆动`，`0 = 支撑`

        ## 1. 建环境

        ```bash
        bash setup_env.sh
        ```

        ## 2. 运行

        ```bash
        .venv/bin/python gait_phase_labeler.py \\
          --input-xlsx /path/to/walk_20km_01.xlsx \\
          --session-id walk_20240701_20km_01 \\
          --input-xlsx /path/to/walk_20km_03.xlsx \\
          --session-id walk_20240717_20km_03 \\
          --output-jsonl reference_labels.jsonl \\
          --summary-json summary.json
        ```

        ## 3. 输出说明

        `reference_labels.jsonl` 每一行对应一个运动文件，包含：
        - `session_id`
        - `n_samples`
        - `sample_rate_hz`
        - `toe_labels.RHTOE_z.swing_intervals`
        - `toe_labels.RFTOE_z.swing_intervals`

        `summary.json` 包含：
        - 使用的方法配置
        - 每只脚的状态统计
        - 简单质量摘要

        另外包里还带两个辅助脚本：
        - `plot_head_mid_tail_labels.py`：把每段数据的开头 / 中间 / 结尾各抽 10 秒画出来，看 `0/1` 实际怎么切
        - `step_duration_qc.py`：统计步周期和摆动时长，把明显过长的异常段标出来

        ## 4. 对照输出

        如果包里有 `reference_outputs/` 目录，其中放的是我们这边对两段选定数据已经跑好的结果。
        合作方可以先直接复现这两段，确认输出一致。
        """
    )


def _build_collab_note() -> str:
    return textwrap.dedent(
        """\
        # 合作方复现说明

        这次先只对齐两段运动数据上的步态二分类参考标签生成方式。

        对齐目标：
        - 输入文件相同
        - 输出的 `reference_labels.jsonl` 与 `summary.json` 结构相同
        - `RHTOE_z`、`RFTOE_z` 的 `swing_intervals` 与我们这边一致

        建议先复现这两段：
        - `walk_20240701_20km_01`
        - `walk_20240717_20km_03`

        运行步骤：

        ```bash
        bash setup_env.sh

        .venv/bin/python gait_phase_labeler.py \
          --input-xlsx /path/to/walk_20km_01.xlsx \
          --session-id walk_20240701_20km_01 \
          --input-xlsx /path/to/walk_20km_03.xlsx \
          --session-id walk_20240717_20km_03 \
          --output-jsonl reference_labels.jsonl \
          --summary-json summary.json
        ```

        说明：
        - 当前规则是基于 `RHTOE_z` 和 `RFTOE_z` 的绝对高度
        - 先做 `75ms` 平滑，再做滞回阈值切分
        - `1 = 摆动`，`0 = 支撑`
        - 当前不会自动接入脑电训练，只先对齐步态标签生成

        包内 `reference_outputs/` 目录放的是我们这边已经跑好的对照结果。
        建议先用同一段 `.xlsx` 跑一遍，再对照：
        - `reference_outputs/reference_labels.jsonl`
        - `reference_outputs/summary.json`
        - `reference_outputs/head_mid_tail_plots/`
        - `reference_outputs/step_duration_qc/`

        如果这两段输出一致，再继续讨论后面的数据划分和脑电训练。
        """
    )


def _build_setup_env() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail

        python3 -m venv .venv
        . .venv/bin/activate
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        """
    )


def _write_reference_outputs(package_dir: Path) -> None:
    existing = [(session_id, path) for session_id, path in SELECTED_REFERENCE_INPUTS if path.exists()]
    if not existing:
        return
    output_dir = package_dir / "reference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "reference_labels.jsonl"
    summary_json = output_dir / "summary.json"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(package_dir / "gait_phase_labeler.py"),
        "--output-jsonl",
        str(output_jsonl),
        "--summary-json",
        str(summary_json),
    ]
    for session_id, path in existing:
        cmd.extend(["--input-xlsx", str(path), "--session-id", session_id])
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)

    plot_dir = output_dir / "head_mid_tail_plots"
    plot_cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(package_dir / "plot_head_mid_tail_labels.py"),
        "--output-dir",
        str(plot_dir),
    ]
    for session_id, path in existing:
        plot_cmd.extend(["--input-xlsx", str(path), "--session-id", session_id])
    plot_completed = subprocess.run(
        plot_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if plot_completed.returncode != 0:
        raise RuntimeError(plot_completed.stderr or plot_completed.stdout)

    qc_dir = output_dir / "step_duration_qc"
    qc_cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(package_dir / "step_duration_qc.py"),
        "--output-dir",
        str(qc_dir),
    ]
    for session_id, path in existing:
        qc_cmd.extend(["--input-xlsx", str(path), "--session-id", session_id])
    qc_completed = subprocess.run(
        qc_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if qc_completed.returncode != 0:
        raise RuntimeError(qc_completed.stderr or qc_completed.stdout)

    manifest = {
        "inputs": [{"session_id": session_id, "xlsx_path": str(path)} for session_id, path in existing],
        "output_jsonl": str(output_jsonl.name),
        "summary_json": str(summary_json.name),
        "head_mid_tail_plots_dir": str(plot_dir.name),
        "step_duration_qc_dir": str(qc_dir.name),
    }
    _write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def _zip_directory(directory: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_dir():
                continue
            if "__pycache__" in path.parts:
                continue
            zf.write(path, arcname=path.relative_to(directory))


def main() -> None:
    args = parse_args()
    package_dir = args.output_dir.expanduser().resolve()
    zip_path = args.zip_path.expanduser().resolve()

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_SCRIPT, package_dir / "gait_phase_labeler.py")
    shutil.copy2(PLOT_SCRIPT, package_dir / "plot_head_mid_tail_labels.py")
    shutil.copy2(QC_SCRIPT, package_dir / "step_duration_qc.py")
    _write_text(package_dir / "requirements.txt", "numpy==2.4.4\nmatplotlib==3.10.8\n")
    _write_text(package_dir / "README.md", _build_readme())
    _write_text(package_dir / "合作方说明.md", _build_collab_note())
    setup_env_path = package_dir / "setup_env.sh"
    _write_text(setup_env_path, _build_setup_env())
    setup_env_path.chmod(0o755)
    _write_reference_outputs(package_dir)
    for cache_dir in package_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    _zip_directory(package_dir, zip_path)

    print(json.dumps({"output_dir": str(package_dir), "zip_path": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
