from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOCAL_CACHE_BASE_ROOT = Path.home() / "Library" / "Application Support" / "AutoBci" / "session_cache"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    cache_subdir: Path
    sessions: tuple[str, ...]
    cache_root: Path | None = None


def _cache_dir_name(cache_subdir: str | Path) -> str:
    return Path(cache_subdir).name


def _resolve_cache_base_dir(
    base_dir: str | Path | None,
    *,
    cache_subdir: str | Path,
) -> Path:
    if base_dir is None:
        raise ValueError("cache base dir is required")
    base_path = Path(base_dir).expanduser()
    if base_path.name == _cache_dir_name(cache_subdir):
        return base_path
    return base_path / _cache_dir_name(cache_subdir)


def resolve_dataset_cache_dir(
    *,
    project_root: str | Path,
    cache_subdir: str | Path,
    cache_root: str | Path | None = None,
    env_root: str | Path | None = None,
) -> Path:
    if env_root:
        return _resolve_cache_base_dir(env_root, cache_subdir=cache_subdir)
    if cache_root is not None:
        return _resolve_cache_base_dir(cache_root, cache_subdir=cache_subdir)
    return (Path(project_root) / Path(cache_subdir)).resolve()


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    sessions = tuple(
        str(item["session_id"])
        for item in payload.get("sessions", [])
        if item and item.get("session_id")
    )
    cache_root_value = payload.get("cache_root")
    return DatasetConfig(
        dataset_name=str(payload["dataset_name"]),
        cache_subdir=Path(str(payload["cache_subdir"])),
        sessions=sessions,
        cache_root=Path(str(cache_root_value)).expanduser() if cache_root_value else None,
    )


def _same_cache_file(source: Path, target: Path) -> bool:
    if not target.exists():
        return False
    source_stat = source.stat()
    target_stat = target.stat()
    return (
        source_stat.st_size == target_stat.st_size
        and int(source_stat.st_mtime) == int(target_stat.st_mtime)
    )


def build_cache_sync_plan(
    *,
    dataset: DatasetConfig,
    project_root: str | Path,
    source_cache_root: str | Path | None = None,
    target_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    source_cache_dir = resolve_dataset_cache_dir(
        project_root=project_root,
        cache_subdir=dataset.cache_subdir,
        cache_root=source_cache_root or dataset.cache_root,
        env_root="",
    )
    target_base_root = Path(target_cache_root).expanduser() if target_cache_root else DEFAULT_LOCAL_CACHE_BASE_ROOT
    target_cache_dir = resolve_dataset_cache_dir(
        project_root=project_root,
        cache_subdir=dataset.cache_subdir,
        cache_root=target_base_root,
        env_root="",
    )
    files: list[dict[str, str]] = []
    for session_id in dataset.sessions:
        files.append(
            {
                "session_id": session_id,
                "source": str(source_cache_dir / f"{session_id}.npz"),
                "target": str(target_cache_dir / f"{session_id}.npz"),
            }
        )
    return {
        "dataset_name": dataset.dataset_name,
        "source_cache_dir": str(source_cache_dir),
        "target_cache_dir": str(target_cache_dir),
        "files": files,
    }


def sync_session_cache_local(
    *,
    dataset_config: str | Path,
    project_root: str | Path = ROOT,
    source_cache_root: str | Path | None = None,
    target_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = load_dataset_config(dataset_config)
    plan = build_cache_sync_plan(
        dataset=dataset,
        project_root=project_root,
        source_cache_root=source_cache_root,
        target_cache_root=target_cache_root,
    )
    source_cache_dir = Path(plan["source_cache_dir"])
    target_cache_dir = Path(plan["target_cache_dir"])
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    reused_files: list[str] = []
    missing_source_files: list[str] = []
    pending_copy: list[tuple[str, Path, Path]] = []

    for item in plan["files"]:
        source = Path(item["source"])
        target = Path(item["target"])
        if target.exists():
            try:
                source_exists = source.exists()
            except PermissionError:
                reused_files.append(item["session_id"])
                continue
            if not source_exists or _same_cache_file(source, target):
                reused_files.append(item["session_id"])
                continue
        pending_copy.append((item["session_id"], source, target))

    if not pending_copy:
        return {
            "status": "ok",
            "dataset_name": dataset.dataset_name,
            "source_cache_dir": str(source_cache_dir),
            "target_cache_dir": str(target_cache_dir),
            "copied_count": 0,
            "reused_count": len(reused_files),
            "copied_sessions": [],
            "reused_sessions": reused_files,
        }

    try:
        source_cache_dir.stat()
    except PermissionError as exc:
        raise RuntimeError(f"data_access_blocked: 无法读取源 cache 目录：{source_cache_dir}") from exc

    for session_id, source, target in pending_copy:
        try:
            exists = source.exists()
        except PermissionError as exc:
            raise RuntimeError(f"data_access_blocked: 无法读取源 cache 文件：{source}") from exc
        if not exists:
            missing_source_files.append(session_id)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
        except PermissionError as exc:
            raise RuntimeError(f"data_access_blocked: 无法复制源 cache 文件：{source}") from exc
        copied_files.append(session_id)

    if missing_source_files:
        raise RuntimeError(
            "missing_local_cache: 源 cache 缺少这些试次："
            + ", ".join(sorted(missing_source_files))
        )

    return {
        "status": "ok",
        "dataset_name": dataset.dataset_name,
        "source_cache_dir": str(source_cache_dir),
        "target_cache_dir": str(target_cache_dir),
        "copied_count": len(copied_files),
        "reused_count": len(reused_files),
        "copied_sessions": copied_files,
        "reused_sessions": reused_files,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把当前数据集需要的 session cache 同步到本机可读目录。")
    parser.add_argument("--dataset-config", required=True, help="数据集配置路径")
    parser.add_argument("--source-cache-root", help="源 cache 根目录；默认按当前数据集 cache_subdir 解析")
    parser.add_argument(
        "--target-cache-root",
        help=f"目标 cache 根目录；默认写到 {DEFAULT_LOCAL_CACHE_BASE_ROOT}",
    )
    parser.add_argument("--project-root", default=str(ROOT), help="AutoBci 仓库根目录")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = sync_session_cache_local(
            dataset_config=args.dataset_config,
            project_root=args.project_root,
            source_cache_root=args.source_cache_root,
            target_cache_root=args.target_cache_root,
        )
    except RuntimeError as exc:
        payload = {"status": "error", "reason": str(exc)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(payload["reason"])
        return 1

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(
            f"已同步 {summary['dataset_name']}：复制 {summary['copied_count']} 个，复用 {summary['reused_count']} 个，"
            f"目标目录 {summary['target_cache_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
