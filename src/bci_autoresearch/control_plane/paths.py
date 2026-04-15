from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


AUTOBCI_ROOT_ENV = "AUTOBCI_ROOT"
DEFAULT_CACHE_ROOT_ENV = "AUTOBCI_CACHE_ROOT"
DEFAULT_LOCAL_CACHE_ROOT = Path.home() / "Library" / "Application Support" / "AutoBci" / "session_cache"


@dataclass(frozen=True)
class AutoBciControlPlanePaths:
    repo_root: Path
    monitor_dir: Path
    artifacts_dir: Path
    autoresearch_status: Path
    runtime_state: Path
    process_registry: Path
    mission_process_registry: Path
    runtime_track_manifest: Path
    runtime_overrides_dir: Path
    experiment_ledger: Path
    research_queries: Path
    research_evidence: Path
    topics_inbox: Path
    retrieval_packets_dir: Path
    decision_packets_dir: Path
    judgment_updates: Path
    hypothesis_log: Path
    supervisor_events: Path
    memory_events: Path
    current_strategy: Path
    research_tree: Path
    program_doc: Path
    program_current: Path
    program_archive_dir: Path
    track_manifest: Path
    track_structure_manifest: Path
    direction_tags: Path
    launch_logs_dir: Path
    execution_worktrees_root: Path
    director_reasoning: Path
    dashboard_url: str


def get_control_plane_paths(repo_root: str | Path | None = None) -> AutoBciControlPlanePaths:
    root = Path(repo_root or os.environ.get(AUTOBCI_ROOT_ENV) or Path(__file__).resolve().parents[3]).expanduser().resolve()
    monitor_dir = root / "artifacts" / "monitor"
    return AutoBciControlPlanePaths(
        repo_root=root,
        monitor_dir=monitor_dir,
        artifacts_dir=root / "artifacts",
        autoresearch_status=monitor_dir / "autoresearch_status.json",
        runtime_state=monitor_dir / "autobci_remote_runtime.json",
        process_registry=monitor_dir / "process_registry.json",
        mission_process_registry=monitor_dir / "mission_process_registry.json",
        runtime_track_manifest=monitor_dir / "autobci_runtime_tracks.json",
        runtime_overrides_dir=monitor_dir / "runtime_overrides",
        experiment_ledger=monitor_dir / "experiment_ledger.jsonl",
        research_queries=monitor_dir / "research_queries.jsonl",
        research_evidence=monitor_dir / "research_evidence.jsonl",
        topics_inbox=monitor_dir / "topics.inbox.json",
        retrieval_packets_dir=monitor_dir / "retrieval_packets",
        decision_packets_dir=monitor_dir / "decision_packets",
        judgment_updates=monitor_dir / "judgment_updates.jsonl",
        hypothesis_log=monitor_dir / "hypothesis_log.jsonl",
        supervisor_events=monitor_dir / "supervisor_events.jsonl",
        memory_events=monitor_dir / "memory_events.jsonl",
        current_strategy=root / "memory" / "current_strategy.md",
        research_tree=root / "memory" / "hermes_research_tree.md",
        program_doc=root / "tools" / "autoresearch" / "program.md",
        program_current=root / "tools" / "autoresearch" / "program.current.md",
        program_archive_dir=root / "memory" / "programs",
        track_manifest=root / "tools" / "autoresearch" / "tracks.current.json",
        track_structure_manifest=root / "tools" / "autoresearch" / "tracks.structure.json",
        direction_tags=root / "configs" / "control_plane_direction_tags.json",
        launch_logs_dir=monitor_dir / "control_plane_launch_logs",
        execution_worktrees_root=Path("/Users/mac/Code/.hermes-worktrees/autobci"),
        director_reasoning=monitor_dir / "director_reasoning.json",
        dashboard_url="http://127.0.0.1:8878/",
    )
