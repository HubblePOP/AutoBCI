from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.splits import (
    AUTOBCI_CACHE_ROOT_ENV,
    load_dataset_config,
    resolve_dataset_cache_dir,
)
from scripts.sync_session_cache_local import sync_session_cache_local


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class SessionCacheLocalTests(unittest.TestCase):
    def test_resolve_dataset_cache_dir_prefers_env_then_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "configs/datasets/demo.yaml"
            _write(
                config_path,
                textwrap.dedent(
                    """
                    dataset_name: demo_dataset
                    cache_subdir: data/cache_demo_dataset
                    cache_root: /tmp/config-cache-root
                    sessions:
                      - session_id: demo_01
                        intan_rhd: /tmp/raw/demo_01.rhd
                        vicon_csv: /tmp/raw/demo_01.csv
                        alignment: {lag_seconds: 0.0}
                      - session_id: demo_02
                        intan_rhd: /tmp/raw/demo_02.rhd
                        vicon_csv: /tmp/raw/demo_02.csv
                        alignment: {lag_seconds: 0.0}
                      - session_id: demo_03
                        intan_rhd: /tmp/raw/demo_03.rhd
                        vicon_csv: /tmp/raw/demo_03.csv
                        alignment: {lag_seconds: 0.0}
                    splits:
                      train: [demo_01]
                      val: [demo_02]
                      test: [demo_03]
                    """
                ).strip()
                + "\n",
            )
            dataset = load_dataset_config(config_path)

            self.assertEqual(
                resolve_dataset_cache_dir(
                    project_root=root,
                    cache_subdir=dataset.cache_subdir,
                    cache_root=dataset.cache_root,
                    env_root="",
                ),
                Path("/tmp/config-cache-root/cache_demo_dataset"),
            )

            original = os.environ.get(AUTOBCI_CACHE_ROOT_ENV)
            try:
                os.environ[AUTOBCI_CACHE_ROOT_ENV] = "/tmp/env-cache-root"
                self.assertEqual(
                    dataset.cache_dir(root),
                    Path("/tmp/env-cache-root/cache_demo_dataset"),
                )
            finally:
                if original is None:
                    os.environ.pop(AUTOBCI_CACHE_ROOT_ENV, None)
                else:
                    os.environ[AUTOBCI_CACHE_ROOT_ENV] = original

    def test_sync_session_cache_local_copies_then_reuses_dataset_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "configs/datasets/demo.yaml"
            _write(
                config_path,
                textwrap.dedent(
                    """
                    dataset_name: demo_dataset
                    cache_subdir: data/cache_demo_dataset
                    sessions:
                      - session_id: demo_01
                        intan_rhd: /tmp/raw/demo_01.rhd
                        vicon_csv: /tmp/raw/demo_01.csv
                        alignment: {lag_seconds: 0.0}
                      - session_id: demo_02
                        intan_rhd: /tmp/raw/demo_02.rhd
                        vicon_csv: /tmp/raw/demo_02.csv
                        alignment: {lag_seconds: 0.0}
                      - session_id: demo_03
                        intan_rhd: /tmp/raw/demo_03.rhd
                        vicon_csv: /tmp/raw/demo_03.csv
                        alignment: {lag_seconds: 0.0}
                    splits:
                      train: [demo_01]
                      val: [demo_02]
                      test: [demo_03]
                    """
                ).strip()
                + "\n",
            )
            source_dir = root / "data/cache_demo_dataset"
            _write_bytes(source_dir / "demo_01.npz", b"demo-01")
            _write_bytes(source_dir / "demo_02.npz", b"demo-02")
            _write_bytes(source_dir / "demo_03.npz", b"demo-03")

            target_root = root / "local-cache"
            first = sync_session_cache_local(
                dataset_config=config_path,
                project_root=root,
                target_cache_root=target_root,
            )
            second = sync_session_cache_local(
                dataset_config=config_path,
                project_root=root,
                target_cache_root=target_root,
            )

            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["copied_count"], 3)
            self.assertEqual(first["reused_count"], 0)
            self.assertEqual(second["copied_count"], 0)
            self.assertEqual(second["reused_count"], 3)
            self.assertTrue((target_root / "cache_demo_dataset" / "demo_01.npz").exists())
            self.assertTrue((target_root / "cache_demo_dataset" / "demo_02.npz").exists())
            self.assertTrue((target_root / "cache_demo_dataset" / "demo_03.npz").exists())


if __name__ == "__main__":
    unittest.main()
