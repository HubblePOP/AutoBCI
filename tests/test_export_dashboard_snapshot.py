from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_dashboard_snapshot import export_dashboard_snapshot


class ExportDashboardSnapshotTests(unittest.TestCase):
    def test_export_dashboard_snapshot_copies_index_assets_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dashboard_dir = root / "dashboard"
            assets_dir = dashboard_dir / "assets"
            assets_dir.mkdir(parents=True)
            (dashboard_dir / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
            (assets_dir / "style.css").write_text("body {}", encoding="utf-8")

            output_dir = root / "public"

            export_dashboard_snapshot(
                output_dir=output_dir,
                dashboard_dir=dashboard_dir,
                status_payload={"exported_at": "2026-04-18T12:00:00Z", "mission_control": {"summary": "ok"}},
            )

            self.assertEqual((output_dir / "index.html").read_text(encoding="utf-8"), "<html>dashboard</html>")
            self.assertEqual((output_dir / "assets" / "style.css").read_text(encoding="utf-8"), "body {}")
            status_payload = json.loads((output_dir / "status.snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(status_payload["exported_at"], "2026-04-18T12:00:00Z")
            self.assertEqual(status_payload["mission_control"]["summary"], "ok")


if __name__ == "__main__":
    unittest.main()
