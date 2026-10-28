import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dashboard import HOST, create_server, dashboard_snapshot, render_dashboard
from research.report_store import LATEST_REPORT_PATH, save_latest_report, save_run_report


class TestResearchReportStore(unittest.TestCase):
    def test_saves_report_and_refuses_historical_data_directory(self):
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "reports" / "latest.txt"
            self.assertEqual(save_latest_report("research result", destination), destination.resolve())
            self.assertEqual(destination.read_text(encoding="utf-8"), "research result")
            with self.assertRaises(ValueError):
                save_latest_report("not allowed", Path(temporary) / "historical" / "original.csv")
        with self.assertRaises(ValueError):
            save_latest_report(" ")

    def test_run_report_is_exclusive_and_latest_is_atomically_replaced(self):
        run_id = "20261027T123456123456Z"
        with TemporaryDirectory() as temporary:
            archive = save_run_report("full-year result", "full-year", run_id, temporary)
            self.assertEqual(archive.name, f"research-full-year-{run_id}.txt")
            self.assertEqual(archive.read_text(encoding="utf-8"), "full-year result")
            with self.assertRaises(FileExistsError):
                save_run_report("replacement", "full-year", run_id, temporary)
            latest = save_latest_report("complete report", Path(temporary) / "latest.txt")
            self.assertEqual(latest.read_text(encoding="utf-8"), "complete report")
            self.assertEqual(sorted(path.name for path in Path(temporary).iterdir()), sorted([
                archive.name, "latest.txt",
            ]))
        for invalid_label, invalid_run_id in (("../outside", run_id), ("research", "not-a-run-id")):
            with self.assertRaises(ValueError):
                save_run_report("report", invalid_label, invalid_run_id)


class TestDashboard(unittest.TestCase):
    REPORT = """generated_at: 2026-10-27T18:00:00+00:00

APEX offline research suite
source: local.csv
candles: 100  range: 2026-09-01T00:00:00+00:00 -> 2026-09-01T08:15:00+00:00

== market_regime [ok] ==
No result.

== candidate_strategy [ok] ==
Promotion gate: NOT met
Pooled out-of-sample: trades=2 max_drawdown=1.00%
Out-of-sample breakdown (net of placeholder costs):
  by month: 2026-09: n=2 exp=0.1R pf=1.1
"""

    def test_unavailable_sources_are_explicit_and_never_fabricated(self):
        with TemporaryDirectory() as temporary:
            snapshot = dashboard_snapshot(temporary)
        self.assertFalse(snapshot["system"]["live_trading_enabled"])
        self.assertFalse(snapshot["research"]["available"])
        self.assertEqual(snapshot["research"]["agents"], [])
        self.assertEqual(snapshot["simulation"]["daily_performance"], "Not available")
        self.assertEqual(snapshot["simulation"]["weekly_performance"], "Not available")
        self.assertIn("Not available", snapshot["simulation"]["positions_and_trade_history"])
        self.assertEqual(snapshot["simulation"]["monthly_performance"], "Not available")
        self.assertIn("No saved research report", snapshot["health"]["status"])

    def test_saved_report_populates_only_reported_research_results(self):
        with TemporaryDirectory() as temporary:
            report_path = Path(temporary) / LATEST_REPORT_PATH.relative_to(LATEST_REPORT_PATH.parents[1])
            report_path.parent.mkdir(parents=True)
            report_path.write_text(self.REPORT, encoding="utf-8")
            snapshot = dashboard_snapshot(temporary)
        self.assertEqual(snapshot["research"]["source"], "local.csv")
        self.assertEqual(snapshot["research"]["candle_count"], 100)
        self.assertEqual(snapshot["research"]["agents"], [
            {"name": "market_regime", "status": "ok"},
            {"name": "candidate_strategy", "status": "ok"},
        ])
        self.assertEqual(snapshot["candidates"]["validation"], "Promotion gate: NOT met")
        self.assertIn("candidate_strategy", snapshot["candidates"]["details"])
        self.assertIn("2026-09:", snapshot["simulation"]["monthly_performance"])
        self.assertIn("max_drawdown=1.00%", snapshot["simulation"]["drawdown"])
        self.assertIn("Live trading: DISABLED", render_dashboard(snapshot))

    def test_agent_errors_are_reported_and_report_content_is_escaped(self):
        report = self.REPORT.replace(
            "== market_regime [ok] ==\nNo result.",
            "== market_regime [error] ==\nERROR: ValueError: <script>failed</script>",
        )
        with TemporaryDirectory() as temporary:
            report_path = Path(temporary) / LATEST_REPORT_PATH.relative_to(LATEST_REPORT_PATH.parents[1])
            report_path.parent.mkdir(parents=True)
            report_path.write_text(report, encoding="utf-8")
            snapshot = dashboard_snapshot(temporary)
        self.assertEqual(
            snapshot["research"]["errors"],
            ["market_regime: ValueError: <script>failed</script>"],
        )
        self.assertIn("agent errors", snapshot["health"]["status"])
        rendered = render_dashboard(snapshot)
        self.assertNotIn("<script>failed</script>", rendered)
        self.assertIn("&lt;script&gt;failed&lt;/script&gt;", rendered)

    def test_local_http_startup_serves_dashboard_and_read_only_status(self):
        with TemporaryDirectory() as temporary:
            server = create_server(0, temporary)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urlopen(f"http://{HOST}:{port}/", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn("Local read-only dashboard", response.read().decode("utf-8"))
                with urlopen(f"http://{HOST}:{port}/api/status", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.assertFalse(payload["system"]["live_trading_enabled"])
                with self.assertRaises(HTTPError) as raised:
                    urlopen(Request(f"http://{HOST}:{port}/", method="POST"), timeout=5)
                self.assertEqual(raised.exception.code, 405)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
