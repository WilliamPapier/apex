from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from research import run_suite


class TestResearchSuiteCli(unittest.TestCase):
    def test_completed_full_year_run_archives_before_publishing_latest(self):
        now = datetime(2026, 10, 25, tzinfo=timezone.utc)
        report = Mock(
            all_ok=True,
            first_timestamp=now - timedelta(days=364),
            last_timestamp=now,
        )
        report.to_text.return_value = "research body"
        saves = Mock()
        saves.archive.return_value = Path(r"logs\research-full-year-20261027T123456000000Z.txt")
        context = object()
        output = StringIO()
        with (
            patch.object(run_suite, "load_context", return_value=context),
            patch.object(run_suite, "default_agents", return_value=[]),
            patch.object(run_suite, "run_agents", return_value=report),
            patch.object(run_suite, "save_run_report", side_effect=lambda *args: saves.archive(*args)),
            patch.object(run_suite, "save_latest_report", side_effect=lambda text: saves.latest(text)),
            redirect_stdout(output),
        ):
            result = run_suite.main([r"data\historical\full-year.csv"])

        self.assertEqual(result, 0)
        self.assertEqual(
            [entry[0] for entry in saves.mock_calls],
            ["archive", "latest"],
        )
        archived_text = saves.archive.call_args.args[0]
        self.assertIn("run_runtime_seconds:", archived_text)
        self.assertIn("run_archive: logs\\research-full-year-", archived_text)
        self.assertIn("research body", archived_text)
        self.assertEqual(saves.latest.call_args.args[0], archived_text)
        self.assertIn("Archived report:", output.getvalue())

    def test_failed_dataset_load_does_not_publish_or_archive(self):
        output = StringIO()
        errors = StringIO()
        with (
            patch.object(run_suite, "load_context", side_effect=OSError("missing dataset")),
            patch.object(run_suite, "save_run_report") as archive,
            patch.object(run_suite, "save_latest_report") as latest,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = run_suite.main([r"data\historical\missing.csv"])

        self.assertEqual(result, 2)
        archive.assert_not_called()
        latest.assert_not_called()
        self.assertIn("missing dataset", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
