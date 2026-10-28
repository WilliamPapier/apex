import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from data.historical_data_loader import HistoricalDataError
from research.liquidity_sweeps import SweepConfig, analyze_liquidity_sweeps
from research.market_structure import StructureConfig, analyze_market_structure
from research.pipeline import (
    PENDING_STRATEGY_DECISIONS,
    PipelineConfig,
    analyze_candles,
    main,
    render_report,
    run_research,
)
from data.historical_data_loader import load_candles_csv


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
HEADER = ["timestamp_utc", "open", "high", "low", "close", "tick_volume"]
CFG = PipelineConfig(StructureConfig(1, 1), SweepConfig(), (1, 2), 2)

# (minute offset, high, close); all lows are 8. One confirmed swing high (14),
# swept by the last candle (high 15, close 13.5).
SWEEP_ROWS = [(0, 10, 9), (5, 14, 11), (10, 12, 10), (15, 15, 13.5)]
GAP_ROWS = [(0, 10, 9), (5, 14, 11), (65, 12, 10), (130, 15, 13.5)]


def write_csv(directory, rows, name="sample.csv", base=BASE):
    path = Path(directory) / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for minute, high, close in rows:
            stamp = (base + timedelta(minutes=minute)).isoformat()
            writer.writerow([stamp, close, high, 8, close, 1])
    return path


class TestPipeline(unittest.TestCase):

    def run_rows(self, rows, config=CFG, base=BASE):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, rows, base=base)
            return run_research(path, config)

    def test_end_to_end_counts_timestamps_and_features(self):
        report = self.run_rows(SWEEP_ROWS)

        self.assertEqual(report.data.candle_count, 4)
        self.assertEqual(report.data.first_open, BASE)
        self.assertEqual(report.data.last_open, BASE + timedelta(minutes=15))
        self.assertTrue(report.data.strictly_ascending_unique_utc)
        self.assertEqual(
            (report.data.five_minute_intervals, report.data.longer_intervals), (3, 0)
        )
        self.assertEqual((report.swing_highs, report.swing_lows), (1, 0))
        self.assertEqual((report.upward_breaks, report.downward_breaks), (0, 0))
        self.assertEqual((report.high_side_sweeps, report.low_side_sweeps), (1, 0))
        self.assertEqual(report.swept_swings, 1)
        self.assertEqual(
            report.swing_availability.first_available_at, BASE + timedelta(minutes=15)
        )
        self.assertEqual(
            report.sweep_availability.last_available_at, BASE + timedelta(minutes=20)
        )
        self.assertEqual(report.break_availability.count, 0)
        self.assertIsNone(report.break_availability.first_available_at)
        self.assertEqual(
            report.counts_by_utc_date, ((date(2025, 9, 1), 1, 0, 0, 0, 1, 0),)
        )
        # ATR(2): TRs 2, 6, 4, 7 -> 4, 4, 5.5; sweep penetration 1 at ATR 5.5.
        self.assertEqual(report.features.candles_with_atr, 3)
        self.assertEqual(report.features.sweeps_with_atr, 1)
        self.assertAlmostEqual(report.features.median_penetration_to_atr, 1 / 5.5)
        self.assertEqual(report.features.candles_with_full_volatility_window, 2)

    def test_matches_individual_modules_on_a_larger_series(self):
        rows = []
        price = 100.0
        for i in range(120):
            price += ((i * 7) % 11 - 5) * 0.3
            rows.append((5 * i, price + 0.8, price))
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, rows)
            candles = load_candles_csv(path)
            report = run_research(path, CFG)

        structure = analyze_market_structure(candles, CFG.structure)
        sweeps = analyze_liquidity_sweeps(candles, CFG.structure, CFG.sweep)
        self.assertEqual(report.swing_highs, len(structure.swing_highs))
        self.assertEqual(report.swing_lows, len(structure.swing_lows))
        self.assertEqual(report.upward_breaks + report.downward_breaks, len(structure.breaks))
        self.assertEqual(report.high_side_sweeps + report.low_side_sweeps, len(sweeps))
        self.assertEqual(report.sample_sweeps, sweeps[:5])
        totals = [sum(row[i] for row in report.counts_by_utc_date) for i in range(1, 7)]
        self.assertEqual(
            totals,
            [
                report.swing_highs,
                report.swing_lows,
                report.upward_breaks,
                report.downward_breaks,
                report.high_side_sweeps,
                report.low_side_sweeps,
            ],
        )

    def test_gaps_are_reported_without_filling_or_assuming_missing_candles(self):
        report = self.run_rows(GAP_ROWS)

        self.assertEqual(report.data.candle_count, 4)
        self.assertEqual(report.data.longer_intervals, 2)
        self.assertEqual(report.data.longest_interval_seconds, 65 * 60)
        self.assertEqual(report.swings_spanning_long_interval, 1)
        self.assertEqual(report.sweeps_spanning_long_interval, 1)
        self.assertEqual(
            report.sweep_availability.first_available_at, BASE + timedelta(minutes=135)
        )
        self.assertEqual(report.high_side_sweeps, 1)

    def test_counts_use_utc_date_of_event_candle_open(self):
        late = datetime(2025, 9, 1, 23, 50, tzinfo=UTC)

        report = self.run_rows(SWEEP_ROWS, base=late)

        self.assertEqual(
            [(row[0], row[1], row[5]) for row in report.counts_by_utc_date],
            [(date(2025, 9, 2), 0, 1), (date(2025, 9, 1), 1, 0)][::-1],
        )

    def test_warnings_flag_material_choices_and_unresolved_rules_are_listed(self):
        config = PipelineConfig(
            StructureConfig(1, 1, equal_policy="inclusive", break_basis="wick"),
            SweepConfig(),
            (1,),
            2,
        )

        report = self.run_rows(SWEEP_ROWS, config)
        text = " ".join(report.warnings)

        for fragment in ("inclusive", "wick", "no expiry", "min_penetration=0", "not MSS"):
            self.assertIn(fragment, text)
        self.assertEqual(report.pending_strategy_decisions, PENDING_STRATEGY_DECISIONS)
        strict = self.run_rows(SWEEP_ROWS)
        self.assertNotIn("inclusive", " ".join(strict.warnings))
        quiet = self.run_rows(
            SWEEP_ROWS,
            PipelineConfig(StructureConfig(1, 1), SweepConfig(first_sweep_only=True), (1,), 2),
        )
        self.assertNotIn("no expiry", " ".join(quiet.warnings))

    def test_report_is_deterministic_and_separates_observations_from_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, SWEEP_ROWS)
            first = render_report(run_research(path, CFG))
            second = render_report(run_research(path, CFG))

        self.assertEqual(first, second)
        observations, notes = first.split("== INTERPRETATION NOTES (not observations) ==")
        self.assertIn("OBSERVATIONS: EVENTS", observations)
        self.assertIn("candidate sweeps: high_side=1 low_side=0", observations)
        self.assertNotIn("warning:", observations)
        self.assertNotIn("unresolved:", observations)
        self.assertIn("warning:", notes)
        self.assertIn("unresolved: Entry timing", notes)
        self.assertIn("not a trading strategy", first)

    def test_prefix_reports_never_contain_events_available_later(self):
        full = self.run_rows(SWEEP_ROWS)
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, SWEEP_ROWS[:3])
            prefix = run_research(path, CFG)

        self.assertEqual((prefix.swing_highs, prefix.high_side_sweeps), (1, 0))
        self.assertLessEqual(
            prefix.swing_availability.last_available_at,
            full.swing_availability.last_available_at,
        )

    def test_invalid_inputs_fail_loudly_and_source_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            good = write_csv(directory, SWEEP_ROWS)
            before = good.read_bytes()
            run_research(good, CFG)
            self.assertEqual(good.read_bytes(), before)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["sample.csv"])

            bad = write_csv(directory, [(0, 10, 9), (0, 11, 9)], name="dup.csv")
            with self.assertRaisesRegex(HistoricalDataError, "duplicate"):
                run_research(bad, CFG)
            with self.assertRaises(HistoricalDataError):
                run_research(_empty_csv(directory), CFG)
        with self.assertRaises(ValueError):
            analyze_candles((), CFG)
        with self.assertRaises(ValueError):
            PipelineConfig(structure="bad")
        with self.assertRaises(ValueError):
            PipelineConfig(sweep="bad")

    def test_cli_prints_report_with_explicit_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, SWEEP_ROWS)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main([str(path), "--left", "1", "--right", "1", "--first-sweep-only"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("left=1 right=1 equal_policy=strict", output)
        self.assertIn("first_sweep_only=True", output)
        self.assertIn("candidate sweeps: high_side=1", output)


def _empty_csv(directory):
    path = Path(directory) / "empty.csv"
    path.write_text("", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
