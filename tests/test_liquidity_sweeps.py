import csv
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from core.market_data import Candle
from research.liquidity_sweeps import (
    SweepConfig,
    analyze_liquidity_sweeps,
    detect_liquidity_sweeps,
    diagnose_historical_csv,
    diagnose_liquidity_sweeps,
)
from research.market_structure import (
    StructureConfig,
    analyze_market_structure,
    find_confirmed_swing_highs,
)


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)
CFG1 = StructureConfig(1, 1)


def at(minute, high, low, close=None, base=BASE):
    close = (high + low) / 2 if close is None else close
    return Candle(base + timedelta(minutes=minute), close, high, low, close, 1)


def highs_series(rows, base=BASE):
    """Rows are (high, close); lows are all 8, so no low swings form."""
    return tuple(
        at(5 * i, h, 8, c, base) for i, (h, c) in enumerate(rows)
    )


def lows_series(rows):
    """Rows are (low, close); highs are all 12, so no high swings form."""
    return tuple(at(5 * i, 12, low, c) for i, (low, c) in enumerate(rows))


HIGH_SWEEP = highs_series([(10, 9), (14, 11), (12, 10), (15, 13.5)])
MULTI = highs_series([(10, 9), (14, 11), (12, 10), (13, 11), (12, 10), (15, 12.5)])


class TestSweepDefinitions(unittest.TestCase):

    def test_high_side_sweep_fields_and_timestamps(self):
        events = analyze_liquidity_sweeps(HIGH_SWEEP, CFG1)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.side, "high")
        self.assertEqual(event.sweep_index, 3)
        self.assertEqual(event.sweep_timestamp, BASE + timedelta(minutes=15))
        self.assertEqual(event.available_at, BASE + timedelta(minutes=20))
        self.assertEqual(
            (event.level, event.sweep_extreme, event.close, event.penetration),
            (14, 15, 13.5, 1),
        )
        self.assertEqual(event.swing.candle_index, 1)
        self.assertEqual(event.swing.timestamp, BASE + timedelta(minutes=5))
        self.assertEqual(event.swing.confirmed_at, BASE + timedelta(minutes=15))

    def test_low_side_sweep(self):
        candles = lows_series([(10, 11), (6, 9), (9, 10), (5, 7)])

        events = analyze_liquidity_sweeps(candles, CFG1)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event.side, event.sweep_index), ("low", 3))
        self.assertEqual(
            (event.level, event.sweep_extreme, event.close, event.penetration),
            (6, 5, 7, 1),
        )
        self.assertEqual(event.available_at, BASE + timedelta(minutes=20))

    def test_close_beyond_the_level_is_not_a_sweep(self):
        high = highs_series([(10, 9), (14, 11), (12, 10), (15, 14.5)])
        low = lows_series([(10, 11), (6, 9), (9, 10), (5, 5.5)])

        self.assertEqual(analyze_liquidity_sweeps(high, CFG1), ())
        self.assertEqual(analyze_liquidity_sweeps(low, CFG1), ())

    def test_exact_touch_and_close_at_level_are_not_sweeps(self):
        touch_high = highs_series([(10, 9), (14, 11), (12, 10), (14, 13)])
        touch_low = lows_series([(10, 11), (6, 9), (9, 10), (6, 7)])
        close_at_level = highs_series([(10, 9), (14, 11), (12, 10), (15, 14)])

        for candles in (touch_high, touch_low, close_at_level):
            with self.subTest(candles=candles[-1]):
                self.assertEqual(analyze_liquidity_sweeps(candles, CFG1), ())

    def test_min_penetration_is_strict_and_default_does_not_weaken(self):
        self.assertEqual(len(analyze_liquidity_sweeps(HIGH_SWEEP, CFG1, SweepConfig(0.5))), 1)
        self.assertEqual(analyze_liquidity_sweeps(HIGH_SWEEP, CFG1, SweepConfig(1.0)), ())
        self.assertEqual(len(analyze_liquidity_sweeps(HIGH_SWEEP, CFG1, SweepConfig())), 1)

    def test_one_candle_can_sweep_both_sides_high_first(self):
        candles = (
            at(0, 10, 8),
            at(5, 14, 5, 9),
            at(10, 12, 8),
            at(15, 15, 3, 9),
        )

        events = analyze_liquidity_sweeps(candles, CFG1)

        self.assertEqual([(e.side, e.sweep_index, e.level) for e in events],
                         [("high", 3, 14), ("low", 3, 5)])
        self.assertEqual([e.penetration for e in events], [1, 2])


class TestAvailabilityAndConfirmation(unittest.TestCase):

    def test_unconfirmed_swing_cannot_be_swept(self):
        candles = highs_series([(10, 9), (14, 11), (12, 10)])
        structure = analyze_market_structure(candles, CFG1)

        self.assertEqual(len(structure.swing_highs), 1)
        self.assertEqual(analyze_liquidity_sweeps(candles[:2], CFG1), ())
        sweep_candle = highs_series([(10, 9), (14, 11), (15, 13.5), (12, 10), (11, 10)])
        events = analyze_liquidity_sweeps(sweep_candle, CFG1)
        # The candle that exceeds 14 inside the right window prevents confirmation.
        self.assertNotIn(14, [e.level for e in events])

    def test_sweep_on_first_candle_after_confirmation_is_allowed_not_earlier(self):
        config = StructureConfig(1, 2)
        candles = highs_series([(10, 9), (14, 11), (12, 10), (11, 10), (15, 13.5)])

        events = analyze_liquidity_sweeps(candles, config)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].swing.confirmation_index, 3)
        self.assertEqual(events[0].sweep_timestamp, events[0].swing.confirmed_at)
        self.assertEqual(events[0].sweep_index, 4)

    def test_every_event_follows_its_swings_confirmation(self):
        candles = _random_candles(11, 400)
        events = analyze_liquidity_sweeps(candles, StructureConfig(2, 2))

        self.assertGreater(len(events), 10)
        for event in events:
            self.assertGreater(event.sweep_index, event.swing.confirmation_index)
            self.assertGreaterEqual(event.sweep_timestamp, event.swing.confirmed_at)
            self.assertGreater(event.available_at, event.sweep_timestamp)
            self.assertEqual(event.available_at, event.sweep_timestamp + STEP)

    def test_gaps_are_not_filled_and_availability_uses_the_sweep_candle(self):
        candles = (
            at(0, 10, 8, 9),
            at(5, 14, 8, 11),
            at(65, 12, 8, 10),
            at(130, 15, 8, 13.5),
        )
        before = tuple(candles)

        events = analyze_liquidity_sweeps(candles, CFG1)

        self.assertEqual(candles, before)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].swing.confirmed_at, BASE + timedelta(minutes=70))
        self.assertEqual(events[0].sweep_timestamp, BASE + timedelta(minutes=130))
        self.assertEqual(events[0].available_at, BASE + timedelta(minutes=135))
        self.assertEqual(analyze_liquidity_sweeps(candles[:3], CFG1), ())

    def test_offset_timestamps_give_equal_events_in_utc(self):
        offset = timezone(timedelta(hours=2))
        shifted = tuple(
            Candle(c.timestamp.astimezone(offset), c.open, c.high, c.low, c.close, 1)
            for c in HIGH_SWEEP
        )

        event = analyze_liquidity_sweeps(shifted, CFG1)[0]
        reference = analyze_liquidity_sweeps(HIGH_SWEEP, CFG1)[0]

        self.assertEqual(event.available_at, reference.available_at)
        self.assertEqual(event.sweep_timestamp, reference.sweep_timestamp)
        self.assertEqual(event.available_at.utcoffset(), timedelta(0))


class TestOrderingAndDuplicates(unittest.TestCase):

    def test_multiple_levels_on_one_candle_are_ordered_by_swing(self):
        events = analyze_liquidity_sweeps(MULTI, CFG1)

        self.assertEqual([(e.sweep_index, e.swing.candle_index) for e in events],
                         [(5, 1), (5, 3)])
        self.assertEqual([e.level for e in events], [14, 13])
        self.assertEqual([e.penetration for e in events], [1, 2])

    def test_no_duplicate_swing_candle_pairs_and_repeat_sweeps_are_separate(self):
        candles = MULTI + (at(30, 15.5, 8, 12),)

        events = analyze_liquidity_sweeps(candles, CFG1)

        keys = [(e.side, e.swing.candle_index, e.sweep_index) for e in events]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            keys, [("high", 1, 5), ("high", 3, 5), ("high", 1, 6), ("high", 3, 6)]
        )

    def test_first_sweep_only_keeps_one_event_per_swing(self):
        candles = MULTI + (at(30, 15.5, 8, 12),)

        events = analyze_liquidity_sweeps(candles, CFG1, SweepConfig(first_sweep_only=True))

        self.assertEqual([(e.swing.candle_index, e.sweep_index) for e in events],
                         [(1, 5), (3, 5)])

    def test_repeated_processing_is_identical(self):
        candles = _random_candles(5, 300)
        structure_config = StructureConfig(2, 2)

        self.assertEqual(
            analyze_liquidity_sweeps(candles, structure_config),
            analyze_liquidity_sweeps(candles, structure_config),
        )


class TestNoLookahead(unittest.TestCase):

    def test_prefix_events_equal_full_events_known_by_then(self):
        for candles, config in (
            (MULTI + (at(30, 15.5, 8, 12),), CFG1),
            (_random_candles(3, 250), StructureConfig(2, 2)),
            (_random_candles(4, 250), StructureConfig(1, 3)),
        ):
            full = analyze_liquidity_sweeps(candles, config)
            self.assertTrue(full)
            for end in range(len(candles) + 1):
                prefix = analyze_liquidity_sweeps(candles[:end], config)
                known = candles[end - 1].timestamp + STEP if end else None
                expected = tuple(e for e in full if known and e.available_at <= known)
                self.assertEqual(prefix, expected, msg=f"prefix length {end}")

    def test_changing_a_future_candle_does_not_change_earlier_events(self):
        original = MULTI + (at(30, 15.5, 8, 12), at(35, 20, 8, 19))
        altered = MULTI + (at(30, 15.5, 8, 12), at(35, 9, 8, 8.5))

        first = analyze_liquidity_sweeps(original, CFG1)
        second = analyze_liquidity_sweeps(altered, CFG1)

        self.assertEqual(
            [e for e in first if e.sweep_index < 7],
            [e for e in second if e.sweep_index < 7],
        )


class TestValidation(unittest.TestCase):

    def test_empty_and_insufficient_data(self):
        for candles in ((), HIGH_SWEEP[:1], HIGH_SWEEP[:3]):
            with self.subTest(length=len(candles)):
                self.assertEqual(analyze_liquidity_sweeps(candles, CFG1), ())
        empty = diagnose_liquidity_sweeps(())
        self.assertEqual((empty.candle_count, empty.high_side_sweeps, empty.low_side_sweeps),
                         (0, 0, 0))

    def test_invalid_configuration(self):
        for kwargs in (
            {"min_penetration": -1},
            {"min_penetration": float("nan")},
            {"min_penetration": float("inf")},
            {"min_penetration": "1"},
            {"min_penetration": True},
            {"first_sweep_only": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SweepConfig(**kwargs)
        with self.assertRaises(ValueError):
            analyze_liquidity_sweeps(HIGH_SWEEP, CFG1, config="bad")
        with self.assertRaises(ValueError):
            analyze_liquidity_sweeps(HIGH_SWEEP, structure_config="bad")
        with self.assertRaises(ValueError):
            diagnose_liquidity_sweeps(HIGH_SWEEP, CFG1, sample_size=-1)

    def test_rejects_malformed_candles(self):
        naive = (Candle(datetime(2025, 9, 1), 10, 11, 9, 10, 1),)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            analyze_liquidity_sweeps(naive, CFG1)
        with self.assertRaisesRegex(ValueError, "out of order"):
            analyze_liquidity_sweeps(HIGH_SWEEP[::-1], CFG1)
        with self.assertRaisesRegex(ValueError, "High price"):
            analyze_liquidity_sweeps((at(0, 10, 8, 11),), CFG1)

    def test_rejects_swings_that_do_not_match_candles(self):
        swing = find_confirmed_swing_highs(HIGH_SWEEP, CFG1)[0]

        bad_swings = (
            replace(swing, price=13.0),
            replace(swing, candle_index=0),
            replace(swing, confirmation_index=99),
            replace(swing, kind="low"),
        )
        for bad in bad_swings:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                detect_liquidity_sweeps(HIGH_SWEEP, [bad], [], CFG1)
        with self.assertRaises(ValueError):
            detect_liquidity_sweeps(HIGH_SWEEP, ["not a swing"], [], CFG1)
        with self.assertRaises(ValueError):
            detect_liquidity_sweeps(HIGH_SWEEP, [], [swing], CFG1)


class TestDiagnostics(unittest.TestCase):

    def test_counts_by_utc_day_and_samples(self):
        late = datetime(2025, 9, 1, 23, 50, tzinfo=UTC)
        candles = highs_series(
            [(10, 9), (14, 11), (12, 10), (13, 11), (12, 10), (15, 12.5)], base=late
        )

        result = diagnose_liquidity_sweeps(candles, CFG1, sample_size=1)

        self.assertEqual(result.candle_count, 6)
        self.assertEqual((result.high_side_sweeps, result.low_side_sweeps), (2, 0))
        self.assertEqual(result.high_side_by_utc_day, ((date(2025, 9, 2), 2),))
        self.assertEqual(result.low_side_by_utc_day, ())
        self.assertEqual(len(result.sample_events), 1)

    def test_csv_diagnostic_reads_without_modifying(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_utc", "open", "high", "low", "close", "tick_volume"])
                for c in HIGH_SWEEP:
                    writer.writerow([c.timestamp.isoformat(), c.open, c.high, c.low, c.close, 1])
            original = path.read_bytes()

            result = diagnose_historical_csv(path, CFG1)

            self.assertEqual(path.read_bytes(), original)
        self.assertEqual((result.high_side_sweeps, result.low_side_sweeps), (1, 0))


def _random_candles(seed, count):
    rng = random.Random(seed)
    candles, price = [], 100.0
    for i in range(count):
        open_price = price
        price = max(1.0, price + rng.uniform(-1.5, 1.5))
        high = max(open_price, price) + rng.uniform(0, 1.0)
        low = min(open_price, price) - rng.uniform(0, 1.0)
        candles.append(
            Candle(BASE + i * STEP, open_price, high, max(low, 0.5), price, 1)
        )
    return tuple(candles)


if __name__ == "__main__":
    unittest.main()
