import csv
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path

from core.market_data import Candle
from research.market_structure import (
    StructureConfig,
    analyze_market_structure,
    diagnose_historical_csv,
    diagnose_market_structure,
    find_confirmed_swing_highs,
    find_confirmed_swing_lows,
    find_structural_breaks,
    find_swing_high_candidates,
    find_swing_low_candidates,
)


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)


def make(index, high, low, close=None):
    close = (high + low) / 2 if close is None else close
    return Candle(BASE + index * STEP, close, high, low, close, 1)


def series(highs, lows):
    return tuple(make(i, h, l) for i, (h, l) in enumerate(zip(highs, lows)))


S1_HIGHS = [10, 11, 15, 12, 11, 12, 14, 13, 12]
S1_LOWS = [8, 7, 9, 8, 5, 8, 7, 8, 9]

# (high, low, close); with left=right=1 gives up-break at 4 and down-break at 5.
S2 = tuple(
    make(i, *values)
    for i, values in enumerate(
        [
            (10, 8, 9),
            (14, 9, 12),
            (12, 7, 8),
            (13, 8, 13),
            (15, 12, 14.5),
            (14, 5, 6),
        ]
    )
)
CONFIG_1 = StructureConfig(1, 1)


class TestSwings(unittest.TestCase):

    def test_obvious_swings_and_confirmation_timestamps(self):
        candles = series(S1_HIGHS, S1_LOWS)
        config = StructureConfig(2, 2)

        highs = find_confirmed_swing_highs(candles, config)
        lows = find_confirmed_swing_lows(candles, config)

        self.assertEqual([s.candle_index for s in highs], [2, 6])
        self.assertEqual([s.price for s in highs], [15, 14])
        self.assertEqual([s.confirmation_index for s in highs], [4, 8])
        self.assertEqual(highs[0].timestamp, BASE + 2 * STEP)
        self.assertEqual(highs[0].confirmed_at, BASE + 25 * timedelta(minutes=1))
        self.assertEqual(highs[1].confirmed_at, BASE + timedelta(minutes=45))
        self.assertEqual([s.candle_index for s in lows], [4])
        self.assertEqual(lows[0].price, 5)
        self.assertEqual(lows[0].confirmed_at, BASE + timedelta(minutes=35))
        self.assertTrue(all(s.confirmed_at > s.timestamp for s in highs + lows))

    def test_candidates_need_only_left_side_and_insufficient_right_candles(self):
        candles = series(S1_HIGHS, S1_LOWS)[:8]
        config = StructureConfig(2, 2)

        self.assertEqual(
            [c.candle_index for c in find_swing_high_candidates(candles, config)],
            [2, 6],
        )
        self.assertEqual(
            [s.candle_index for s in find_confirmed_swing_highs(candles, config)], [2]
        )
        self.assertEqual(
            [c.candle_index for c in find_swing_low_candidates(candles, config)], [4]
        )
        candidate = find_swing_high_candidates(candles, config)[1]
        self.assertEqual(candidate.available_at, BASE + timedelta(minutes=35))

    def test_multiple_lookback_configurations(self):
        candles = series([5, 7, 6, 9, 8, 10, 9], [1] * 7)
        expected = {(1, 1): [1, 3, 5], (2, 1): [3, 5], (1, 2): [], (2, 2): []}
        for (left, right), indices in expected.items():
            with self.subTest(left=left, right=right):
                swings = find_confirmed_swing_highs(candles, StructureConfig(left, right))
                self.assertEqual([s.candle_index for s in swings], indices)

    def test_equal_highs_and_lows_follow_tie_policy(self):
        highs = series([10, 12, 12, 10, 9], [5] * 5)
        lows = series([20] * 5, [9, 7, 7, 9, 10])
        strict = StructureConfig(1, 1, equal_policy="strict")
        inclusive = StructureConfig(1, 1, equal_policy="inclusive")

        self.assertEqual(find_confirmed_swing_highs(highs, strict), ())
        self.assertEqual(find_confirmed_swing_lows(lows, strict), ())
        self.assertEqual(
            [s.candle_index for s in find_confirmed_swing_highs(highs, inclusive)],
            [1, 2],
        )
        self.assertEqual(
            [s.candle_index for s in find_confirmed_swing_lows(lows, inclusive)],
            [1, 2],
        )

    def test_candle_confirmed_only_if_not_exceeded_before_confirmation(self):
        candles = series([10, 14, 15, 9], [8, 8, 8, 8])

        self.assertEqual(
            [c.candle_index for c in find_swing_high_candidates(candles, CONFIG_1)],
            [1, 2],
        )
        self.assertEqual(
            [s.candle_index for s in find_confirmed_swing_highs(candles, CONFIG_1)], [2]
        )


class TestBreaks(unittest.TestCase):

    def test_upward_and_downward_breaks(self):
        structure = analyze_market_structure(S2, CONFIG_1)

        self.assertEqual([s.candle_index for s in structure.swing_highs], [1, 4])
        self.assertEqual([s.candle_index for s in structure.swing_lows], [2])
        up, down = structure.breaks
        self.assertEqual((up.direction, up.break_index, up.level, up.break_price), ("up", 4, 14, 14.5))
        self.assertEqual(up.swing.candle_index, 1)
        self.assertEqual(up.break_timestamp, BASE + timedelta(minutes=20))
        self.assertEqual(up.available_at, BASE + timedelta(minutes=25))
        self.assertEqual((down.direction, down.break_index, down.level, down.break_price), ("down", 5, 7, 6))
        self.assertEqual(down.swing.candle_index, 2)

    def test_break_never_uses_swing_unconfirmed_at_break_time(self):
        # The swing at index 4 (15) is only confirmed at index 5, so it cannot
        # label the index-4 break of the earlier 14 swing.
        structure = analyze_market_structure(S2, CONFIG_1)

        for event in structure.breaks:
            self.assertGreater(event.break_index, event.swing.confirmation_index)
            self.assertGreaterEqual(event.available_at, event.swing.confirmed_at)
        self.assertEqual([e.swing.candle_index for e in structure.breaks], [1, 2])

    def test_unconfirmed_candidate_exceeded_by_close_is_not_a_break_level(self):
        candles = (
            make(0, 10, 8),
            make(1, 14, 8),
            make(2, 15, 8, close=14.8),
            make(3, 9, 8),
        )

        structure = analyze_market_structure(candles, CONFIG_1)

        self.assertEqual([s.candle_index for s in structure.swing_highs], [2])
        self.assertEqual(structure.breaks, ())

    def test_break_basis_and_exact_touch(self):
        def candles(high):
            return (
                make(0, 10, 8, 9),
                make(1, 14, 9, 12),
                make(2, 12, 8, 10),
                make(3, high, 10, 13),
            )

        wick = StructureConfig(1, 1, break_basis="wick")
        self.assertEqual(analyze_market_structure(candles(15), CONFIG_1).breaks, ())
        events = analyze_market_structure(candles(15), wick).breaks
        self.assertEqual([(e.break_index, e.break_price) for e in events], [(3, 15)])
        self.assertEqual(analyze_market_structure(candles(14), wick).breaks, ())

    def test_one_candle_breaking_two_swings_has_stable_order(self):
        candles = tuple(
            make(i, *v)
            for i, v in enumerate(
                [(10, 8, 9), (14, 8, 12), (12, 8, 10), (13, 8, 11), (12, 8, 10), (16, 8, 15)]
            )
        )

        breaks = analyze_market_structure(candles, CONFIG_1).breaks

        self.assertEqual([(b.break_index, b.level) for b in breaks], [(5, 14), (5, 13)])

    def test_each_swing_breaks_once(self):
        candles = S2 + (make(6, 20, 14, 19), make(7, 21, 15, 20))

        breaks = analyze_market_structure(candles, CONFIG_1).breaks

        self.assertEqual(len([b for b in breaks if b.swing.candle_index == 1]), 1)

    def test_break_finder_accepts_explicit_swings(self):
        highs = find_confirmed_swing_highs(S2, CONFIG_1)
        self.assertEqual(len(find_structural_breaks(S2, highs, (), CONFIG_1)), 1)


class TestSafetyAndDeterminism(unittest.TestCase):

    def test_no_future_data_leakage_for_every_prefix(self):
        for candles, config in (
            (series(S1_HIGHS, S1_LOWS), StructureConfig(2, 2)),
            (S2, CONFIG_1),
        ):
            full = analyze_market_structure(candles, config)
            for end in range(len(candles) + 1):
                with self.subTest(end=end, left=config.left_lookback):
                    prefix = analyze_market_structure(candles[:end], config)
                    known = candles[end - 1].timestamp + STEP if end else None
                    self.assertEqual(
                        prefix.swing_highs,
                        tuple(s for s in full.swing_highs if known and s.confirmed_at <= known),
                    )
                    self.assertEqual(
                        prefix.swing_lows,
                        tuple(s for s in full.swing_lows if known and s.confirmed_at <= known),
                    )
                    self.assertEqual(
                        prefix.breaks,
                        tuple(b for b in full.breaks if known and b.available_at <= known),
                    )

    def test_repeated_processing_is_identical_and_inputs_unchanged(self):
        candles = series(S1_HIGHS, S1_LOWS)
        before = tuple(candles)

        first = analyze_market_structure(candles, StructureConfig(2, 2))
        second = analyze_market_structure(candles, StructureConfig(2, 2))

        self.assertEqual(first, second)
        self.assertEqual(candles, before)

    def test_empty_and_short_datasets(self):
        for candles in ((), series([10], [8]), series([10, 12], [8, 9])):
            with self.subTest(length=len(candles)):
                structure = analyze_market_structure(candles, StructureConfig(2, 2))
                self.assertEqual(structure.swing_highs, ())
                self.assertEqual(structure.swing_lows, ())
                self.assertEqual(structure.breaks, ())
        self.assertEqual(diagnose_market_structure(()).candle_count, 0)

    def test_invalid_configuration(self):
        bad = (
            {"left_lookback": 0},
            {"right_lookback": -1},
            {"left_lookback": 1.5},
            {"left_lookback": True},
            {"equal_policy": "loose"},
            {"break_basis": "open"},
            {"candle_duration": timedelta(0)},
            {"candle_duration": 5},
        )
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StructureConfig(**kwargs)
        with self.assertRaises(ValueError):
            analyze_market_structure(S2, config="not a config")

    def test_rejects_invalid_naive_and_disordered_candles(self):
        naive = (Candle(datetime(2025, 9, 1), 10, 11, 9, 10, 1),)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            analyze_market_structure(naive, CONFIG_1)
        with self.assertRaisesRegex(ValueError, "out of order"):
            analyze_market_structure(S2[::-1], CONFIG_1)
        with self.assertRaisesRegex(ValueError, "High price"):
            analyze_market_structure((make(0, 10, 8, close=11),), CONFIG_1)


class TestDiagnostics(unittest.TestCase):

    def test_counts_by_day_and_samples(self):
        result = diagnose_market_structure(S2, CONFIG_1, sample_size=1)

        self.assertEqual(result.candle_count, 6)
        self.assertEqual(
            (result.confirmed_swing_highs, result.confirmed_swing_lows), (2, 1)
        )
        self.assertEqual((result.upward_breaks, result.downward_breaks), (1, 1))
        day = date(2025, 9, 1)
        self.assertEqual(result.swing_highs_by_utc_day, ((day, 2),))
        self.assertEqual(result.swing_lows_by_utc_day, ((day, 1),))
        self.assertEqual(result.upward_breaks_by_utc_day, ((day, 1),))
        self.assertEqual(result.downward_breaks_by_utc_day, ((day, 1),))
        self.assertEqual(len(result.sample_swings), 1)
        self.assertEqual(result.sample_swings[0].candle_index, 1)
        self.assertEqual(len(result.sample_breaks), 1)
        with self.assertRaises(ValueError):
            diagnose_market_structure(S2, CONFIG_1, sample_size=-1)

    def test_diagnoses_csv_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_utc", "open", "high", "low", "close", "tick_volume"])
                for c in S2:
                    writer.writerow([c.timestamp.isoformat(), c.open, c.high, c.low, c.close, c.tick_volume])
            original = path.read_bytes()

            result = diagnose_historical_csv(path, CONFIG_1)

            self.assertEqual(path.read_bytes(), original)
        self.assertEqual((result.upward_breaks, result.downward_breaks), (1, 1))


def at(minute, high, low, close=None):
    close = (high + low) / 2 if close is None else close
    return Candle(BASE + timedelta(minutes=minute), close, high, low, close, 1)


class FallBackTimezone(tzinfo):
    """-04:00 until the 2025-11-02 01:00-02:00 wall hour; fold=1 there is -05:00."""

    def utcoffset(self, value):
        wall = value.replace(tzinfo=None)
        if wall < datetime(2025, 11, 2, 1):
            return timedelta(hours=-4)
        if wall < datetime(2025, 11, 2, 2):
            return timedelta(hours=-4 if value.fold == 0 else -5)
        return timedelta(hours=-5)

    def dst(self, value):  # noqa: ARG002 - fixed standard offset; DST is in utcoffset
        return timedelta(0)


class TestTimestampSemantics(unittest.TestCase):

    def test_timestamps_are_open_times_and_availability_is_close_of_last_right_candle(self):
        candles = series(S1_HIGHS, S1_LOWS)
        config = StructureConfig(2, 2)
        swing = find_confirmed_swing_highs(candles, config)[0]

        self.assertEqual(swing.candle_index, 2)
        self.assertEqual(swing.timestamp, candles[2].timestamp)
        last_right = candles[swing.candle_index + config.right_lookback]
        self.assertEqual(swing.confirmed_at, last_right.timestamp + config.candle_duration)
        self.assertGreater(swing.confirmed_at, last_right.timestamp)

        for end, expected in ((4, 0), (5, 1)):
            with self.subTest(candles_closed=end):
                prefix = find_confirmed_swing_highs(candles[:end], config)
                self.assertEqual(len([s for s in prefix if s.candle_index == 2]), expected)

    def test_candle_duration_controls_availability_consistently(self):
        candles = tuple(
            Candle(BASE + i * timedelta(minutes=15), 9, h, 8, 9, 1)
            for i, h in enumerate([10, 12, 10, 9])
        )
        config = StructureConfig(1, 1, candle_duration=timedelta(minutes=15))

        swing = find_confirmed_swing_highs(candles, config)[0]
        candidate = find_swing_high_candidates(candles, config)[0]

        self.assertEqual(swing.confirmed_at, BASE + timedelta(minutes=45))
        self.assertEqual(candidate.available_at, BASE + timedelta(minutes=30))

    def test_breaks_are_never_available_before_their_swing(self):
        for config in (
            StructureConfig(1, 1),
            StructureConfig(1, 1, equal_policy="inclusive"),
            StructureConfig(1, 1, break_basis="wick"),
        ):
            structure = analyze_market_structure(S2, config)
            self.assertTrue(structure.breaks)
            for event in structure.breaks:
                with self.subTest(config=config, swing=event.swing.candle_index):
                    self.assertGreater(event.break_timestamp, event.swing.timestamp)
                    self.assertGreaterEqual(event.break_timestamp, event.swing.confirmed_at)
                    self.assertGreater(event.available_at, event.swing.confirmed_at)

    def test_first_possible_break_is_the_candle_after_confirmation(self):
        candles = (at(0, 10, 8), at(5, 14, 9), at(10, 12, 8), at(15, 15, 9, 14.5))

        event = analyze_market_structure(candles, CONFIG_1).breaks[0]

        self.assertEqual(event.swing.confirmed_at, BASE + timedelta(minutes=15))
        self.assertEqual(event.break_timestamp, BASE + timedelta(minutes=15))
        self.assertEqual(event.available_at, BASE + timedelta(minutes=20))

    def test_gaps_are_counted_as_candles_not_filled_and_use_duration_for_availability(self):
        candles = (
            at(0, 10, 8),
            at(5, 14, 9),
            at(65, 12, 8),
            at(70, 15, 9, 14.5),
        )
        before = tuple(candles)

        structure = analyze_market_structure(candles, CONFIG_1)

        self.assertEqual(candles, before)
        swing = structure.swing_highs[0]
        self.assertEqual(swing.confirmation_index, 2)
        self.assertEqual(swing.confirmed_at, BASE + timedelta(minutes=70))
        self.assertLess(swing.confirmed_at, candles[3].timestamp + CONFIG_1.candle_duration)
        event = structure.breaks[0]
        self.assertEqual(event.break_timestamp, BASE + timedelta(minutes=70))
        self.assertEqual(event.available_at, BASE + timedelta(minutes=75))

    def test_non_utc_offsets_give_identical_events_and_utc_day_counts(self):
        offset = timezone(timedelta(hours=2))
        shifted = tuple(
            Candle(c.timestamp.astimezone(offset), c.open, c.high, c.low, c.close, 1)
            for c in S2
        )

        self.assertEqual(
            analyze_market_structure(shifted, CONFIG_1),
            analyze_market_structure(S2, CONFIG_1),
        )

        late = tuple(
            Candle(
                datetime(2025, 9, 1, 23, 50, tzinfo=UTC).astimezone(offset)
                + timedelta(minutes=5 * i),
                c.open, c.high, c.low, c.close, 1,
            )
            for i, c in enumerate(S2)
        )
        days = diagnose_market_structure(late, CONFIG_1).swing_highs_by_utc_day
        self.assertEqual(days, ((date(2025, 9, 1), 1), (date(2025, 9, 2), 1)))

    def test_availability_is_exact_across_a_dst_fold(self):
        tz = FallBackTimezone()
        stamps = (
            datetime(2025, 11, 2, 1, 50, tzinfo=tz),
            datetime(2025, 11, 2, 1, 55, tzinfo=tz),
            datetime(2025, 11, 2, 1, 0, tzinfo=tz, fold=1),
            datetime(2025, 11, 2, 1, 5, tzinfo=tz, fold=1),
        )
        highs = (10, 14, 12, 15)
        candles = tuple(
            Candle(t, 9, h, 8, h - 0.5 if i == 3 else 9, 1)
            for i, (t, h) in enumerate(zip(stamps, highs))
        )
        utc_open = [c.timestamp.astimezone(UTC) for c in candles]
        self.assertEqual(
            [b - a for a, b in zip(utc_open, utc_open[1:])], [timedelta(minutes=5)] * 3
        )

        swing = analyze_market_structure(candles, CONFIG_1).swing_highs[0]

        self.assertEqual(swing.confirmed_at, utc_open[2] + CONFIG_1.candle_duration)


class TestEqualPricePlateaus(unittest.TestCase):

    def swing_indices(self, highs, policy, left=1, right=1, kind="high"):
        flat = [5] * len(highs)
        candles = series(highs, flat) if kind == "high" else series([20] * len(highs), highs)
        config = StructureConfig(left, right, equal_policy=policy)
        find = find_confirmed_swing_highs if kind == "high" else find_confirmed_swing_lows
        return [(s.candle_index, s.confirmation_index, s.confirmed_at) for s in find(candles, config)]

    def test_three_equal_highs(self):
        values = [10, 12, 12, 12, 10, 9]

        self.assertEqual(self.swing_indices(values, "strict"), [])
        inclusive = self.swing_indices(values, "inclusive")
        self.assertEqual([(i, c) for i, c, _ in inclusive], [(1, 2), (2, 3), (3, 4)])
        self.assertEqual(
            [t for _, _, t in inclusive],
            [BASE + timedelta(minutes=m) for m in (15, 20, 25)],
        )

    def test_three_equal_highs_with_wider_lookback(self):
        values = [10, 11, 12, 12, 12, 11, 10]

        self.assertEqual(self.swing_indices(values, "strict", 2, 2), [])
        self.assertEqual(
            [(i, c) for i, c, _ in self.swing_indices(values, "inclusive", 2, 2)],
            [(2, 4), (3, 5), (4, 6)],
        )

    def test_three_equal_lows(self):
        values = [9, 7, 7, 7, 9, 10]

        self.assertEqual(self.swing_indices(values, "strict", kind="low"), [])
        self.assertEqual(
            [(i, c) for i, c, _ in self.swing_indices(values, "inclusive", kind="low")],
            [(1, 2), (2, 3), (3, 4)],
        )

    def test_plateau_followed_by_breakout_gives_one_event_per_plateau_swing(self):
        candles = (
            at(0, 10, 5),
            at(5, 12, 5),
            at(10, 12, 5),
            at(15, 12, 5),
            at(20, 10, 5),
            at(25, 14, 5, 13.5),
        )
        strict = analyze_market_structure(candles, StructureConfig(1, 1))
        inclusive = analyze_market_structure(
            candles, StructureConfig(1, 1, equal_policy="inclusive")
        )

        self.assertEqual((strict.swing_highs, strict.breaks), ((), ()))
        self.assertEqual([s.candle_index for s in inclusive.swing_highs], [1, 2, 3])
        self.assertEqual(
            [(b.swing.candle_index, b.break_index, b.level) for b in inclusive.breaks if b.direction == "up"],
            [(1, 5, 12), (2, 5, 12), (3, 5, 12)],
        )

    def test_repeated_equal_highs_separated_by_lower_highs_are_distinct_swings_in_both_policies(self):
        values = [10, 12, 10, 12, 10, 12, 9]

        for policy in ("strict", "inclusive"):
            with self.subTest(policy=policy):
                self.assertEqual(
                    [(i, c) for i, c, _ in self.swing_indices(values, policy)],
                    [(1, 2), (3, 4), (5, 6)],
                )

    def test_plateau_at_start_of_data(self):
        values = [12, 12, 12, 10, 9]

        self.assertEqual(self.swing_indices(values, "strict"), [])
        self.assertEqual(
            [(i, c) for i, c, _ in self.swing_indices(values, "inclusive")],
            [(1, 2), (2, 3)],
        )

    def test_plateau_at_end_of_data_waits_for_right_candles(self):
        values = [9, 10, 12, 12, 12]
        candles = series(values, [5] * 5)
        strict = StructureConfig(1, 1)
        inclusive = StructureConfig(1, 1, equal_policy="inclusive")

        self.assertEqual(find_confirmed_swing_highs(candles, strict), ())
        self.assertEqual(
            [s.candle_index for s in find_confirmed_swing_highs(candles, inclusive)],
            [2, 3],
        )
        self.assertEqual(
            [c.candle_index for c in find_swing_high_candidates(candles, inclusive)],
            [1, 2, 3, 4],
        )

    def test_outside_bar_can_be_both_swings_and_ordering_is_stable(self):
        candles = (
            at(0, 12, 8),
            at(5, 11, 9),
            at(10, 14, 6, 10),
            at(15, 11, 9),
            at(20, 13, 7, 8),
            at(25, 12, 8, 10),
            at(30, 15, 8, 14.5),
            at(35, 14, 3, 4),
        )

        structure = analyze_market_structure(candles, CONFIG_1)

        self.assertEqual([s.candle_index for s in structure.swing_highs], [2, 4, 6])
        self.assertEqual([s.candle_index for s in structure.swing_lows], [2, 4])
        self.assertEqual(
            [(b.break_index, b.direction, b.swing.candle_index) for b in structure.breaks],
            [(6, "up", 2), (6, "up", 4), (7, "down", 2), (7, "down", 4)],
        )


if __name__ == "__main__":
    unittest.main()
