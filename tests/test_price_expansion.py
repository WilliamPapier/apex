import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from core.market_data import Candle
from research.market_structure import StructureConfig
from research.price_expansion import (
    ExpansionConfig,
    ExpansionLabel,
    FlagConfig,
    analyze_expansion,
    baseline_flags,
    build_predictors,
    evaluate_flag,
    label_expansions,
    select_non_overlapping,
    summarize_labels,
)


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
FIVE = timedelta(minutes=5)
CFG = ExpansionConfig(
    structure=StructureConfig(1, 1),
    horizons=(FIVE, 2 * FIVE),
    atr_multiples=(1.0, 2.0),
    atr_period=2,
    volatility_periods=(1, 2),
    compression_short=2,
    compression_long=3,
    volume_lookback=2,
    momentum_candles=1,
    sequence_candles=2,
    event_lookback=3,
)


def series(rows, volumes=None, gaps=None, base=BASE):
    """rows are (open, high, low, close); gaps maps index -> extra minutes before it."""
    candles, shift = [], 0
    for i, (o, h, l, c) in enumerate(rows):
        shift += (gaps or {}).get(i, 0)
        candles.append(
            Candle(
                timestamp=base + timedelta(minutes=5 * i + shift),
                open=o, high=h, low=l, close=c,
                tick_volume=(volumes or [10] * len(rows))[i],
            )
        )
    return candles


# TRs 2,2,2,3,... -> ATR(2) = None, 2, 2, 2.5, ...
A = [
    (10, 11, 9, 10),
    (10, 11, 9, 10),
    (10, 11, 9, 10),
    (10, 13, 10, 12),
    (12, 12, 11, 11),
]
A_VOLUMES = [10, 10, 20, 10, 10]


def label(labels, index, horizon=FIVE, multiple=1.0):
    return next(
        l for l in labels
        if l.index == index and l.horizon == horizon and l.atr_multiple == multiple
    )


class TestConfig(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        bad = [
            dict(horizons=()),
            dict(horizons=(FIVE, FIVE)),
            dict(horizons=(timedelta(minutes=7),)),
            dict(horizons=(timedelta(0),)),
            dict(atr_multiples=()),
            dict(atr_multiples=(0.0,)),
            dict(atr_multiples=(float("nan"),)),
            dict(min_absolute_move=-1.0),
            dict(continuation_retention=0.0),
            dict(continuation_retention=1.5),
            dict(atr_period=0),
            dict(volatility_periods=()),
            dict(compression_short=5, compression_long=5),
            dict(pin_wick_ratio=0.5),
            dict(pin_body_ratio=0.5),
            dict(allow_gap_windows="yes"),
            dict(structure="x"),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExpansionConfig(**kwargs)

    def test_non_config_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            label_expansions(series(A), config="bad")
        with self.assertRaises(ValueError):
            build_predictors(series(A), config="bad")
        with self.assertRaises(ValueError):
            analyze_expansion(series(A), config="bad")


class TestLabels(unittest.TestCase):
    def test_upward_expansion_hand_calculated(self):
        labels = label_expansions(series(A), CFG)
        # i=2: ATR 2, reference close 10, window = candle 3 (high 13, close 12).
        one = label(labels, 2)
        self.assertEqual((one.status, one.window_candles), ("complete", 1))
        self.assertEqual((one.atr, one.threshold), (2.0, 2.0))
        self.assertEqual((one.expanded, one.direction), (True, "up"))
        self.assertEqual((one.max_up, one.max_down, one.mfe, one.mae), (3, 0, 3, 0))
        self.assertEqual(one.available_at, BASE + timedelta(minutes=15))
        self.assertEqual(one.threshold_reached_at, BASE + timedelta(minutes=20))
        self.assertEqual(one.time_to_threshold, FIVE)
        self.assertEqual(one.end_close_change, 2)
        self.assertAlmostEqual(one.retained_fraction, 2 / 3)
        self.assertEqual(one.outcome, "continuation")
        self.assertIsNone(one.momentum_profile)  # k = 1

        # Two-candle window adds candle 4 (high 12, low 11, close 11): retained 1/3.
        two = label(labels, 2, 2 * FIVE)
        self.assertEqual((two.window_candles, two.end_close_change), (2, 1))
        self.assertAlmostEqual(two.retained_fraction, 1 / 3)
        self.assertEqual(two.outcome, "retracement")
        self.assertEqual(two.momentum_profile, "front_loaded")  # 3 > 1.5

        # Threshold 4 (2 x ATR) is not reached by a 3-point excursion.
        none = label(labels, 2, 2 * FIVE, 2.0)
        self.assertEqual((none.threshold, none.expanded, none.direction), (4.0, False, None))
        self.assertEqual((none.max_up, none.mfe), (3, None))

    def test_warmup_and_incomplete_windows_are_not_labelled(self):
        labels = label_expansions(series(A), CFG)

        self.assertEqual(label(labels, 0).status, "atr_unavailable")
        self.assertIsNone(label(labels, 0).expanded)
        self.assertEqual(label(labels, 4).status, "incomplete")  # no future candle
        self.assertIsNone(label(labels, 4).expanded)
        partial = label(labels, 3, 2 * FIVE)
        self.assertEqual((partial.status, partial.window_candles), ("incomplete", 1))
        self.assertIsNone(partial.expanded)
        self.assertEqual(len(labels), 5 * 2 * 2)

    def test_threshold_boundary_is_inclusive_and_floors_apply(self):
        def expanded(high, **kwargs):
            rows = A[:3] + [(10, high, 10, 10)]
            config = replace(CFG, **kwargs)
            return label(label_expansions(series(rows), config), 2).expanded

        self.assertTrue(expanded(12.0))  # excursion exactly 2 x 1.0 ATR
        self.assertFalse(expanded(11.99))
        self.assertFalse(expanded(12.0, min_absolute_move=2.5))
        self.assertTrue(expanded(12.5, min_absolute_move=2.5))
        self.assertFalse(expanded(12.0, min_fractional_change=0.25))  # needs 2.5
        self.assertTrue(expanded(12.5, min_fractional_change=0.25))

    def test_downward_both_and_reversal(self):
        down = A[:3] + [(10, 10, 7, 8)]
        both = A[:3] + [(10, 13, 7, 10)]
        reversal = A[:3] + [(10, 13, 9.5, 9.5)]

        d = label(label_expansions(series(down), CFG), 2)
        self.assertEqual((d.direction, d.mfe, d.mae), ("down", 3, 0))
        self.assertAlmostEqual(d.retained_fraction, 2 / 3)
        b = label(label_expansions(series(both), CFG), 2)
        self.assertEqual((b.expanded, b.direction), (True, "both"))
        self.assertEqual((b.mfe, b.outcome), (None, None))
        r = label(label_expansions(series(reversal), CFG), 2)
        self.assertEqual((r.direction, r.outcome), ("up", "reversal"))
        self.assertAlmostEqual(r.retained_fraction, -0.5 / 3)

    def test_back_loaded_profile(self):
        rows = A[:3] + [(10, 10.5, 10, 10.5), (10.5, 13, 10.5, 13)]
        result = label(label_expansions(series(rows), CFG), 2, 2 * FIVE)
        self.assertEqual((result.mfe, result.momentum_profile), (3, "back_loaded"))

    def test_gaps_are_flagged_never_filled(self):
        rows = A[:3] + [(10, 13, 10, 12), (12, 12, 11, 11)]
        candles = series(rows, gaps={3: 60})
        snapshot = list(candles)

        labels = label_expansions(candles, CFG)
        gap = label(labels, 2)
        self.assertEqual((gap.status, gap.window_candles, gap.expanded), ("gap", 0, None))
        permissive = label_expansions(candles, replace(CFG, allow_gap_windows=True))
        self.assertEqual(label(permissive, 2).status, "gap")
        self.assertEqual(label(permissive, 2).window_candles, 0)
        self.assertEqual(candles, snapshot)
        self.assertEqual(len(candles), 5)

        inside = series(rows, gaps={4: 60})  # window for i=2, h=10 holds candle 3 only
        strict = label(label_expansions(inside, CFG), 2, 2 * FIVE)
        loose = label(label_expansions(inside, replace(CFG, allow_gap_windows=True)), 2, 2 * FIVE)
        self.assertEqual((strict.status, strict.expanded), ("gap", None))
        self.assertEqual((loose.window_candles, loose.expanded, loose.direction), (1, True, "up"))

    def test_timezone_aware_candles_match_utc(self):
        plus_two = timezone(timedelta(hours=2))
        local = series(A, volumes=A_VOLUMES, base=BASE.astimezone(plus_two))
        utc = series(A, volumes=A_VOLUMES)

        local_labels = label_expansions(local, CFG)
        utc_labels = label_expansions(utc, CFG)

        for a, b in zip(local_labels, utc_labels):
            self.assertEqual(a.available_at, b.available_at)
            self.assertEqual(a.available_at.utcoffset(), timedelta(0))
            self.assertEqual(replace(a, timestamp=b.timestamp), b)
        self.assertEqual(local_labels[0].timestamp.utcoffset(), timedelta(hours=2))

    def test_invalid_and_empty_inputs(self):
        self.assertEqual(label_expansions([], CFG), ())
        self.assertEqual(build_predictors([], CFG), ())
        single = label_expansions(series(A[:1]), CFG)
        self.assertTrue(all(l.status == "atr_unavailable" for l in single))
        naive = [Candle(datetime(2025, 9, 1), 1, 1, 1, 1, 1)]
        with self.assertRaises(ValueError):
            label_expansions(naive, CFG)
        with self.assertRaises(ValueError):
            label_expansions(list(reversed(series(A))), CFG)


def fake_label(index, expanded, lead=None, horizon=FIVE, multiple=1.0):
    available = BASE + timedelta(minutes=5 * index + 5)
    return ExpansionLabel(
        index=index, timestamp=BASE + timedelta(minutes=5 * index), available_at=available,
        horizon=horizon, atr_multiple=multiple, status="complete", reference_close=10.0,
        atr=1.0, threshold=1.0, window_end=available + horizon, window_candles=1,
        expanded=expanded, direction="up" if expanded else None, threshold_reached_at=None,
        time_to_threshold=lead, max_up=None, max_down=None, mfe=None, mae=None,
        end_close_change=None, retained_fraction=None, outcome=None, momentum_profile=None,
    )


class TestOverlapSummaryAndBaseline(unittest.TestCase):
    def test_non_overlapping_selection_is_chronological_and_greedy(self):
        ten = 2 * FIVE
        labels = [fake_label(i, True, horizon=ten) for i in (3, 0, 1, 2, 4)]
        labels.append(fake_label(5, False, horizon=ten))
        labels.append(fake_label(0, True, horizon=FIVE))

        picked = select_non_overlapping(labels, ten, 1.0)

        # 0 ends at +15min and index 2 is available at +15min; 4 starts after 2 ends.
        self.assertEqual([l.index for l in picked], [0, 2, 4])
        self.assertEqual([l.index for l in select_non_overlapping(labels, FIVE, 1.0)], [0])
        self.assertEqual(select_non_overlapping(labels, ten, 2.0), ())

    def test_summary_counts_hand_calculated(self):
        research = analyze_expansion(series(A, volumes=A_VOLUMES), CFG)
        summary = {
            (s.horizon, s.atr_multiple): s for s in summarize_labels(research.labels, CFG)
        }[(FIVE, 1.0)]

        self.assertEqual(
            (summary.total, summary.atr_unavailable, summary.incomplete, summary.gap),
            (5, 1, 1, 0),
        )
        self.assertEqual((summary.labelled, summary.expanded), (3, 1))
        self.assertEqual((summary.up, summary.down, summary.both), (1, 0, 0))
        self.assertEqual(summary.continuation, 1)
        self.assertAlmostEqual(summary.base_rate, 1 / 3)
        self.assertEqual(summary.median_time_to_threshold, FIVE)
        self.assertEqual(summary.non_overlapping_events, 1)

    def test_flag_evaluation_hand_calculated(self):
        candles = series([(10, 11, 9, 10)] * 8)
        snapshots = build_predictors(candles, CFG)
        expanded = {1: FIVE, 2: FIVE, 5: 2 * FIVE}
        labels = [fake_label(i, i in expanded, expanded.get(i)) for i in range(8)]
        flagged = {1, 3, 5, 6}

        first, second = evaluate_flag(
            snapshots, labels, "x", lambda s: s.index in flagged, FIVE, 1.0, FIVE
        )

        self.assertEqual((first.eligible, first.flagged, first.events, first.true_positives), (4, 2, 2, 1))
        self.assertEqual((first.precision, first.recall, first.miss_rate), (0.5, 0.5, 0.5))
        self.assertEqual((first.base_rate, first.lift, first.false_alarms), (0.5, 1.0, 1))
        self.assertAlmostEqual(first.false_alarms_per_hour, 3.0)  # 1 / (4 x 5min)
        self.assertEqual(first.median_lead_time, FIVE)
        self.assertEqual((second.events, second.true_positives, second.recall), (1, 1, 1.0))
        self.assertEqual((second.base_rate, second.lift), (0.25, 2.0))
        self.assertEqual(second.median_lead_time, 2 * FIVE)
        with self.assertRaises(ValueError):
            evaluate_flag(snapshots, labels, "x", lambda s: True, FIVE, 1.0, FIVE, 1.0)

    def test_flag_without_events_or_flags_has_no_ratios(self):
        snapshots = build_predictors(series([(10, 11, 9, 10)] * 4), CFG)
        labels = [fake_label(i, False) for i in range(4)]

        first, _ = evaluate_flag(snapshots, labels, "x", lambda s: False, FIVE, 1.0, FIVE)

        self.assertEqual((first.precision, first.recall, first.lift), (None, None, None))


class TestPredictors(unittest.TestCase):
    def test_features_hand_calculated(self):
        snaps = build_predictors(series(A, volumes=A_VOLUMES), CFG)

        self.assertIsNone(snaps[0].atr)
        self.assertIsNone(snaps[0].range_to_atr)
        s = snaps[2]
        self.assertEqual(s.atr, 2.0)
        self.assertEqual((s.range_to_atr, s.body_to_atr), (1.0, 0.0))
        self.assertEqual((s.upper_wick_to_atr, s.lower_wick_to_atr), (0.5, 0.5))
        self.assertEqual(s.momentum_to_atr, 0.0)
        self.assertEqual(s.recent_directions, ("neutral", "neutral"))
        self.assertEqual(s.range_compression_ratio, 1.0)
        self.assertEqual(s.tick_volume_ratio, 2.0)  # 20 / mean(10, 10)
        self.assertIsNone(s.volatility_ratio)  # zero long-window volatility
        self.assertEqual(s.available_at, BASE + timedelta(minutes=15))
        # warm-up values stay None instead of being invented
        self.assertIsNone(snaps[1].range_compression_ratio)
        self.assertIsNone(snaps[1].tick_volume_ratio)
        self.assertIsNone(snaps[0].recent_directions)
        # big up candle: momentum uses only candles up to i
        self.assertAlmostEqual(snaps[3].momentum_to_atr, 2 / 2.5)

    def test_pin_bars_and_zero_range(self):
        rows = [
            (10, 10.2, 8, 10.1),
            (10, 12, 9.9, 9.95),
            (10, 11, 9, 10),
            (10, 10, 10, 10),
            (10, 11, 8, 10.9),
        ]
        pins = [s.pin_bar for s in build_predictors(series(rows), CFG)]

        self.assertEqual(pins, ["bullish_pin", "bearish_pin", None, None, None])

    def test_structure_features_respect_availability_times(self):
        rows = [(10, 10, 9, 9.5), (9.5, 12, 9.5, 11.5), (11, 11.5, 10, 10.5), (10.5, 12.5, 10, 11.5)]
        snaps = build_predictors(series(rows), CFG)

        # Swing high 12 (candle 1) needs candle 2 to close at minute 15.
        self.assertIsNone(snaps[1].latest_swing_high)
        self.assertEqual(snaps[2].latest_swing_high, 12)
        self.assertEqual(snaps[2].latest_swing_high_confirmed_at, BASE + timedelta(minutes=15))
        self.assertAlmostEqual(snaps[2].distance_to_swing_high_atr, (12 - 10.5) / 1.625)
        self.assertEqual(snaps[2].recent_high_sweeps, 0)
        # Candle 3 wicks to 12.5 and closes at 11.5: sweep known when candle 3 closes.
        self.assertEqual(snaps[3].recent_high_sweeps, 1)
        self.assertEqual(snaps[3].recent_low_sweeps, 0)
        self.assertIsNone(snaps[3].latest_swing_low)

    def test_flag_config_and_flags(self):
        snaps = build_predictors(series(A, volumes=A_VOLUMES), CFG)
        flags = baseline_flags(FlagConfig(volume_spike_min=2.0, compression_ratio_max=1.0))

        self.assertTrue(flags["tick_volume_spike"](snaps[2]))
        self.assertTrue(flags["compressed_range"](snaps[2]))
        self.assertFalse(flags["large_closed_candle"](snaps[2]))
        self.assertFalse(flags["tick_volume_spike"](snaps[0]))
        with self.assertRaises(ValueError):
            FlagConfig(volume_spike_min=0)


def walk(count=60):
    rows, price = [], 100.0
    for i in range(count):
        step = ((i * 7) % 11 - 5) * 0.4
        high = max(price, price + step) + 0.5 + (i % 3) * 0.2
        low = min(price, price + step) - 0.5 - (i % 2) * 0.3
        rows.append((price, high, low, price + step))
        price += step
    return series(rows, volumes=[10 + (i * 3) % 7 for i in range(count)])


class TestNoLookahead(unittest.TestCase):
    def test_predictors_equal_on_every_prefix(self):
        candles = walk()
        full = build_predictors(candles, CFG)

        for m in range(1, len(candles) + 1, 5):
            self.assertEqual(build_predictors(candles[:m], CFG), full[:m])

    def test_changing_future_candles_never_alters_earlier_predictors(self):
        candles = walk()
        altered = list(candles)
        for j in range(40, 60):
            c = candles[j]
            altered[j] = Candle(c.timestamp, c.open + 9, c.high + 50, c.low - 9, c.close + 5, 999)

        before = build_predictors(candles, CFG)
        after = build_predictors(altered, CFG)

        self.assertEqual(before[:40], after[:40])
        self.assertNotEqual(before[40:], after[40:])

    def test_labels_depend_only_on_their_forward_window(self):
        candles = walk()
        base = label_expansions(candles, CFG)

        def altered(index):
            c = candles[index]
            changed = list(candles)
            changed[index] = Candle(c.timestamp, c.open, c.high + 30, c.low - 30, c.close, c.tick_volume)
            return label_expansions(changed, CFG)

        # Candle 21 lies inside the 5- and 10-minute windows of i=20; candle 22 only in the 10-minute
        # one; candle 23 in neither.
        far = altered(23)
        near = altered(21)
        self.assertEqual(label(base, 20, FIVE), label(far, 20, FIVE))
        self.assertEqual(label(base, 20, 2 * FIVE), label(far, 20, 2 * FIVE))
        self.assertNotEqual(label(base, 20, FIVE), label(near, 20, FIVE))
        self.assertEqual(label(base, 19, FIVE), label(near, 19, FIVE))
        # ATR at the prediction point is unaffected by the altered future candle.
        self.assertEqual(label(base, 20).atr, label(near, 20).atr)

    def test_labels_for_a_prefix_match_complete_windows_of_the_full_run(self):
        candles = walk()
        full = label_expansions(candles, CFG)
        prefix = label_expansions(candles[:30], CFG)

        for p in prefix:
            if p.status == "complete":
                self.assertEqual(p, label(full, p.index, p.horizon, p.atr_multiple))

    def test_deterministic(self):
        candles = walk()
        self.assertEqual(analyze_expansion(candles, CFG), analyze_expansion(candles, CFG))


if __name__ == "__main__":
    unittest.main()
