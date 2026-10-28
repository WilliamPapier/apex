import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.market_data import Candle
from research.coordinator import AgentResult, SuiteConfig, build_context, run_agents
from research.liquidity_structure import (
    BaseRates,
    LiquidityStudyConfig,
    PointEvent,
    SweepDefinition,
    build_families,
    detect_sweeps,
    tally_events,
)
from research.liquidity_sweeps import detect_liquidity_sweeps
from research.market_structure import StructureConfig, StructuralBreak, analyze_market_structure
from research.regimes import (
    RegimeStudyConfig,
    RegimeVariant,
    causal_percentile_ranks,
    classify_rank,
    compute_feature,
    contiguous_run_lengths,
)
from research.run_suite import default_agents
from research.study_agents import LiquidityStructureAgent, MarketRegimeAgent
from research.study_stats import run_lengths, state_changes, thin, z_one_proportion, z_two_proportion
from research.walk_forward import WalkForwardConfig
from research.price_expansion import ExpansionConfig, ExpansionLabel
from tools.benchmark_liquidity_structure import _json_default, _write_json_exclusive
from tests.test_coordinator import FakeAgent, SMALL, synthetic

UTC = timezone.utc
T0 = datetime(2025, 9, 1, tzinfo=UTC)
FIVE = timedelta(minutes=5)


class TestBenchmarkReportEncoding(unittest.TestCase):
    def test_timedelta_parameters_are_serialized_as_seconds(self):
        encoded = json.dumps({"duration": FIVE}, default=_json_default)
        self.assertEqual(json.loads(encoded), {"duration": 300.0})

    def test_failed_serialization_does_not_create_partial_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "benchmark.json"
            with self.assertRaises(TypeError):
                _write_json_exclusive(path, {"unsupported": object()})
            self.assertFalse(path.exists())


def candle(i, o, h, l, c, gap=0):
    return Candle(T0 + FIVE * (i + gap), o, h, l, c, 100)


def sweep_series(overrides=None):
    """Swing high 15 at index 2 (confirmed at the close of index 4); sweeps at indices 5 and 6."""

    rows = {
        0: (10, 11, 9, 10), 1: (10, 11, 9, 10), 2: (10, 15, 9.5, 10), 3: (10, 12, 9, 10),
        4: (10, 11, 9, 10), 5: (10, 16, 9.5, 14), 6: (10, 15.5, 9.5, 14),
        7: (10, 11, 9, 10), 8: (10, 11, 9, 10), 9: (10, 11, 9, 10),
    }
    rows.update(overrides or {})
    return [candle(i, *rows[i]) for i in range(10)]


def high_events(candles, definition, atr=1.0):
    structure = analyze_market_structure(candles, StructureConfig(2, 2))
    result = detect_sweeps(candles, structure, [atr] * len(candles), definition, FIVE)
    return result, [e for e in result.events if e.side == "high" and e.swing_index == 2]


def definition(lifetime=None, first_only=False, pen=0.0):
    return SweepDefinition(2, 2, "strict", lifetime, first_only, pen)


class TestStudyStats(unittest.TestCase):
    def test_thin_keeps_minimum_gap(self):
        self.assertEqual(thin([0, 1, 2, 5, 6, 10], 3), [0, 5, 10])
        with self.assertRaises(ValueError):
            thin([2, 1], 1)
        with self.assertRaises(ValueError):
            thin([1], 0)
        self.assertEqual(thin([], 3), [])

    def test_z_helpers(self):
        self.assertAlmostEqual(z_one_proportion(60, 100, 0.5), 2.0)
        self.assertIsNone(z_one_proportion(1, 0, 0.5))
        self.assertIsNone(z_one_proportion(1, 10, 0.0))
        self.assertIsNone(z_two_proportion(0, 10, 0, 10))
        self.assertGreater(z_two_proportion(8, 10, 2, 10), 0)

    def test_runs_and_changes_respect_gaps(self):
        states = ["a", "a", "b", None, "b", "b"]
        self.assertEqual(run_lengths(states), [("a", 2), ("b", 1), ("b", 2)])
        self.assertEqual(run_lengths(["a", "a", "a"], [False, False, True]), [("a", 2), ("a", 1)])
        self.assertEqual(state_changes(states), (1, 3))
        self.assertEqual(state_changes(["a", "b", "a"], [False, False, True]), (1, 1))


class TestRegimeBuildingBlocks(unittest.TestCase):
    def test_ranks_are_causal_midranks_with_warmup(self):
        ranks = causal_percentile_ranks([1.0, 2.0, 3.0, None, 2.0, 10.0], min_history=2)
        self.assertEqual([ranks[0], ranks[1], ranks[3]], [None, None, None])
        self.assertEqual(ranks[2], 1.0)
        # history before index 4 is [1, 2, 3]; value 2 has one below and one equal -> (1+2)/2/3
        self.assertAlmostEqual(ranks[4], 0.5)
        self.assertEqual(ranks[5], 1.0)

    def test_prefix_property_future_values_never_change_earlier_ranks(self):
        values = [float((i * 7) % 11) for i in range(60)]
        full = causal_percentile_ranks(values, 5)
        for cut in (10, 30, 45):
            self.assertEqual(causal_percentile_ranks(values[:cut], 5), full[:cut])

    def test_classify_rank_boundaries(self):
        self.assertIsNone(classify_rank(None, 0.25, 0.75))
        self.assertEqual(classify_rank(0.24, 0.25, 0.75), "low")
        self.assertEqual(classify_rank(0.25, 0.25, 0.75), "mid")
        self.assertEqual(classify_rank(0.75, 0.25, 0.75), "mid")
        self.assertEqual(classify_rank(0.76, 0.25, 0.75), "high")

    def test_contiguous_runs_do_not_fill_gaps(self):
        candles = [candle(0, 1, 2, 0, 1), candle(1, 1, 2, 0, 1), candle(3, 1, 2, 0, 1), candle(4, 1, 2, 0, 1)]
        self.assertEqual(contiguous_run_lengths(candles, FIVE), [1, 2, 1, 2])
        self.assertEqual(len(candles), 4)

    def test_efficiency_matches_hand_calculation_and_respects_gaps(self):
        closes = [10, 11, 10, 12, 13]
        candles = [candle(i, c, c + 1, c - 1, c) for i, c in enumerate(closes)]
        runs = contiguous_run_lengths(candles, FIVE)
        out = compute_feature("efficiency", 4, candles, [], runs, ExpansionConfig())
        # net 3, path 1+1+2+1 = 5
        self.assertEqual(out[:4], [None] * 4)
        self.assertAlmostEqual(out[4], 3 / 5)
        gapped = candles[:2] + [candle(i, c, c + 1, c - 1, c, gap=1) for i, c in enumerate(closes) if i >= 2]
        runs = contiguous_run_lengths(gapped, FIVE)
        out = compute_feature("efficiency", 4, gapped, [], runs, ExpansionConfig())
        self.assertIsNone(out[4])

    def test_config_validation_and_variant_count(self):
        study = RegimeStudyConfig()
        self.assertEqual(len(study.variants()), 14)
        self.assertIsInstance(study.variants()[0], RegimeVariant)
        with self.assertRaises(ValueError):
            RegimeStudyConfig(quantile_pairs=())
        with self.assertRaises(ValueError):
            RegimeStudyConfig(quantile_pairs=((0.8, 0.2),))
        with self.assertRaises(ValueError):
            RegimeStudyConfig(min_history=0)
        narrow = RegimeStudyConfig(quantile_pairs=((0.25, 0.75),))
        self.assertLess(len(narrow.variants()), len(study.variants()))


class TestSweepDetection(unittest.TestCase):
    def test_unlimited_matches_existing_detector(self):
        candles = synthetic(300)
        config = StructureConfig(2, 2)
        structure = analyze_market_structure(candles, config)
        atrs = [None] * len(candles)
        ours = detect_sweeps(candles, structure, atrs, definition(), FIVE).events
        theirs = detect_liquidity_sweeps(candles, structure.swing_highs, structure.swing_lows, config)
        self.assertGreater(len(theirs), 0)
        self.assertEqual(
            sorted((e.side, e.index, e.level) for e in ours),
            sorted((e.side, e.sweep_index, e.level) for e in theirs),
        )

    def test_swing_not_usable_before_confirmation(self):
        _, events = high_events(sweep_series(), definition())
        self.assertEqual(
            [(e.side, e.index, e.swing_index, e.timestamp, e.expected) for e in events],
            [
                ("high", 5, 2, T0 + FIVE * 5, "down"),
                ("high", 6, 2, T0 + FIVE * 6, "down"),
            ],
        )
        self.assertTrue(all(e.index >= 2 + 2 for e in events))
        self.assertEqual(events[0].available_at, T0 + FIVE * 6)
        self.assertEqual(events[0].age, timedelta(0))
        self.assertEqual((events[0].penetration, events[0].prior_sweeps), (1.0, 0))
        self.assertEqual((events[1].penetration, events[1].prior_sweeps), (0.5, 1))

    def test_candidate_sweep_before_confirmation_is_ignored(self):
        # index 4 pokes above the swing: the swing at 2 is never confirmed, so no event can reference it
        series = sweep_series()
        series[4] = candle(4, 10, 16, 9, 10)
        _, events = high_events(series, definition())
        self.assertEqual(events, [])

    def test_lifetime_boundary_and_monotonic_event_count(self):
        series = sweep_series()
        counts = {}
        for name, life in (("5m", FIVE), ("10m", FIVE * 2), ("none", None)):
            _, events = high_events(series, definition(life))
            counts[name] = [e.index for e in events]
        self.assertEqual(counts["5m"], [5])
        self.assertEqual(counts["10m"], [5, 6])
        self.assertEqual(counts["none"], [5, 6])

    def test_lifetime_expires_by_elapsed_time_across_gap(self):
        series = sweep_series()
        series[5:] = [
            Candle(item.timestamp + timedelta(hours=2), item.open, item.high, item.low, item.close, item.tick_volume)
            for item in series[5:]
        ]
        _, short_lived = high_events(series, definition(timedelta(hours=1)))
        _, unlimited = high_events(series, definition())
        self.assertEqual(short_lived, [])
        self.assertEqual([event.index for event in unlimited], [5, 6])

    def test_first_only_and_prior_sweep_count(self):
        _, every = high_events(sweep_series(), definition())
        self.assertEqual([e.prior_sweeps for e in every], [0, 1])
        _, first = high_events(sweep_series(), definition(first_only=True))
        self.assertEqual([e.index for e in first], [5])

    def test_penetration_filter_is_strict_in_atr(self):
        series = sweep_series()
        # penetrations are 1.0 and 0.5 ATR
        self.assertEqual([e.index for e in high_events(series, definition(pen=0.75))[1]], [5])
        self.assertEqual(high_events(series, definition(pen=1.0))[1], [])
        self.assertEqual([e.penetration_atr for e in high_events(series, definition())[1]], [1.0, 0.5])

    def test_touch_and_close_beyond_are_not_sweeps(self):
        touch, events = high_events(sweep_series({6: (10, 15, 9.5, 14)}), definition())
        self.assertEqual([e.index for e in events], [5])
        self.assertGreaterEqual(touch.counts.exact_touches, 1)
        beyond, events = high_events(sweep_series({6: (10, 15.5, 9.5, 15.2)}), definition())
        self.assertEqual([e.index for e in events], [5])
        self.assertGreaterEqual(beyond.counts.close_beyond, 1)
        self.assertGreater(beyond.counts.wick_crossings, beyond.counts.unfiltered_sweeps)

    def test_prefix_causality(self):
        candles = synthetic(300)
        structure = analyze_market_structure(candles, StructureConfig(2, 2))
        atrs = [1.0] * len(candles)
        d = definition(timedelta(hours=1))
        full = detect_sweeps(candles, structure, atrs, d, FIVE).events
        cut = 200
        short_structure = analyze_market_structure(candles[:cut], StructureConfig(2, 2))
        short = detect_sweeps(candles[:cut], short_structure, atrs[:cut], d, FIVE).events
        self.assertEqual([e for e in full if e.index < cut], list(short))

    def test_empty_and_determinism(self):
        structure = analyze_market_structure([], StructureConfig(2, 2))
        empty = detect_sweeps([], structure, [], definition(), FIVE)
        self.assertEqual(empty.events, ())
        self.assertIsNone(empty.active_rate)
        a = high_events(sweep_series(), definition())[0]
        b = high_events(sweep_series(), definition())[0]
        self.assertEqual(a, b)

    def test_lifetime_reduces_always_on_effect(self):
        candles = synthetic(300)
        structure = analyze_market_structure(candles, StructureConfig(2, 2))
        atrs = [1.0] * len(candles)
        short = detect_sweeps(candles, structure, atrs, definition(timedelta(hours=1)), FIVE)
        unlimited = detect_sweeps(candles, structure, atrs, definition(), FIVE)
        self.assertLessEqual(short.mean_live_swings, unlimited.mean_live_swings)
        self.assertLessEqual(len(short.events), len(unlimited.events))

    def test_definition_and_config_validation(self):
        with self.assertRaises(ValueError):
            SweepDefinition(0, 2, "strict", None, False, 0.0)
        with self.assertRaises(ValueError):
            SweepDefinition(2, 2, "bogus", None, False, 0.0)
        with self.assertRaises(ValueError):
            SweepDefinition(2, 2, "strict", timedelta(0), False, 0.0)
        with self.assertRaises(ValueError):
            LiquidityStudyConfig(lifetimes=())
        self.assertEqual(len(LiquidityStudyConfig().definitions()), 64)
        self.assertEqual(LiquidityStudyConfig().reference.name, "sw2x2_str_lifeunlimited_all_pen0")


class TestEventTally(unittest.TestCase):
    def test_streaming_tally_keeps_duplicate_and_cutoff_semantics(self):
        labels = [None] * 8
        for index, expanded, direction, end_index in (
            (0, True, "up", 1),
            (1, False, None, 2),
            (4, True, "down", 6),
            (7, True, "up", 8),
        ):
            timestamp = T0 + FIVE * index
            labels[index] = ExpansionLabel(
                index=index,
                timestamp=timestamp,
                available_at=timestamp + FIVE,
                horizon=FIVE,
                atr_multiple=1.0,
                status="complete",
                reference_close=10.0,
                atr=1.0,
                threshold=1.0,
                window_end=T0 + FIVE * end_index,
                window_candles=1,
                expanded=expanded,
                direction=direction,
                threshold_reached_at=None,
                time_to_threshold=None,
                max_up=None,
                max_down=None,
                mfe=None,
                mae=None,
                end_close_change=None,
                retained_fraction=None,
                outcome="continuation" if expanded else None,
                momentum_profile=None,
            )
        events = [
            PointEvent(index, T0 + FIVE * index, expected)
            for index, expected in ((0, "up"), (1, "up"), (1, "down"), (2, "up"), (4, "down"), (7, "up"))
        ]

        raw, thinned = tally_events(
            events, labels, BaseRates(10, 0.5, 0.4, 0.6), horizon_candles=3, cutoff=T0 + FIVE * 7
        )

        self.assertEqual((raw.n, raw.expanded, raw.aligned), (4, 2, 2))
        self.assertEqual((thinned.n, thinned.expanded, thinned.aligned), (2, 2, 2))
        self.assertAlmostEqual(raw.expected_aligned, 2.0)
        self.assertAlmostEqual(thinned.expected_aligned, 1.0)
        self.assertAlmostEqual(raw.variance, 0.96)
        self.assertAlmostEqual(thinned.variance, 0.48)


class TestFamilies(unittest.TestCase):
    def test_break_after_opposite_sweep_split(self):
        _, events = high_events(sweep_series(), definition())
        structure = analyze_market_structure(sweep_series(), StructureConfig(2, 2))
        swing = structure.swing_highs[0]
        down = StructuralBreak("down", 8, T0 + FIVE * 8, T0 + FIVE * 9, 9.0, 8.5, swing)
        up = StructuralBreak("up", 9, T0 + FIVE * 9, T0 + FIVE * 10, 15.0, 15.5, swing)
        families = build_families(events, [down, up], window=12)
        self.assertEqual([p.index for p in families["break_after_opposite_sweep"]], [8])
        self.assertEqual([p.index for p in families["break_without_recent_sweep"]], [9])
        self.assertEqual(len(families["sweep"]), 2)
        self.assertEqual(len(families["sweep_with_prior_break"]), 0)
        self.assertIsInstance(families["break"][0], PointEvent)


class TestAgentsAndCoordinator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = replace(
            SMALL,
            walk_forward=WalkForwardConfig(timedelta(hours=8), timedelta(hours=3), timedelta(hours=3),
                                           embargo=timedelta(hours=1), min_train_trades=1),
            regime_study=RegimeStudyConfig(min_history=20, min_state_rows=3),
            liquidity_study=LiquidityStudyConfig(
                structures=((2, 2, "strict"),), lifetimes=(timedelta(hours=1), None),
                first_only_options=(False,), min_penetration_atrs=(0.0,), min_events=3, min_pooled_events=5,
            ),
        )
        cls.context = build_context(synthetic(400), cls.config)

    def test_agents_run_and_are_deterministic(self):
        for agent in (MarketRegimeAgent(), LiquidityStructureAgent()):
            first = agent.run(self.context)
            second = agent.run(self.context)
            self.assertEqual(first.status, "ok", first.error)
            self.assertEqual(first, second)
            self.assertTrue(first.limitations)
            self.assertTrue(first.lines)

    def test_liquidity_agent_reports_cross_check_and_variant_count(self):
        result = LiquidityStructureAgent().run(self.context)
        text = "\n".join(result.lines)
        self.assertIn("IDENTICAL", text)
        self.assertIn("Definitions tried: 2", text)

    def test_regime_agent_reports_variant_count_and_exclusions(self):
        result = MarketRegimeAgent().run(self.context)
        text = "\n".join((result.headline, *result.lines))
        self.assertIn("14", text)

    def test_empty_and_tiny_datasets_do_not_crash(self):
        with self.assertRaises(ValueError):
            build_context([], self.config)
        for count in (3, 30):
            context = build_context(synthetic(count), self.config)
            for agent in (MarketRegimeAgent(), LiquidityStructureAgent()):
                result = agent.run(context)
                self.assertIn(result.status, ("ok", "error"))
                self.assertIsInstance(result, AgentResult)

    def test_failures_are_isolated_and_order_is_kept(self):
        agents = [FakeAgent("first"), MarketRegimeAgent(), FakeAgent("bad", fail=True), LiquidityStructureAgent()]
        report = run_agents(self.context, agents)
        self.assertEqual([r.agent for r in report.results],
                         ["first", MarketRegimeAgent.name, "bad", LiquidityStructureAgent.name])
        self.assertEqual(report.results[2].status, "error")
        self.assertEqual(report.results[1].status, "ok")
        self.assertEqual(report.results[3].status, "ok")

    def test_default_agents_order_and_suite_defaults(self):
        names = [a.name for a in default_agents()]
        self.assertEqual(names[1:3], [MarketRegimeAgent.name, LiquidityStructureAgent.name])
        self.assertEqual(len(names), 4)
        suite = SuiteConfig()
        self.assertIsInstance(suite.regime_study, RegimeStudyConfig)
        self.assertIsInstance(suite.liquidity_study, LiquidityStudyConfig)


if __name__ == "__main__":
    unittest.main()
