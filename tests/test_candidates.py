import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from core.market_data import Candle
from research.backtest import ExitRule
from research.candidates import (
    CandidateGrid,
    CandidateSpec,
    build_signal,
    generate_candidates,
    rejection_reasons,
    to_experimental_rule_spec,
)
from research.market_structure import StructureConfig
from research.price_expansion import ExpansionConfig, build_predictors


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
CFG = ExpansionConfig(structure=StructureConfig(1, 1), atr_period=2, volatility_periods=(1, 2),
                      compression_short=2, compression_long=3, volume_lookback=2,
                      momentum_candles=1, sequence_candles=2)
EXIT = ExitRule(1.0, 2.0, 12)


def base_snapshot():
    candles = [Candle(BASE + timedelta(minutes=5 * i), 10, 11, 9, 10, 10) for i in range(5)]
    return build_predictors(candles, CFG)[-1]


def spec(family, params, **kwargs):
    return CandidateSpec(name="x", family=family, hypothesis="h", source="data_discovered",
                         params=tuple(params.items()), exit=kwargs.pop("exit", EXIT), **kwargs)


class TestGeneration(unittest.TestCase):
    def test_default_grid_counts_every_variant_and_rejects_the_unresolved_hypothesis(self):
        result = generate_candidates()

        self.assertEqual(result.variants_generated, 57)  # 24 momentum + 16 fade + 8 sweep + 8 control + 1 user
        self.assertEqual(len(result.accepted), 56)
        self.assertEqual([name for name, _ in result.rejected], ["user_sweep_mss_ifvg_keylevel"])
        reasons = result.rejected[0][1]
        self.assertTrue(any("unresolved: key level" in r for r in reasons))
        self.assertIn("no exit rule (stop, target, maximum holding time) defined", reasons)
        self.assertTrue(any("no signal builder" in r for r in reasons))

    def test_generation_is_deterministic_with_unique_names(self):
        first = [g.spec.name for g in generate_candidates().accepted]
        second = [g.spec.name for g in generate_candidates().accepted]

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_small_grid_and_optional_families(self):
        grid = CandidateGrid(momentum_atr=(1.0,), fade_range_atr=(2.0,), volume_ratios=(0.0,),
                             stop_atr_multiples=(1.0,), target_rs=(1.0,),
                             include_controls=False, include_sweep_baseline=False,
                             include_user_hypothesis=False)

        result = generate_candidates(grid)

        self.assertEqual(result.variants_generated, 2)
        self.assertEqual(result.rejected, ())

    def test_empty_grid_dimension_is_invalid(self):
        with self.assertRaises(ValueError):
            CandidateGrid(momentum_atr=())


class TestRejection(unittest.TestCase):
    def test_incomplete_candidates_are_rejected_not_filled_in(self):
        self.assertEqual(rejection_reasons(spec("momentum_continuation",
                         {"min_momentum_atr": 1.0, "min_tick_volume_ratio": 0.0})), ())
        self.assertIn("missing signal parameter 'min_tick_volume_ratio'",
                      rejection_reasons(spec("momentum_continuation", {"min_momentum_atr": 1.0})))
        self.assertIn("no exit rule (stop, target, maximum holding time) defined",
                      rejection_reasons(spec("control_always_long", {}, exit=None)))
        self.assertTrue(any("must be a non-negative number" in r for r in rejection_reasons(
            spec("mean_reversion_fade", {"min_range_atr": -1.0, "min_tick_volume_ratio": 0.0}))))
        self.assertTrue(any("unresolved: stop" in r for r in rejection_reasons(
            spec("control_always_long", {}, unresolved=("stop rule",)))))
        self.assertTrue(any("unknown source" in r for r in rejection_reasons(
            replace(spec("control_always_long", {}), source="guess"))))

    def test_build_signal_refuses_incomplete_candidates(self):
        with self.assertRaises(ValueError):
            build_signal(spec("momentum_continuation", {}))


class TestSignals(unittest.TestCase):
    def test_momentum_boundaries_and_volume_gate(self):
        signal = build_signal(spec("momentum_continuation", {"min_momentum_atr": 1.0, "min_tick_volume_ratio": 1.5}))
        s = base_snapshot()

        self.assertEqual(signal(replace(s, momentum_to_atr=1.0, tick_volume_ratio=1.5)), "long")
        self.assertEqual(signal(replace(s, momentum_to_atr=-1.0, tick_volume_ratio=2.0)), "short")
        self.assertIsNone(signal(replace(s, momentum_to_atr=0.99, tick_volume_ratio=2.0)))
        self.assertIsNone(signal(replace(s, momentum_to_atr=2.0, tick_volume_ratio=1.49)))
        self.assertIsNone(signal(replace(s, momentum_to_atr=2.0, tick_volume_ratio=None)))
        self.assertIsNone(signal(replace(s, momentum_to_atr=None, tick_volume_ratio=2.0)))

    def test_fade_trades_against_the_large_candle(self):
        signal = build_signal(spec("mean_reversion_fade", {"min_range_atr": 2.0, "min_tick_volume_ratio": 0.0}))
        s = base_snapshot()

        self.assertEqual(signal(replace(s, range_to_atr=2.0, direction="bullish")), "short")
        self.assertEqual(signal(replace(s, range_to_atr=3.0, direction="bearish")), "long")
        self.assertIsNone(signal(replace(s, range_to_atr=3.0, direction="neutral")))
        self.assertIsNone(signal(replace(s, range_to_atr=1.99, direction="bullish")))
        self.assertIsNone(signal(replace(s, range_to_atr=None, direction="bullish")))

    def test_sweep_baseline_and_controls(self):
        plain = build_signal(spec("sweep_reversal_baseline", {"require_confirming_candle": 0}))
        confirmed = build_signal(spec("sweep_reversal_baseline", {"require_confirming_candle": 1}))
        s = base_snapshot()
        low = replace(s, recent_low_sweeps=1, recent_high_sweeps=0, direction="bearish")

        self.assertEqual(plain(low), "long")
        self.assertIsNone(confirmed(low))
        self.assertEqual(confirmed(replace(low, direction="bullish")), "long")
        self.assertEqual(plain(replace(s, recent_high_sweeps=2, recent_low_sweeps=0)), "short")
        self.assertIsNone(plain(replace(s, recent_high_sweeps=1, recent_low_sweeps=1)))
        self.assertIsNone(plain(s))
        self.assertEqual(build_signal(spec("control_always_long", {}))(s), "long")
        self.assertEqual(build_signal(spec("control_always_short", {}))(s), "short")


class TestExperimentalSpec(unittest.TestCase):
    def test_runnable_candidate_becomes_experimental_and_never_live(self):
        generated = generate_candidates().accepted[0]

        rule = to_experimental_rule_spec(generated.spec)

        self.assertEqual((rule.status, rule.source, rule.approved_by), ("experimental", "data_discovered", ""))
        self.assertFalse(rule.live_trading_allowed)
        self.assertEqual(rule.risk_fraction, 0.005)
        self.assertIn(("max_hold_candles", 12), rule.parameters)

    def test_unresolved_hypothesis_cannot_become_a_spec(self):
        user = next(s for s in __import__("research.candidates", fromlist=["candidate_specs"])
                    .candidate_specs(CandidateGrid()) if s.source == "user_hypothesis")

        with self.assertRaises(ValueError):
            to_experimental_rule_spec(user)


if __name__ == "__main__":
    unittest.main()
