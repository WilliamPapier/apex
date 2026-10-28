import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config.settings import SETTINGS
from core.market_data import Candle
from research.backtest import BacktestConfig, CostModel, ExitRule
from research.rule_spec import (
    EvidenceRecord,
    PromotionGate,
    RuleSpec,
    check_promotion,
    evidence_from_walk_forward,
    save_spec,
)
from research.walk_forward import WalkForwardResult
from research.backtest import summarize_trades


UTC = timezone.utc
STRONG = EvidenceRecord(
    data_start=datetime(2024, 1, 1, tzinfo=UTC), data_end=datetime(2025, 9, 30, tzinfo=UTC),
    distinct_months=21, folds=10, folds_with_selection=9, oos_trades=250, candidates_evaluated=12,
    oos_expectancy_r=0.12, oos_profit_factor=1.2, oos_max_drawdown_fraction=0.1,
    spread=0.25, slippage_per_side=0.05, commission_per_side=0.0,
    regimes_tested=("low_vol", "high_vol"), limitations=("synthetic test fixture",),
)
WEAK = replace(STRONG, distinct_months=1, folds_with_selection=1, oos_trades=12, regimes_tested=())


def spec(**kwargs):
    base = dict(
        rule_id="exp_volume_spike", version=1, status="experimental", source="user_hypothesis",
        hypothesis="Tick-volume spikes precede ATR expansions.",
        rationale="Task 49 descriptive lift; unproven.",
        entry_conditions=("tick_volume_ratio >= 2.0 at candle close",),
        exit_conditions=("stop 1.0 ATR", "target 1.5R", "time exit 6 candles"),
        parameters=(("target_r", 1.5), ("stop_atr_multiple", 1.0)),
    )
    base.update(kwargs)
    return RuleSpec(**base)


class TestPromotionGate(unittest.TestCase):
    def test_strong_evidence_passes_and_weak_evidence_lists_every_reason(self):
        self.assertEqual(check_promotion(STRONG), ())

        failures = " | ".join(check_promotion(WEAK))

        for fragment in ("1 calendar month", "1 walk-forward", "12 out-of-sample trades", "regimes"):
            self.assertIn(fragment, failures)

    def test_each_economic_requirement_is_enforced(self):
        cases = {
            "no spread": replace(STRONG, spread=0.0),
            "expectancy": replace(STRONG, oos_expectancy_r=0.0),
            "missing expectancy": replace(STRONG, oos_expectancy_r=None),
            "profit factor": replace(STRONG, oos_profit_factor=0.9),
            "drawdown": replace(STRONG, oos_max_drawdown_fraction=0.3),
            "trials": replace(STRONG, candidates_evaluated=0),
        }
        for name, evidence in cases.items():
            with self.subTest(name):
                self.assertTrue(check_promotion(evidence))
        self.assertEqual(check_promotion(replace(STRONG, oos_trades=10), PromotionGate(min_oos_trades=10)), ())


class TestRuleSpec(unittest.TestCase):
    def test_status_ladder_and_separation_of_experimental_from_approved(self):
        self.assertEqual(spec().status, "experimental")
        with self.assertRaisesRegex(ValueError, "evidence"):
            spec(status="candidate")
        with self.assertRaisesRegex(ValueError, "Promotion gate not met"):
            spec(status="candidate", evidence=WEAK)
        candidate = spec(status="candidate", evidence=STRONG)
        self.assertEqual(candidate.status, "candidate")
        with self.assertRaisesRegex(ValueError, "approver"):
            spec(status="approved", evidence=STRONG)
        with self.assertRaises(ValueError):
            spec(status="candidate", evidence=STRONG, approved_by="someone")
        self.assertEqual(spec(status="approved", evidence=STRONG, approved_by="owner").status, "approved")

    def test_no_spec_can_enable_live_trading_or_exceed_risk_limits(self):
        for s in (spec(), spec(status="approved", evidence=STRONG, approved_by="owner")):
            self.assertFalse(s.live_trading_allowed)
        self.assertFalse(SETTINGS.live_trading_enabled)
        for kwargs in (dict(risk_fraction=0.01), dict(risk_fraction=0), dict(max_open_positions=2)):
            with self.assertRaises(ValueError):
                spec(**kwargs)

    def test_validation_of_required_fields(self):
        for kwargs in (
            dict(rule_id=""), dict(rule_id="bad id!"), dict(version=0), dict(status="live"),
            dict(source="guess"), dict(hypothesis=" "), dict(entry_conditions=()),
            dict(exit_conditions=()),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                spec(**kwargs)

    def test_serialisation_and_hash_are_stable_and_version_sensitive(self):
        a, b = spec(), spec()

        self.assertEqual(a.content_hash(), b.content_hash())
        self.assertNotEqual(a.content_hash(), spec(version=2).content_hash())
        data = json.loads(spec(status="candidate", evidence=STRONG).to_json())
        self.assertEqual(data["evidence"]["data_start"], "2024-01-01T00:00:00+00:00")
        self.assertEqual(data["risk_fraction"], 0.005)
        self.assertEqual(data["max_open_positions"], 1)

    def test_save_is_exclusive_and_never_targets_historical_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_spec(spec(), directory)
            self.assertEqual(path.name, "exp_volume_spike_v1.json")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_spec(spec(), directory)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.read_text())["rule_id"], "exp_volume_spike")
            self.assertTrue(save_spec(spec(version=2), directory).exists())
            historical = Path(directory) / "historical"
            historical.mkdir()
            with self.assertRaises(ValueError):
                save_spec(spec(version=3), historical)
            with self.assertRaises(ValueError):
                save_spec(spec(version=3), Path(directory) / "missing")


class TestEvidenceFromWalkForward(unittest.TestCase):
    def test_one_month_of_data_cannot_reach_candidate_status(self):
        candles = [Candle(datetime(2025, 9, d, tzinfo=UTC), 1, 1, 1, 1, 1) for d in (1, 15, 30)]
        config = BacktestConfig(ExitRule(1, 1, 1), CostModel(spread=0.3))
        result = WalkForwardResult((), (), summarize_trades((), 10_000.0, 0.005), 7, 0)

        evidence = evidence_from_walk_forward(result, candles, config, regimes_tested=("all",))

        self.assertEqual((evidence.distinct_months, evidence.candidates_evaluated, evidence.spread), (1, 7, 0.3))
        failures = " ".join(check_promotion(evidence))
        self.assertIn("1 calendar month", failures)
        with self.assertRaises(ValueError):
            spec(status="candidate", evidence=evidence)
        with self.assertRaises(ValueError):
            evidence_from_walk_forward(result, [], config)


if __name__ == "__main__":
    unittest.main()
