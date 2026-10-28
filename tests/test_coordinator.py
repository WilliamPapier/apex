import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import sin
from pathlib import Path

from core.market_data import Candle
from data.historical_data_loader import HistoricalDataError
from data.mt5_history_collector import write_candles_csv
from research.agents import CandidateStrategyAgent, PriceExpansionAgent
from research.backtest import CostModel
from research.candidates import CandidateGrid
from research.coordinator import (
    AgentResult,
    RegimeConfig,
    SuiteConfig,
    build_context,
    load_context,
    run_agents,
)
from research.market_structure import StructureConfig
from research.price_expansion import ExpansionConfig
from research.walk_forward import WalkForwardConfig


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)


def synthetic(count=400):
    """Deterministic wavy series with varying range and tick volume."""

    candles, price, seed = [], 100.0, 12345
    for i in range(count):
        seed = (seed * 1103515245 + 12345) % 2**31
        noise = (seed % 1000) / 1000 - 0.5
        open_ = price
        close = price + 0.8 * sin(i / 7) + noise
        spread = 0.2 + (seed // 7 % 100) / 100
        candles.append(Candle(BASE + timedelta(minutes=5 * i), open_, max(open_, close) + spread,
                              min(open_, close) - spread, close, 100 + seed % 50))
        price = close
    return candles


SMALL = SuiteConfig(
    expansion=ExpansionConfig(structure=StructureConfig(2, 2), horizons=(timedelta(minutes=5), timedelta(minutes=15)),
                              atr_multiples=(1.0, 2.0)),
    grid=CandidateGrid(momentum_atr=(0.5,), fade_range_atr=(1.5,), volume_ratios=(0.0,),
                       stop_atr_multiples=(1.0,), target_rs=(1.0, 2.0)),
    walk_forward=WalkForwardConfig(timedelta(hours=8), timedelta(hours=3), timedelta(hours=3),
                                   embargo=timedelta(hours=1), min_train_trades=1),
    min_regime_trades=1,
)


class FakeAgent:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def run(self, context):
        if self.fail:
            raise RuntimeError("boom")
        return AgentResult(self.name, "ok", f"{len(context.candles)} candles")


class TestRegimes(unittest.TestCase):
    def test_boundaries_are_inclusive_of_normal(self):
        regimes = RegimeConfig(contracting_below=0.4, expanding_above=0.65)

        self.assertEqual(regimes.classify(0.39), "contracting")
        self.assertEqual(regimes.classify(0.4), "normal")
        self.assertEqual(regimes.classify(0.65), "normal")
        self.assertEqual(regimes.classify(0.66), "expanding")
        self.assertEqual(regimes.classify(None), "unavailable")
        with self.assertRaises(ValueError):
            RegimeConfig(0.7, 0.6)


class TestCoordinator(unittest.TestCase):
    def test_runs_agents_in_order_and_isolates_failures(self):
        context = build_context(synthetic(60), SMALL)

        report = run_agents(context, [FakeAgent("a"), FakeAgent("b", fail=True), FakeAgent("c")])

        self.assertEqual([r.agent for r in report.results], ["a", "b", "c"])
        self.assertEqual([r.status for r in report.results], ["ok", "error", "ok"])
        self.assertEqual(report.results[1].error, "RuntimeError: boom")
        self.assertFalse(report.all_ok)
        self.assertIn("ERROR: RuntimeError: boom", report.to_text())
        self.assertEqual(report.candle_count, 60)

    def test_rejects_bad_agent_lists_and_empty_data(self):
        context = build_context(synthetic(60), SMALL)
        with self.assertRaises(ValueError):
            run_agents(context, [])
        with self.assertRaises(ValueError):
            run_agents(context, [FakeAgent("a"), FakeAgent("a")])
        with self.assertRaises(ValueError):
            build_context([], SMALL)

    def test_load_context_reads_a_csv_and_reports_invalid_files(self):
        candles = synthetic(40)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            write_candles_csv(candles, path)
            context = load_context(path, SMALL, limit_candles=25)
            bad = Path(folder) / "bad.csv"
            bad.write_text("timestamp_utc,open\n", encoding="utf-8")

            self.assertEqual(len(context.candles), 25)
            self.assertIn("first 25 candles only", context.source)
            with self.assertRaises(HistoricalDataError):
                load_context(bad, SMALL)
            with self.assertRaises(ValueError):
                load_context(path, SMALL, limit_candles=0)


class TestAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candles = synthetic(400)
        cls.context = build_context(cls.candles, SMALL)

    def test_price_expansion_agent_reports_without_changing_inputs(self):
        before = tuple(self.candles)

        result = PriceExpansionAgent().run(self.context)

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.data["summaries"]), 4)
        self.assertEqual(len(result.data["flag_evaluations"]), 2 * 6 * 2)
        self.assertTrue(any("FUTURE candles" in line for line in result.lines))
        self.assertEqual(tuple(self.candles), before)
        self.assertEqual(result.lines, PriceExpansionAgent().run(self.context).lines)  # deterministic

    def test_candidate_agent_counts_variants_rejects_and_keeps_candidates_experimental(self):
        result = CandidateStrategyAgent().run(self.context)

        self.assertEqual(result.status, "ok")
        # 1 momentum x2 exits + 1 fade x2 + sweep (2 confirm options) x2 + controls (2) x2 + 1 user hypothesis
        self.assertEqual(result.data["variants_generated"], 13)
        self.assertEqual([n for n, _ in result.data["generation"].rejected], ["user_sweep_mss_ifvg_keylevel"])
        evidence = result.data["evidence"]
        self.assertEqual(evidence.candidates_evaluated, 8)
        self.assertEqual(evidence.distinct_months, 1)
        self.assertTrue(any("calendar month" in reason for reason in result.data["gate_failures"]))
        self.assertIn("NOT met", "\n".join(result.lines))
        self.assertTrue(any(line.startswith("Robustness: expectancy t-statistic") for line in result.lines))
        self.assertIn("t_statistic", result.data)
        self.assertTrue(result.data["walk_forward"].folds)

    def test_candidate_agent_is_deterministic_and_charges_costs(self):
        first = CandidateStrategyAgent().run(self.context)
        second = CandidateStrategyAgent().run(self.context)

        self.assertEqual(first.lines, second.lines)
        wf = first.data["walk_forward"]
        self.assertGreater(len(wf.oos_trades), 0)
        self.assertTrue(all(t.cost_r > 0 for t in wf.oos_trades))
        self.assertEqual(first.data["evidence"].spread, SMALL.costs.spread)

    def test_zero_spread_is_flagged_by_the_gate(self):
        config = replace(SMALL, costs=CostModel(spread=0.0))
        result = CandidateStrategyAgent().run(build_context(self.candles, config))

        self.assertIn("no spread was modelled", result.data["gate_failures"])

    def test_too_little_data_gives_no_folds_and_no_trades(self):
        result = CandidateStrategyAgent().run(build_context(synthetic(60), SMALL))

        self.assertEqual(result.data["walk_forward"].folds, ())
        self.assertEqual(result.data["walk_forward"].oos_trades, ())
        self.assertTrue(result.data["gate_failures"])

    def test_embargo_shorter_than_holding_time_is_an_error_result(self):
        config = replace(SMALL, walk_forward=replace(SMALL.walk_forward, embargo=timedelta(minutes=5)))
        report = run_agents(build_context(self.candles, config), [CandidateStrategyAgent()])

        self.assertEqual(report.results[0].status, "error")
        self.assertIn("embargo", report.results[0].error)

    def test_a_later_candle_change_does_not_alter_earlier_signals(self):
        changed = list(self.candles)
        last = changed[-1]
        changed[-1] = Candle(last.timestamp, last.open, last.high + 50, last.low - 50, last.close + 40, last.tick_volume)
        other = build_context(changed, SMALL)

        for a, b in zip(self.context.expansion.snapshots[:-1], other.expansion.snapshots[:-1]):
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
