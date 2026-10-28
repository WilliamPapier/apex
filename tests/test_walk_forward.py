import unittest
from datetime import datetime, timedelta, timezone

from core.market_data import Candle
from research.backtest import BacktestConfig, CostModel, ExitRule
from research.market_structure import StructureConfig
from research.price_expansion import ExpansionConfig, build_predictors
from research.walk_forward import (
    RuleCandidate,
    WalkForwardConfig,
    make_folds,
    walk_forward_evaluate,
)


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
FIVE = timedelta(minutes=5)
CFG = ExpansionConfig(structure=StructureConfig(1, 1), atr_period=2, volatility_periods=(1, 2),
                      compression_short=2, compression_long=3, volume_lookback=2,
                      momentum_candles=1, sequence_candles=2)
WF = WalkForwardConfig(
    train_span=timedelta(minutes=100), test_span=timedelta(minutes=50),
    step=timedelta(minutes=50), embargo=timedelta(minutes=15), min_train_trades=1,
)
TEMPLATE = BacktestConfig(ExitRule(1.0, 1.0, 3), CostModel(spread=0.0))


def trend(count, flip_at=None, shifts=None):
    """Up-trend candles (ATR 0.7), switching to a down-trend at ``flip_at``."""
    candles, price, shift = [], 100.0, 0
    for i in range(count):
        shift += (shifts or {}).get(i, 0)
        up = flip_at is None or i < flip_at
        o, c = price, price + (0.5 if up else -0.5)
        h, l = (o + 0.6, o - 0.1) if up else (o + 0.1, o - 0.6)
        candles.append(Candle(BASE + timedelta(minutes=5 * i + shift), o, h, l, c, 10))
        price = c
    return candles


LONG = RuleCandidate("long", lambda s: "long", ExitRule(1.0, 1.0, 3))
SHORT = RuleCandidate("short", lambda s: "short", ExitRule(1.0, 1.0, 3))


class TestFolds(unittest.TestCase):
    def test_fold_boundaries_are_hand_calculated(self):
        folds = make_folds(trend(60), WF, FIVE)

        self.assertEqual(
            [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds],
            [(0, 20, 23, 33), (10, 30, 33, 43), (20, 40, 43, 53)],
        )
        self.assertEqual(folds[0].test_from, BASE + timedelta(minutes=115))
        for f in folds:
            self.assertLess(f.train_until + WF.embargo, f.test_until)
            self.assertEqual(f.test_from, f.train_until + WF.embargo)

    def test_folds_are_time_based_so_gaps_are_not_filled(self):
        folds = make_folds(trend(60, shifts={10: 25}), WF, FIVE)

        self.assertEqual(folds[0].train_end, 20 - 5)  # five candles missing inside the window
        self.assertEqual(len(folds), 4)  # data now spans 325 minutes, so a fourth fold fits

    def test_short_or_empty_data_gives_no_folds(self):
        self.assertEqual(make_folds(trend(30), WF, FIVE), ())
        self.assertEqual(make_folds([], WF, FIVE), ())

    def test_invalid_configuration(self):
        for kwargs in (
            dict(train_span=timedelta(0)), dict(test_span=timedelta(minutes=-1)),
            dict(step=timedelta(0)), dict(embargo=timedelta(minutes=-1)),
            dict(min_train_trades=0), dict(train_span=5),
        ):
            base = dict(train_span=FIVE, test_span=FIVE, step=FIVE)
            base.update(kwargs)
            with self.assertRaises(ValueError):
                WalkForwardConfig(**base)


class TestEvaluation(unittest.TestCase):
    def evaluate(self, candles, candidates=(LONG, SHORT), wf=WF):
        return walk_forward_evaluate(candles, build_predictors(candles, CFG), list(candidates), wf, TEMPLATE)

    def test_selects_on_training_and_trades_only_inside_test_windows(self):
        result = self.evaluate(trend(60))

        self.assertEqual(result.candidates_evaluated, 2)
        self.assertEqual(result.folds_with_selection, 3)
        for fold_result in result.folds:
            self.assertEqual(fold_result.selected, "long")
            train = dict(fold_result.train_metrics)
            self.assertEqual((train["long"].expectancy_r, train["short"].expectancy_r), (1.0, -1.0))
            f = fold_result.fold
            assert fold_result.test is not None
            for trade in fold_result.test.trades:
                self.assertTrue(f.test_start <= trade.entry_index < f.test_end)
                self.assertLessEqual(trade.exit_known_at, f.test_until)
        self.assertEqual(result.oos_metrics.expectancy_r, 1.0)
        self.assertEqual(result.oos_trades, tuple(t for r in result.folds for t in (r.test.trades if r.test else ())))

    def test_selection_cannot_see_the_test_window(self):
        result = self.evaluate(trend(60, flip_at=40))
        last = result.folds[2]

        self.assertEqual(last.selected, "long")  # chosen on a pure up-trend training window
        assert last.test is not None
        self.assertEqual(dict(last.train_metrics)["long"].expectancy_r, 1.0)
        self.assertLess(last.test.metrics.expectancy_r, 0)  # the later down-trend is not anticipated

    def test_no_eligible_candidate_means_no_trades(self):
        strict = WalkForwardConfig(WF.train_span, WF.test_span, WF.step, WF.embargo, min_train_trades=500)

        result = self.evaluate(trend(60), wf=strict)

        self.assertEqual((result.folds_with_selection, result.oos_metrics.trades), (0, 0))
        self.assertTrue(all(r.selected is None and r.test is None for r in result.folds))

    def test_ties_choose_the_first_candidate_and_runs_are_deterministic(self):
        twin = RuleCandidate("twin", LONG.signal, LONG.exit)
        first = self.evaluate(trend(60), candidates=(twin, LONG))

        self.assertTrue(all(r.selected == "twin" for r in first.folds))
        self.assertEqual(first, self.evaluate(trend(60), candidates=(twin, LONG)))

    def test_invalid_inputs(self):
        candles = trend(60)
        snaps = build_predictors(candles, CFG)
        with self.assertRaises(ValueError):
            walk_forward_evaluate(candles, snaps, [], WF, TEMPLATE)
        with self.assertRaises(ValueError):
            walk_forward_evaluate(candles, snaps, [LONG, LONG], WF, TEMPLATE)
        slow = RuleCandidate("slow", LONG.signal, ExitRule(1.0, 1.0, 4))  # holds 20 min > 15 min embargo
        with self.assertRaises(ValueError):
            walk_forward_evaluate(candles, snaps, [slow], WF, TEMPLATE)
        with self.assertRaises(ValueError):
            walk_forward_evaluate(candles, snaps, [LONG], "bad", TEMPLATE)


if __name__ == "__main__":
    unittest.main()
