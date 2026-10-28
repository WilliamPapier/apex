import unittest
from datetime import datetime, timedelta, timezone

from core.market_data import Candle
from research.backtest import (
    BacktestConfig,
    CostModel,
    ExitRule,
    simulate_backtest,
    summarize_trades,
)
from research.market_structure import StructureConfig
from research.price_expansion import ExpansionConfig, build_predictors


UTC = timezone.utc
BASE = datetime(2025, 9, 1, tzinfo=UTC)
CFG = ExpansionConfig(structure=StructureConfig(1, 1), atr_period=2, volatility_periods=(1, 2),
                      compression_short=2, compression_long=3, volume_lookback=2,
                      momentum_candles=1, sequence_candles=2)
PREFIX = [(10, 11, 9, 10)] * 3  # ATR(2) is 2.0 from index 1 on
ZERO = CostModel(spread=0.0)


def series(rows, shifts=None):
    candles, shift = [], 0
    for i, (o, h, l, c) in enumerate(rows):
        shift += (shifts or {}).get(i, 0)
        candles.append(Candle(BASE + timedelta(minutes=5 * i + shift), o, h, l, c, 10))
    return candles


def config(hold=3, target_r=1.5, costs=ZERO, **kwargs):
    return BacktestConfig(ExitRule(1.0, target_r, hold), costs, **kwargs)


def run(rows, signal_indices=(2,), side="long", cfg=None, shifts=None, **kwargs):
    candles = series(rows, shifts)
    snaps = build_predictors(candles, CFG)
    rule = lambda s: side if s.index in signal_indices else None
    return simulate_backtest(candles, snaps, rule, cfg or config(), **kwargs)


class TestExecution(unittest.TestCase):
    def test_target_stop_and_same_candle_tie(self):
        win = run(PREFIX + [(10, 13.5, 9.5, 13)]).trades[0]
        self.assertEqual((win.exit_reason, win.entry_price, win.stop_price, win.target_price),
                         ("target", 10, 8, 13))
        self.assertEqual((win.net_r, win.gross_r, win.cost_r, win.entry_index), (1.5, 1.5, 0, 3))
        self.assertEqual(win.entry_time, BASE + timedelta(minutes=15))
        self.assertEqual(win.signal_available_at, win.entry_time)
        self.assertEqual(win.exit_known_at, BASE + timedelta(minutes=20))
        self.assertEqual(run(PREFIX + [(10, 10.5, 7.5, 8)]).trades[0].net_r, -1)
        tie = run(PREFIX + [(10, 13.5, 7.5, 10)]).trades[0]
        self.assertEqual((tie.exit_reason, tie.net_r), ("stop", -1))  # stop assumed first

    def test_short_mirrors_long(self):
        trade = run(PREFIX + [(10, 10.5, 6.5, 7)], side="short").trades[0]
        self.assertEqual((trade.exit_reason, trade.stop_price, trade.target_price, trade.net_r),
                         ("target", 12, 7, 1.5))
        stop = run(PREFIX + [(10, 12.5, 9.5, 12)], side="short").trades[0]
        self.assertEqual((stop.exit_reason, stop.net_r), ("stop", -1))

    def test_gap_through_stop_fills_at_open_and_flags_the_gap(self):
        rows = PREFIX + [(10, 10.2, 9.8, 10), (6, 7, 5, 6)]

        trade = run(rows, shifts={4: 60}).trades[0]

        self.assertEqual((trade.exit_reason, trade.exit_price, trade.net_r), ("stop", 6, -2))
        self.assertTrue(trade.spans_gap)
        self.assertFalse(run(rows).trades[0].spans_gap)  # same data, no gap

    def test_time_exit_and_unfinished_trade(self):
        rows = PREFIX + [(10, 10.5, 9.5, 10), (10, 10.5, 9.5, 10.2), (10.2, 10.8, 9.6, 10.5)]

        trade = run(rows).trades[0]

        self.assertEqual((trade.exit_reason, trade.candles_held, trade.net_r), ("time", 3, 0.25))
        cut = run(rows[:5])  # data ends before the third holding candle
        self.assertEqual((len(cut.trades), cut.unfinished), (0, 1))

    def test_no_fill_across_gap_missing_atr_and_ignored_signals(self):
        rows = PREFIX + [(10, 13.5, 9.5, 13)]
        self.assertEqual(run(rows, shifts={3: 60}).skipped_gap_before_entry, 1)
        self.assertEqual(len(run(rows, shifts={3: 60}).trades), 0)
        self.assertEqual(run(rows, signal_indices=(0,)).skipped_no_atr, 1)
        busy = PREFIX + [(10, 10.5, 9.5, 10), (10, 10.5, 9.5, 10.2), (10.2, 10.8, 9.6, 10.5),
                         (10.5, 10.9, 10.1, 10.6)]
        result = run(busy, signal_indices=(2, 3, 4))
        self.assertEqual((result.signals, result.ignored_position_open, len(result.trades)), (3, 2, 1))

    def test_costs_and_position_size_hand_calculated(self):
        costs = CostModel(spread=0.4, slippage_per_side=0.1, commission_per_side=0.05)  # 0.7 total
        cfg = config(costs=costs)

        win = run(PREFIX + [(10, 13.5, 9.5, 13)], cfg=cfg).trades[0]
        loss = run(PREFIX + [(10, 10.5, 7.5, 8)], cfg=cfg).trades[0]

        self.assertAlmostEqual(win.risk_distance, 2.7)
        self.assertAlmostEqual(win.net_r, 2.3 / 2.7)
        self.assertAlmostEqual(win.gross_r, 3 / 2.7)
        self.assertAlmostEqual(win.cost_r, 0.7 / 2.7)
        self.assertAlmostEqual(loss.net_r, -1.0)  # a stop fill costs exactly 1R including costs
        self.assertAlmostEqual(win.risk_amount, 50.0)  # 0.5% of 10,000
        self.assertAlmostEqual(win.lots, 50 / 2.7)
        self.assertAlmostEqual(CostModel(0.4, 0.1, 0.05).round_trip, 0.7)

    def test_balance_compounds_and_metrics_hand_calculated(self):
        rows = PREFIX + [(10, 13.5, 9.5, 13), (13, 13.2, 9.5, 10)]  # +1.5R then -1R (ATR 3 stop)

        result = run(rows, signal_indices=(2, 3))
        first, second = result.trades

        self.assertEqual((first.net_r, second.net_r), (1.5, -1))
        self.assertAlmostEqual(first.balance_after, 10075.0)
        self.assertAlmostEqual(second.balance_after, 10024.625)
        m = result.metrics
        self.assertEqual((m.trades, m.wins, m.losses, m.win_rate), (2, 1, 1, 0.5))
        self.assertAlmostEqual(m.expectancy_r, 0.25)
        self.assertAlmostEqual(m.profit_factor, 1.5)
        self.assertAlmostEqual(m.max_drawdown_fraction, 50.375 / 10075)
        self.assertEqual(m.max_consecutive_losses, 1)
        self.assertAlmostEqual(m.final_balance, second.balance_after)
        again = summarize_trades(result.trades, 10_000.0, 0.005)
        self.assertEqual(again, m)
        self.assertEqual(summarize_trades((), 10_000.0, 0.005).expectancy_r, None)

    def test_window_end_hides_later_candles(self):
        rows = PREFIX + [(10, 10.5, 9.5, 10), (10, 13.5, 9.5, 13)]
        full = run(rows)
        cut = run(rows, end_index=4)  # target candle (index 4) lies outside the window
        self.assertEqual(full.trades[0].exit_index, 4)
        self.assertEqual((len(cut.trades), cut.unfinished), (0, 1))
        self.assertEqual(len(run(rows, start_index=3).trades), 0)  # signal at index 2 is outside


class TestConfigAndInputs(unittest.TestCase):
    def test_invalid_configuration(self):
        for bad in (
            lambda: CostModel(spread=-1),
            lambda: CostModel(spread=float("nan")),
            lambda: ExitRule(0, 1, 1),
            lambda: ExitRule(1, 0, 1),
            lambda: ExitRule(1, 1, 0),
            lambda: ExitRule(1, 1, 1.5),
            lambda: BacktestConfig(ExitRule(1, 1, 1), ZERO, risk_fraction=0.01),
            lambda: BacktestConfig(ExitRule(1, 1, 1), ZERO, risk_fraction=0),
            lambda: BacktestConfig(ExitRule(1, 1, 1), ZERO, max_open_positions=2),
            lambda: BacktestConfig(ExitRule(1, 1, 1), ZERO, initial_balance=0),
            lambda: BacktestConfig("x", ZERO),
        ):
            with self.assertRaises(ValueError):
                bad()

    def test_invalid_inputs_fail_loudly(self):
        candles = series(PREFIX + [(10, 13.5, 9.5, 13)])
        snaps = build_predictors(candles, CFG)
        with self.assertRaises(ValueError):
            simulate_backtest(candles, snaps, lambda s: "buy", config())
        with self.assertRaises(ValueError):
            simulate_backtest(candles, snaps[:-1], lambda s: None, config())
        with self.assertRaises(ValueError):
            simulate_backtest(candles, snaps, lambda s: None, config(), start_index=3, end_index=2)
        with self.assertRaises(ValueError):
            simulate_backtest(candles, snaps, lambda s: None, "bad")
        naive = [Candle(datetime(2025, 9, 1), 1, 1, 1, 1, 1)]
        with self.assertRaises(ValueError):
            simulate_backtest(naive, [], lambda s: None, config())
        empty = simulate_backtest([], [], lambda s: "long", config())
        self.assertEqual((empty.signals, len(empty.trades)), (0, 0))


def noisy(count=120):
    rows, price = [], 100.0
    for i in range(count):
        step = ((i * 7) % 11 - 5) * 0.4
        rows.append((price, max(price, price + step) + 0.5, min(price, price + step) - 0.5, price + step))
        price += step
    return series(rows)


class TestCausality(unittest.TestCase):
    def test_trades_before_a_window_end_do_not_depend_on_later_candles(self):
        candles = noisy()
        snaps = build_predictors(candles, CFG)
        rule = lambda s: "long" if s.index % 4 == 0 else "short" if s.index % 4 == 2 else None
        cfg = config(hold=4, costs=CostModel(0.3, 0.05))
        full = simulate_backtest(candles, snaps, rule, cfg)
        self.assertGreater(len(full.trades), 10)

        for m in (30, 55, 90):
            prefix = simulate_backtest(candles, snaps, rule, cfg, end_index=m)
            expected = tuple(t for t in full.trades if t.exit_index < m)
            self.assertEqual(prefix.trades[: len(expected)], expected)

    def test_altering_future_candles_leaves_earlier_trades_identical(self):
        candles = noisy()
        altered = list(candles)
        for j in range(80, 120):
            c = candles[j]
            altered[j] = Candle(c.timestamp, c.open + 7, c.high + 40, c.low - 7, c.close + 3, 1)
        rule = lambda s: "long" if s.index % 3 == 0 else None
        cfg = config(hold=4)

        a = simulate_backtest(candles, build_predictors(candles, CFG), rule, cfg)
        b = simulate_backtest(altered, build_predictors(altered, CFG), rule, cfg)

        early = [t for t in a.trades if t.exit_index < 80]
        self.assertTrue(early)
        self.assertEqual([t for t in b.trades if t.exit_index < 80], early)
        self.assertEqual(a, simulate_backtest(candles, build_predictors(candles, CFG), rule, cfg))


if __name__ == "__main__":
    unittest.main()
