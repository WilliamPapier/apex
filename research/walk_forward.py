"""Chronological walk-forward evaluation of candidate rules. Research only.

Folds are defined by candle-open time, never by shuffling:

    train  [t, t + train_span)  |  embargo  |  test [train_end + embargo, + test_span)

and slide forward by ``step``. In every fold each candidate is simulated on the
training window only; the best one by training expectancy (with at least
``min_train_trades`` trades) is then simulated once on the unseen test window.
A fold with no eligible candidate trades nothing. Test windows never feed
selection, so the pooled out-of-sample trades are an honest, if small, sample.

The embargo must be at least the longest holding time of any candidate so a
trade opened late in training cannot overlap the test window. Each window is
simulated in isolation: no candle at or beyond a window's end is used, and
positions still open at the end are dropped as ``unfinished``.

The number of candidates evaluated is reported because every extra candidate
raises the chance that the selected one looks good by luck. Parameter choices
must be made on training data only; the pooled test result must not be used to
re-tune and re-run (that turns the test into training).
"""

from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Sequence

from core.market_data import Candle, validate_candle_sequence
from research.backtest import (
    BacktestConfig,
    BacktestResult,
    ExitRule,
    SignalRule,
    Trade,
    TradeMetrics,
    simulate_backtest,
    summarize_trades,
)
from research.price_expansion import PredictorSnapshot


@dataclass(frozen=True)
class WalkForwardConfig:
    train_span: timedelta
    test_span: timedelta
    step: timedelta
    embargo: timedelta = timedelta(0)
    min_train_trades: int = 30

    def __post_init__(self) -> None:
        for name in ("train_span", "test_span", "step"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise ValueError(f"{name} must be a positive timedelta.")
        if not isinstance(self.embargo, timedelta) or self.embargo < timedelta(0):
            raise ValueError("embargo must be a non-negative timedelta.")
        if type(self.min_train_trades) is not int or self.min_train_trades < 1:
            raise ValueError("min_train_trades must be an integer of at least 1.")


@dataclass(frozen=True)
class RuleCandidate:
    name: str
    signal: SignalRule
    exit: ExitRule


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_from: datetime
    train_until: datetime
    test_from: datetime
    test_until: datetime


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    train_metrics: tuple[tuple[str, TradeMetrics], ...]
    selected: str | None
    test: BacktestResult | None


@dataclass(frozen=True)
class WalkForwardResult:
    folds: tuple[FoldResult, ...]
    oos_trades: tuple[Trade, ...]
    oos_metrics: TradeMetrics
    candidates_evaluated: int
    folds_with_selection: int


def make_folds(
    candles: Sequence[Candle], config: WalkForwardConfig, candle_duration: timedelta
) -> tuple[Fold, ...]:
    """Time-based folds whose test window lies fully inside the observed data range."""

    validate_candle_sequence(candles)
    if not candles:
        return ()
    opens = [c.timestamp.astimezone(timezone.utc) for c in candles]
    data_end = opens[-1] + candle_duration
    folds: list[Fold] = []
    start = opens[0]
    while True:
        train_end = start + config.train_span
        test_start = train_end + config.embargo
        test_end = test_start + config.test_span
        if test_end > data_end:
            break
        folds.append(
            Fold(
                number=len(folds),
                train_start=bisect_left(opens, start),
                train_end=bisect_left(opens, train_end),
                test_start=bisect_left(opens, test_start),
                test_end=bisect_left(opens, test_end),
                train_from=start, train_until=train_end,
                test_from=test_start, test_until=test_end,
            )
        )
        start += config.step
    return tuple(folds)


def walk_forward_evaluate(
    candles: Sequence[Candle],
    snapshots: Sequence[PredictorSnapshot],
    candidates: Sequence[RuleCandidate],
    config: WalkForwardConfig,
    template: BacktestConfig,
) -> WalkForwardResult:
    if not candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("candidates must be non-empty with unique names.")
    if not isinstance(config, WalkForwardConfig) or not isinstance(template, BacktestConfig):
        raise ValueError("config and template must be WalkForwardConfig and BacktestConfig.")
    longest_hold = max(c.exit.max_hold_candles for c in candidates) * template.candle_duration
    if config.embargo < longest_hold:
        raise ValueError("embargo must be at least the longest candidate holding time.")

    results: list[FoldResult] = []
    pooled: list[Trade] = []
    for fold in make_folds(candles, config, template.candle_duration):
        trained = []
        for candidate in candidates:
            run_config = replace(template, exit=candidate.exit)
            train = simulate_backtest(
                candles, snapshots, candidate.signal, run_config,
                start_index=fold.train_start, end_index=fold.train_end,
            )
            trained.append((candidate, train.metrics))
        eligible = [
            (c, m) for c, m in trained
            if m.trades >= config.min_train_trades and m.expectancy_r is not None
        ]
        selected = max(eligible, key=lambda pair: pair[1].expectancy_r, default=None)  # first wins ties
        test = None
        if selected is not None:
            test = simulate_backtest(
                candles, snapshots, selected[0].signal, replace(template, exit=selected[0].exit),
                start_index=fold.test_start, end_index=fold.test_end,
            )
            pooled.extend(test.trades)
        results.append(
            FoldResult(
                fold=fold,
                train_metrics=tuple((c.name, m) for c, m in trained),
                selected=selected[0].name if selected else None,
                test=test,
            )
        )
    return WalkForwardResult(
        folds=tuple(results),
        oos_trades=tuple(pooled),
        oos_metrics=summarize_trades(pooled, template.initial_balance, template.risk_fraction),
        candidates_evaluated=len(candidates),
        folds_with_selection=sum(r.selected is not None for r in results),
    )
