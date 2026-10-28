"""Offline, cost-aware trade simulator for candidate rules. Research only; never trades.

Signals come from a pure function of one causal ``PredictorSnapshot`` (candle
``i`` has closed), so a rule cannot see later candles by construction.

Execution model (deliberately simple, conservative and fully deterministic)
    * Entry: at the OPEN of candle i+1, only if that candle opens exactly one
      candle-duration after candle i opened (no fill across an unobserved gap)
      and the snapshot ATR is known. The ATR is never taken from later candles.
    * Levels are on the raw price series: stop = entry -/+ stop_atr_multiple x
      ATR; target = entry +/- target_r x (stop_atr_multiple x ATR).
    * Candle i+1 itself is checked for stop/target. If one candle touches both,
      the stop is assumed first (intrabar order is unknowable).
    * A stop is filled at the stop level, or at the candle open when that open
      is already beyond it (gap through). Targets fill exactly at the target.
    * Time exit: at the close of the max_hold_candles-th candle held.
    * A position left open when the data window ends is reported as
      ``unfinished`` and excluded from metrics, never marked to market.
    * Gaps inside a trade are not filled: the position is held through them and
      the trade is flagged ``spans_gap``.
    * One position at a time (settings.max_open_positions == 1): signals while
      a position is open are ignored and counted.

Costs (price units; the CSVs carry no spread, so the spread is configuration)
    cost_price = spread + 2 x slippage_per_side + 2 x commission_per_side,
    charged on every trade (slippage is charged even on targets, conservatively).
    One R is the cost-inclusive stop risk, ``stop_distance + cost_price``, so a
    stop fill at the stop level is exactly -1R. Position size comes from the
    existing ``calculate_position_risk`` at risk_fraction (hard-capped at 0.5%).
    Lot rounding, broker minimum volume, swaps and margin are not modelled.

Balance compounds on closed trades only; drawdown is measured on closed-trade
balances, so intratrade drawdown is not captured.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Callable, Literal, Sequence

from config.settings import SETTINGS
from core.market_data import Candle, validate_candle_sequence
from data.mt5_history_collector import M5_INTERVAL
from research.price_expansion import PredictorSnapshot
from risk.risk_manager import TradeRisk, calculate_position_risk


Side = Literal["long", "short"]
ExitReason = Literal["stop", "target", "time"]
SignalRule = Callable[[PredictorSnapshot], "Side | None"]


def _non_negative(name: str, value: float) -> None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    if not ok or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")


@dataclass(frozen=True)
class CostModel:
    spread: float
    slippage_per_side: float = 0.0
    commission_per_side: float = 0.0

    def __post_init__(self) -> None:
        _non_negative("spread", self.spread)
        _non_negative("slippage_per_side", self.slippage_per_side)
        _non_negative("commission_per_side", self.commission_per_side)

    @property
    def round_trip(self) -> float:
        return self.spread + 2 * self.slippage_per_side + 2 * self.commission_per_side


@dataclass(frozen=True)
class ExitRule:
    stop_atr_multiple: float
    target_r: float
    max_hold_candles: int

    def __post_init__(self) -> None:
        for name in ("stop_atr_multiple", "target_r"):
            value = getattr(self, name)
            _non_negative(name, value)
            if value == 0:
                raise ValueError(f"{name} must be positive.")
        if type(self.max_hold_candles) is not int or self.max_hold_candles < 1:
            raise ValueError("max_hold_candles must be an integer of at least 1.")


@dataclass(frozen=True)
class BacktestConfig:
    exit: ExitRule
    costs: CostModel
    initial_balance: float = 10_000.0
    risk_fraction: float = SETTINGS.risk_per_trade
    max_open_positions: int = SETTINGS.max_open_positions
    value_per_price_unit_per_lot: float = 1.0  # lot sizes only; R results do not depend on it
    candle_duration: timedelta = M5_INTERVAL

    def __post_init__(self) -> None:
        if not isinstance(self.exit, ExitRule) or not isinstance(self.costs, CostModel):
            raise ValueError("exit and costs must be ExitRule and CostModel instances.")
        for name in ("initial_balance", "value_per_price_unit_per_lot"):
            _non_negative(name, getattr(self, name))
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive.")
        _non_negative("risk_fraction", self.risk_fraction)
        if not 0 < self.risk_fraction <= SETTINGS.risk_per_trade:
            raise ValueError("risk_fraction must be positive and not above the configured risk.")
        if self.max_open_positions != 1:
            raise ValueError("Only one open position is supported.")
        if not isinstance(self.candle_duration, timedelta) or self.candle_duration <= timedelta(0):
            raise ValueError("candle_duration must be a positive timedelta.")


@dataclass(frozen=True)
class Trade:
    side: Side
    signal_index: int
    signal_available_at: datetime
    entry_index: int
    entry_time: datetime
    entry_price: float
    stop_price: float
    target_price: float
    stop_distance: float
    risk_distance: float
    lots: float
    risk_amount: float
    exit_index: int
    exit_known_at: datetime
    exit_reason: ExitReason
    exit_price: float
    gross_r: float
    cost_r: float
    net_r: float
    candles_held: int
    spans_gap: bool
    balance_after: float


@dataclass(frozen=True)
class TradeMetrics:
    trades: int
    wins: int
    losses: int
    win_rate: float | None
    net_r: float
    expectancy_r: float | None
    average_cost_r: float | None
    profit_factor: float | None
    max_drawdown_fraction: float
    max_consecutive_losses: int
    final_balance: float
    trades_spanning_gap: int


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    metrics: TradeMetrics
    signals: int
    ignored_position_open: int
    skipped_gap_before_entry: int
    skipped_no_atr: int
    rejected_by_risk: int
    unfinished: int


def summarize_trades(
    trades: Sequence[Trade], initial_balance: float, risk_fraction: float
) -> TradeMetrics:
    """Metrics from chronological trades; the balance compounds ``risk_fraction`` x R per trade."""

    balance = peak = initial_balance
    drawdown = 0.0
    streak = longest = 0
    for trade in trades:
        balance += trade.net_r * balance * risk_fraction
        peak = max(peak, balance)
        drawdown = max(drawdown, (peak - balance) / peak)
        streak = streak + 1 if trade.net_r < 0 else 0
        longest = max(longest, streak)
    wins = [t.net_r for t in trades if t.net_r > 0]
    losses = [t.net_r for t in trades if t.net_r < 0]
    count = len(trades)
    return TradeMetrics(
        trades=count,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / count if count else None,
        net_r=sum(t.net_r for t in trades),
        expectancy_r=sum(t.net_r for t in trades) / count if count else None,
        average_cost_r=sum(t.cost_r for t in trades) / count if count else None,
        profit_factor=sum(wins) / -sum(losses) if losses else None,
        max_drawdown_fraction=drawdown,
        max_consecutive_losses=longest,
        final_balance=balance,
        trades_spanning_gap=sum(t.spans_gap for t in trades),
    )


@dataclass
class _Counters:
    signals: int = 0
    ignored: int = 0
    gap: int = 0
    no_atr: int = 0
    risk: int = 0
    unfinished: int = 0
    trades: list[Trade] = field(default_factory=list)


def simulate_backtest(
    candles: Sequence[Candle],
    snapshots: Sequence[PredictorSnapshot],
    signal: SignalRule,
    config: BacktestConfig,
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> BacktestResult:
    """Simulate within candle indices ``[start_index, end_index)``.

    Signals are read from snapshots in that range and no candle at or beyond
    ``end_index`` is ever used, so a train or test window is self-contained.
    """

    if not isinstance(config, BacktestConfig):
        raise ValueError("config must be a BacktestConfig.")
    end = len(candles) if end_index is None else end_index
    if type(start_index) is not int or type(end) is not int or not 0 <= start_index <= end <= len(candles):
        raise ValueError("Invalid start_index/end_index.")
    window = candles[start_index:end]
    validate_candle_sequence(window)
    if any(c.timestamp.utcoffset() is None for c in window):
        raise ValueError("Candle timestamps must be timezone-aware.")
    if len(snapshots) != len(candles) or any(s.index != i for i, s in enumerate(snapshots)):
        raise ValueError("snapshots must align one-to-one with candles.")

    duration = config.candle_duration
    opens: list[datetime | None] = [None] * len(candles)
    for k in range(start_index, end):
        opens[k] = candles[k].timestamp.astimezone(timezone.utc)
    cost = config.costs.round_trip
    state = _Counters()
    balance = config.initial_balance
    last_exit = -1

    for i in range(start_index, end - 1):
        snapshot = snapshots[i]
        side = signal(snapshot)
        if side is None:
            continue
        if side not in ("long", "short"):
            raise ValueError("A signal rule must return 'long', 'short' or None.")
        state.signals += 1
        if i < last_exit:
            state.ignored += 1
            continue
        entry = i + 1
        if snapshot.atr is None:
            state.no_atr += 1
            continue
        if opens[entry] != snapshot.available_at:
            state.gap += 1
            continue
        direction = 1 if side == "long" else -1
        raw_entry = candles[entry].open
        stop_distance = config.exit.stop_atr_multiple * snapshot.atr
        risk_distance = stop_distance + cost
        stop = raw_entry - direction * stop_distance
        target = raw_entry + direction * config.exit.target_r * stop_distance
        sizing = calculate_position_risk(
            TradeRisk(
                balance, config.risk_fraction, raw_entry,
                raw_entry - direction * risk_distance, config.value_per_price_unit_per_lot,
            )
        )
        if not sizing.approved:
            state.risk += 1
            continue

        last = min(entry + config.exit.max_hold_candles - 1, end - 1)
        outcome = None
        for j in range(entry, last + 1):
            c = candles[j]
            adverse = c.low if side == "long" else c.high
            favourable = c.high if side == "long" else c.low
            if direction * (adverse - stop) <= 0:
                gap_open = direction * (c.open - stop) <= 0
                outcome = (j, "stop", c.open if gap_open else stop)
            elif direction * (favourable - target) >= 0:
                outcome = (j, "target", target)
            elif j == entry + config.exit.max_hold_candles - 1:
                outcome = (j, "time", c.close)
            if outcome:
                break
        if outcome is None:
            state.unfinished += 1
            break
        exit_index, reason, exit_price = outcome
        gross = direction * (exit_price - raw_entry)
        net_r = (gross - cost) / risk_distance
        balance += net_r * sizing.risk_amount
        last_exit = exit_index
        state.trades.append(
            Trade(
                side=side, signal_index=i, signal_available_at=snapshot.available_at,
                entry_index=entry, entry_time=opens[entry], entry_price=raw_entry,
                stop_price=stop, target_price=target, stop_distance=stop_distance,
                risk_distance=risk_distance, lots=sizing.position_size_lots,
                risk_amount=sizing.risk_amount, exit_index=exit_index,
                exit_known_at=opens[exit_index] + duration, exit_reason=reason,  # type: ignore[arg-type]
                exit_price=exit_price, gross_r=gross / risk_distance, cost_r=cost / risk_distance,
                net_r=net_r, candles_held=exit_index - entry + 1,
                spans_gap=any(
                    opens[k + 1] - opens[k] != duration for k in range(entry, exit_index)
                ),
                balance_after=balance,
            )
        )

    trades = tuple(state.trades)
    return BacktestResult(
        trades=trades,
        metrics=summarize_trades(trades, config.initial_balance, config.risk_fraction),
        signals=state.signals,
        ignored_position_open=state.ignored,
        skipped_gap_before_entry=state.gap,
        skipped_no_atr=state.no_atr,
        rejected_by_risk=state.risk,
        unfinished=state.unfinished,
    )
