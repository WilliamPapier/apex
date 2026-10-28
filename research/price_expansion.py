"""Offline price-expansion research: causal predictors kept apart from outcome labels.

Research only. Nothing here is a trading signal, entry rule or model, and
nothing touches a broker. Candle timestamps are candle-open times; a candle is
knowable at ``open + candle_duration`` (taken from ``StructureConfig``).

Prediction point
    Candle ``i`` has just closed. ``available_at`` = its close time (UTC).
    Everything in ``PredictorSnapshot`` uses candles ``0..i`` only, and only
    swings confirmed, breaks and sweeps available at or before that instant.

Outcome labels (``ExpansionLabel``; use only for evaluation, never as input)
    Reference price = close of candle ``i``. The forward window for horizon H
    holds the candles whose open lies in ``[available_at, available_at + H)``
    (H must be a multiple of the candle duration, k = H / duration candles).
    * upward excursion   = max(0, max window high - reference)
    * downward excursion = max(0, reference - min window low)
    * threshold = max(atr_multiple * ATR_i, min_absolute_move,
      min_fractional_change * reference), with ATR_i known at the prediction
      point. A non-positive threshold is reported as ``atr_unavailable``.
    * expansion = some window candle's high reaches ``reference + threshold``
      or its low reaches ``reference - threshold``. Direction is the side hit
      on the first such candle; if both sides are hit on that same candle the
      order inside the candle is unknowable and direction is ``both``.
    * ``time_to_threshold`` = close of that first candle minus ``available_at``.
    * mfe / mae = largest favourable / adverse excursion over the whole window
      for the expansion direction (mae may occur after the threshold is hit).
    * outcome: retained = signed (last window close - reference) / mfe.
      ``reversal`` if retained < 0, ``continuation`` if retained >=
      ``continuation_retention``, otherwise ``retracement``.
    * momentum_profile (k >= 2): ``front_loaded`` if the favourable excursion
      over the first k//2 candles exceeds half of the final mfe,
      ``back_loaded`` if below half, ``even`` if equal. Descriptive only.
    Window status: ``complete`` (k contiguous candles), ``gap`` (the data
    continues past the window but candles are missing or irregular),
    ``incomplete`` (data ends inside the window) or ``atr_unavailable``. Gaps
    are never filled; ``gap`` windows are only labelled when
    ``allow_gap_windows`` is true. Unlabelled rows keep ``expanded=None``.

Overlap: every eligible timestamp is labelled, so thresholds and horizons can
be compared on identical rows. ``select_non_overlapping`` greedily thins one
(horizon, multiple) series so selected windows do not overlap.

Early-warning baseline: simple fixed-threshold boolean flags built from
predictors. ``evaluate_flag`` reports precision, recall, miss rate, false
alarms per observed hour and lead time on two chronological parts. The flag
thresholds are fixed, never fitted, and this is a descriptive baseline, not a
validated predictor or evidence of profitability.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Literal, Sequence

from core.market_data import Candle
from data.historical_data_loader import load_candles_csv
from research.liquidity_sweeps import LiquiditySweep, SweepConfig, analyze_liquidity_sweeps
from research.market_features import calculate_market_features
from research.market_structure import (
    MarketStructure,
    StructureConfig,
    _closes_at,
    _prepare,
    analyze_market_structure,
)


Direction = Literal["up", "down", "both"]
WindowStatus = Literal["complete", "gap", "incomplete", "atr_unavailable"]
Outcome = Literal["continuation", "retracement", "reversal"]
MomentumProfile = Literal["front_loaded", "back_loaded", "even"]
PinBar = Literal["bullish_pin", "bearish_pin"]


def _positive_finite(name: str, value: float, allow_zero: bool = False) -> None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    if not ok or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'} number.")


def _positive_int(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


@dataclass(frozen=True)
class ExpansionConfig:
    structure: StructureConfig = field(default_factory=StructureConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    horizons: tuple[timedelta, ...] = (
        timedelta(minutes=5),
        timedelta(minutes=15),
        timedelta(minutes=30),
    )
    atr_multiples: tuple[float, ...] = (1.0, 2.0, 3.0)
    min_absolute_move: float = 0.0
    min_fractional_change: float = 0.0
    allow_gap_windows: bool = False
    continuation_retention: float = 0.5
    atr_period: int = 14
    volatility_periods: tuple[int, ...] = (5, 20)
    compression_short: int = 5
    compression_long: int = 20
    volume_lookback: int = 20
    momentum_candles: int = 3
    sequence_candles: int = 3
    event_lookback: int = 12
    pin_wick_ratio: float = 0.6
    pin_body_ratio: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.structure, StructureConfig):
            raise ValueError("structure must be a StructureConfig.")
        if not isinstance(self.sweep, SweepConfig):
            raise ValueError("sweep must be a SweepConfig.")
        duration = self.structure.candle_duration
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be a non-empty tuple of unique timedeltas.")
        for horizon in self.horizons:
            if not isinstance(horizon, timedelta) or horizon <= timedelta(0) or horizon % duration:
                raise ValueError("Each horizon must be a positive multiple of candle_duration.")
        if not self.atr_multiples or len(set(self.atr_multiples)) != len(self.atr_multiples):
            raise ValueError("atr_multiples must be a non-empty tuple of unique values.")
        for multiple in self.atr_multiples:
            _positive_finite("atr_multiples", multiple)
        _positive_finite("min_absolute_move", self.min_absolute_move, allow_zero=True)
        _positive_finite("min_fractional_change", self.min_fractional_change, allow_zero=True)
        _positive_finite("continuation_retention", self.continuation_retention)
        if not self.continuation_retention <= 1:
            raise ValueError("continuation_retention must not exceed 1.")
        if type(self.allow_gap_windows) is not bool:
            raise ValueError("allow_gap_windows must be a bool.")
        _positive_int("atr_period", self.atr_period)
        if not self.volatility_periods:
            raise ValueError("volatility_periods must not be empty.")
        for period in self.volatility_periods:
            _positive_int("volatility_periods", period)
        for name in (
            "compression_short",
            "compression_long",
            "volume_lookback",
            "momentum_candles",
            "sequence_candles",
            "event_lookback",
        ):
            _positive_int(name, getattr(self, name))
        if self.compression_short >= self.compression_long:
            raise ValueError("compression_short must be smaller than compression_long.")
        if not isinstance(self.pin_wick_ratio, (int, float)) or not 0.5 < self.pin_wick_ratio <= 1:
            raise ValueError("pin_wick_ratio must be in (0.5, 1].")
        if not isinstance(self.pin_body_ratio, (int, float)) or not 0 <= self.pin_body_ratio < 0.5:
            raise ValueError("pin_body_ratio must be in [0, 0.5).")

    @property
    def candle_duration(self) -> timedelta:
        return self.structure.candle_duration


@dataclass(frozen=True)
class PredictorSnapshot:
    """Information known when candle ``index`` closes (``available_at``). None = not yet computable."""

    index: int
    timestamp: datetime
    available_at: datetime
    atr: float | None
    range_to_atr: float | None
    body_to_atr: float | None
    upper_wick_to_atr: float | None
    lower_wick_to_atr: float | None
    body_to_range_ratio: float | None
    direction: str
    recent_directions: tuple[str, ...] | None
    realized_volatility: tuple[tuple[int, float | None], ...]
    volatility_ratio: float | None
    range_compression_ratio: float | None
    tick_volume_ratio: float | None
    momentum_to_atr: float | None
    pin_bar: PinBar | None
    latest_swing_high: float | None
    latest_swing_high_confirmed_at: datetime | None
    distance_to_swing_high_atr: float | None
    latest_swing_low: float | None
    latest_swing_low_confirmed_at: datetime | None
    distance_to_swing_low_atr: float | None
    recent_up_breaks: int
    recent_down_breaks: int
    recent_high_sweeps: int
    recent_low_sweeps: int


@dataclass(frozen=True)
class ExpansionLabel:
    """Outcome of the window after the prediction point. Never a predictor."""

    index: int
    timestamp: datetime
    available_at: datetime
    horizon: timedelta
    atr_multiple: float
    status: WindowStatus
    reference_close: float
    atr: float | None
    threshold: float | None
    window_end: datetime
    window_candles: int
    expanded: bool | None
    direction: Direction | None
    threshold_reached_at: datetime | None
    time_to_threshold: timedelta | None
    max_up: float | None
    max_down: float | None
    mfe: float | None
    mae: float | None
    end_close_change: float | None
    retained_fraction: float | None
    outcome: Outcome | None
    momentum_profile: MomentumProfile | None


@dataclass(frozen=True)
class ExpansionResearch:
    snapshots: tuple[PredictorSnapshot, ...]
    labels: tuple[ExpansionLabel, ...]
    structure: MarketStructure
    sweeps: tuple[LiquiditySweep, ...]


def _pin_bar(candle: Candle, config: ExpansionConfig) -> PinBar | None:
    candle_range = candle.high - candle.low
    if candle_range <= 0:
        return None
    body = abs(candle.close - candle.open)
    if body > config.pin_body_ratio * candle_range:
        return None
    if min(candle.open, candle.close) - candle.low >= config.pin_wick_ratio * candle_range:
        return "bullish_pin"
    if candle.high - max(candle.open, candle.close) >= config.pin_wick_ratio * candle_range:
        return "bearish_pin"
    return None


def _latest_before(swings: Sequence, confirmed: Sequence[datetime], limit: datetime):
    position = bisect_right(confirmed, limit)
    return swings[position - 1] if position else None


def _ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def build_predictors(
    candles: Sequence[Candle],
    config: ExpansionConfig | None = None,
    structure: MarketStructure | None = None,
    sweeps: Sequence[LiquiditySweep] | None = None,
) -> tuple[PredictorSnapshot, ...]:
    """One snapshot per candle using only information available at that candle's close.

    ``structure`` and ``sweeps`` may be supplied to avoid recomputation; they
    must come from the same candles. Both are prefix-causal, and each snapshot
    additionally filters them by availability time.
    """

    config = config or ExpansionConfig()
    if not isinstance(config, ExpansionConfig):
        raise ValueError("config must be an ExpansionConfig.")
    _prepare(candles, config.structure)
    if not candles:
        return ()
    if structure is None:
        structure = analyze_market_structure(candles, config.structure)
    if sweeps is None:
        sweeps = analyze_liquidity_sweeps(candles, config.structure, config.sweep)
    features = calculate_market_features(
        candles, volatility_periods=config.volatility_periods, atr_period=config.atr_period
    )
    ordered_sweeps = sorted(sweeps, key=lambda s: (s.sweep_index, s.side, s.swing.candle_index))
    up_breaks = sorted(b.break_index for b in structure.breaks if b.direction == "up")
    down_breaks = sorted(b.break_index for b in structure.breaks if b.direction == "down")
    high_sweeps = [s.sweep_index for s in ordered_sweeps if s.side == "high"]
    low_sweeps = [s.sweep_index for s in ordered_sweeps if s.side == "low"]

    high_confirmed = [s.confirmed_at for s in structure.swing_highs]
    low_confirmed = [s.confirmed_at for s in structure.swing_lows]

    def recent(indices: list[int], i: int) -> int:
        return bisect_right(indices, i) - bisect_right(indices, i - config.event_lookback)

    snapshots: list[PredictorSnapshot] = []
    for i, (candle, feature) in enumerate(zip(candles, features)):
        available_at = _closes_at(candle, config.structure)
        atr = feature.average_true_range
        if atr is not None and atr <= 0:
            atr = None
        ranges = [f.absolute_range for f in features[max(0, i + 1 - config.compression_long) : i + 1]]
        compression = None
        if i + 1 >= config.compression_long:
            long_mean = fmean(ranges[-config.compression_long :])
            compression = _ratio(fmean(ranges[-config.compression_short :]), long_mean)
        volume_ratio = None
        if i >= config.volume_lookback:
            prior = [c.tick_volume for c in candles[i - config.volume_lookback : i]]
            volume_ratio = _ratio(candle.tick_volume, fmean(prior))
        momentum = None
        if i >= config.momentum_candles and atr is not None:
            momentum = (candle.close - candles[i - config.momentum_candles].close) / atr
        sequence = None
        if i + 1 >= config.sequence_candles:
            sequence = tuple(f.direction for f in features[i + 1 - config.sequence_candles : i + 1])
        vols = dict(feature.rolling_realized_volatility)
        first, last = config.volatility_periods[0], config.volatility_periods[-1]
        vol_ratio = None
        if first != last and vols.get(first) is not None and vols.get(last) is not None:
            vol_ratio = _ratio(vols[first], vols[last])
        high = _latest_before(structure.swing_highs, high_confirmed, available_at)
        low = _latest_before(structure.swing_lows, low_confirmed, available_at)
        snapshots.append(
            PredictorSnapshot(
                index=i,
                timestamp=candle.timestamp,
                available_at=available_at,
                atr=atr,
                range_to_atr=_ratio(feature.absolute_range, atr),
                body_to_atr=_ratio(feature.absolute_body_size, atr),
                upper_wick_to_atr=_ratio(feature.upper_wick_size, atr),
                lower_wick_to_atr=_ratio(feature.lower_wick_size, atr),
                body_to_range_ratio=feature.body_to_range_ratio,
                direction=feature.direction,
                recent_directions=sequence,
                realized_volatility=feature.rolling_realized_volatility,
                volatility_ratio=vol_ratio,
                range_compression_ratio=compression,
                tick_volume_ratio=volume_ratio,
                momentum_to_atr=momentum,
                pin_bar=_pin_bar(candle, config),
                latest_swing_high=high.price if high else None,
                latest_swing_high_confirmed_at=high.confirmed_at if high else None,
                distance_to_swing_high_atr=(
                    _signed(high.price - candle.close, atr) if high else None
                ),
                latest_swing_low=low.price if low else None,
                latest_swing_low_confirmed_at=low.confirmed_at if low else None,
                distance_to_swing_low_atr=(
                    _signed(candle.close - low.price, atr) if low else None
                ),
                recent_up_breaks=recent(up_breaks, i),
                recent_down_breaks=recent(down_breaks, i),
                recent_high_sweeps=recent(high_sweeps, i),
                recent_low_sweeps=recent(low_sweeps, i),
            )
        )
    return tuple(snapshots)


def _signed(value: float, atr: float | None) -> float | None:
    return None if atr is None else value / atr


def _unlabelled(base: dict, status: WindowStatus, count: int, threshold: float | None, atr) -> ExpansionLabel:
    return ExpansionLabel(
        **base, status=status, atr=atr, threshold=threshold, window_candles=count,
        expanded=None, direction=None, threshold_reached_at=None, time_to_threshold=None,
        max_up=None, max_down=None, mfe=None, mae=None, end_close_change=None,
        retained_fraction=None, outcome=None, momentum_profile=None,
    )


def label_expansions(
    candles: Sequence[Candle], config: ExpansionConfig | None = None
) -> tuple[ExpansionLabel, ...]:
    """Outcome labels for every candle, horizon and ATR multiple (candle-major order)."""

    config = config or ExpansionConfig()
    if not isinstance(config, ExpansionConfig):
        raise ValueError("config must be an ExpansionConfig.")
    _prepare(candles, config.structure)
    features = calculate_market_features(
        candles, volatility_periods=config.volatility_periods, atr_period=config.atr_period
    )
    duration = config.candle_duration
    opens = [c.timestamp.astimezone(timezone.utc) for c in candles]
    labels: list[ExpansionLabel] = []
    for i, candle in enumerate(candles):
        available_at = opens[i] + duration
        atr = features[i].average_true_range
        for horizon in config.horizons:
            end = available_at + horizon
            k = horizon // duration
            stop = i + 1
            while stop < len(candles) and opens[stop] < end:
                stop += 1
            window = candles[i + 1 : stop]
            contiguous = len(window) == k and all(
                opens[i + 1 + n] == available_at + n * duration for n in range(k)
            )
            if contiguous:
                status: WindowStatus = "complete"
            elif stop < len(candles):
                status = "gap"
            else:
                status = "incomplete"
            for multiple in config.atr_multiples:
                base = dict(
                    index=i, timestamp=candle.timestamp, available_at=available_at,
                    horizon=horizon, atr_multiple=multiple,
                    reference_close=candle.close, window_end=end,
                )
                threshold = None
                if atr is not None:
                    threshold = max(
                        multiple * atr,
                        config.min_absolute_move,
                        config.min_fractional_change * abs(candle.close),
                    )
                if threshold is None or threshold <= 0:
                    labels.append(_unlabelled(base, "atr_unavailable", len(window), None, atr))
                elif status == "incomplete" or (status == "gap" and not config.allow_gap_windows):
                    labels.append(_unlabelled(base, status, len(window), threshold, atr))
                else:
                    labels.append(
                        _label_window(base, status, window, opens[i + 1 : stop], k, threshold, atr, config)
                    )
    return tuple(labels)


def _label_window(
    base: dict, status: WindowStatus, window: Sequence[Candle], window_opens: Sequence[datetime],
    k: int, threshold: float, atr: float | None, config: ExpansionConfig,
) -> ExpansionLabel:
    reference = base["reference_close"]
    available_at = base["available_at"]
    ups = [max(0.0, c.high - reference) for c in window]
    downs = [max(0.0, reference - c.low) for c in window]
    max_up, max_down = (max(ups), max(downs)) if window else (0.0, 0.0)
    hit = next((n for n in range(len(window)) if ups[n] >= threshold or downs[n] >= threshold), None)
    if hit is None:
        return ExpansionLabel(
            **base, status=status, atr=atr, threshold=threshold, window_candles=len(window),
            expanded=False, direction=None, threshold_reached_at=None, time_to_threshold=None,
            max_up=max_up, max_down=max_down, mfe=None, mae=None,
            end_close_change=window[-1].close - reference if window else None,
            retained_fraction=None, outcome=None, momentum_profile=None,
        )
    up_hit, down_hit = ups[hit] >= threshold, downs[hit] >= threshold
    direction: Direction = "both" if up_hit and down_hit else "up" if up_hit else "down"
    reached_at = window_opens[hit] + config.candle_duration
    end_change = window[-1].close - reference
    mfe = mae = retained = None
    outcome: Outcome | None = None
    profile: MomentumProfile | None = None
    if direction != "both":
        favourable, adverse = (ups, downs) if direction == "up" else (downs, ups)
        mfe, mae = max(favourable), max(adverse)
        retained = (end_change if direction == "up" else -end_change) / mfe
        outcome = (
            "reversal" if retained < 0
            else "continuation" if retained >= config.continuation_retention
            else "retracement"
        )
        if k >= 2:
            half = max(favourable[: k // 2])
            profile = (
                "front_loaded" if half > mfe / 2 else "back_loaded" if half < mfe / 2 else "even"
            )
    return ExpansionLabel(
        **base, status=status, atr=atr, threshold=threshold, window_candles=len(window),
        expanded=True, direction=direction, threshold_reached_at=reached_at,
        time_to_threshold=reached_at - available_at, max_up=max_up, max_down=max_down,
        mfe=mfe, mae=mae, end_close_change=end_change, retained_fraction=retained,
        outcome=outcome, momentum_profile=profile,
    )


def select_non_overlapping(
    labels: Sequence[ExpansionLabel], horizon: timedelta, atr_multiple: float
) -> tuple[ExpansionLabel, ...]:
    """Greedy chronological thinning of expanded labels of one (horizon, multiple) series.

    A label is kept only if its prediction point is at or after the previous
    kept label's window end. Earlier events always win, so the selection of an
    earlier part of the data never depends on later candles.
    """

    selected: list[ExpansionLabel] = []
    series = sorted(
        (l for l in labels if l.horizon == horizon and l.atr_multiple == atr_multiple and l.expanded),
        key=lambda l: l.index,
    )
    for label in series:
        if not selected or label.available_at >= selected[-1].window_end:
            selected.append(label)
    return tuple(selected)


def analyze_expansion(
    candles: Sequence[Candle], config: ExpansionConfig | None = None
) -> ExpansionResearch:
    config = config or ExpansionConfig()
    if not isinstance(config, ExpansionConfig):
        raise ValueError("config must be an ExpansionConfig.")
    _prepare(candles, config.structure)
    structure = analyze_market_structure(candles, config.structure)
    sweeps = analyze_liquidity_sweeps(candles, config.structure, config.sweep)
    return ExpansionResearch(
        snapshots=build_predictors(candles, config, structure, sweeps),
        labels=label_expansions(candles, config),
        structure=structure,
        sweeps=tuple(sweeps),
    )


@dataclass(frozen=True)
class ExpansionSummary:
    horizon: timedelta
    atr_multiple: float
    total: int
    atr_unavailable: int
    incomplete: int
    gap: int
    labelled: int
    expanded: int
    up: int
    down: int
    both: int
    continuation: int
    retracement: int
    reversal: int
    base_rate: float | None
    median_time_to_threshold: timedelta | None
    non_overlapping_events: int


def summarize_labels(labels: Sequence[ExpansionLabel], config: ExpansionConfig) -> tuple[ExpansionSummary, ...]:
    summaries = []
    for horizon in config.horizons:
        for multiple in config.atr_multiples:
            group = [l for l in labels if l.horizon == horizon and l.atr_multiple == multiple]
            done = [l for l in group if l.expanded is not None]
            hits = [l for l in done if l.expanded]
            status = Counter(l.status for l in group)
            outcomes = Counter(l.outcome for l in hits)
            directions = Counter(l.direction for l in hits)
            times = [l.time_to_threshold for l in hits if l.time_to_threshold is not None]
            summaries.append(
                ExpansionSummary(
                    horizon=horizon, atr_multiple=multiple, total=len(group),
                    atr_unavailable=status["atr_unavailable"], incomplete=status["incomplete"],
                    gap=status["gap"] if not config.allow_gap_windows else 0,
                    labelled=len(done), expanded=len(hits),
                    up=directions["up"], down=directions["down"], both=directions["both"],
                    continuation=outcomes["continuation"], retracement=outcomes["retracement"],
                    reversal=outcomes["reversal"],
                    base_rate=len(hits) / len(done) if done else None,
                    median_time_to_threshold=median(times) if times else None,
                    non_overlapping_events=len(select_non_overlapping(labels, horizon, multiple)),
                )
            )
    return tuple(summaries)


@dataclass(frozen=True)
class FlagConfig:
    """Fixed, untuned thresholds for the transparent baseline flags."""

    compression_ratio_max: float = 0.7
    volume_spike_min: float = 2.0
    large_candle_atr_min: float = 1.5

    def __post_init__(self) -> None:
        for name in ("compression_ratio_max", "volume_spike_min", "large_candle_atr_min"):
            _positive_finite(name, getattr(self, name))


def baseline_flags(flags: FlagConfig) -> dict[str, Callable[[PredictorSnapshot], bool]]:
    return {
        "compressed_range": lambda s: s.range_compression_ratio is not None
        and s.range_compression_ratio <= flags.compression_ratio_max,
        "tick_volume_spike": lambda s: s.tick_volume_ratio is not None
        and s.tick_volume_ratio >= flags.volume_spike_min,
        "large_closed_candle": lambda s: s.range_to_atr is not None
        and s.range_to_atr >= flags.large_candle_atr_min,
        "pin_bar": lambda s: s.pin_bar is not None,
        "recent_sweep": lambda s: s.recent_high_sweeps + s.recent_low_sweeps > 0,
        "recent_break": lambda s: s.recent_up_breaks + s.recent_down_breaks > 0,
    }


@dataclass(frozen=True)
class FlagEvaluation:
    flag: str
    part: str
    eligible: int
    flagged: int
    events: int
    true_positives: int
    precision: float | None
    recall: float | None
    miss_rate: float | None
    base_rate: float | None
    lift: float | None
    false_alarms: int
    false_alarms_per_hour: float | None
    median_lead_time: timedelta | None


def evaluate_flag(
    snapshots: Sequence[PredictorSnapshot],
    labels: Sequence[ExpansionLabel],
    name: str,
    predicate: Callable[[PredictorSnapshot], bool],
    horizon: timedelta,
    atr_multiple: float,
    candle_duration: timedelta,
    split_fraction: float = 0.5,
) -> tuple[FlagEvaluation, FlagEvaluation]:
    """Evaluate a flag on the first and second chronological parts of the labelled rows.

    The split is by time order only and nothing is fitted, so the second part is
    simply later data; it is not a tuned out-of-sample test. False alarms per
    hour divide by the observed candle time of the eligible rows (rows x
    candle_duration), which excludes unobserved gaps.
    """

    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be strictly between 0 and 1.")
    by_index = {s.index: s for s in snapshots}
    rows = sorted(
        (l for l in labels if l.horizon == horizon and l.atr_multiple == atr_multiple and l.expanded is not None),
        key=lambda l: l.index,
    )
    cut = int(len(rows) * split_fraction)
    results = []
    for part, chunk in (("first_part", rows[:cut]), ("second_part", rows[cut:])):
        flagged = [l for l in chunk if predicate(by_index[l.index])]
        events = [l for l in chunk if l.expanded]
        hits = [l for l in flagged if l.expanded]
        hours = len(chunk) * candle_duration.total_seconds() / 3600
        precision = len(hits) / len(flagged) if flagged else None
        base = len(events) / len(chunk) if chunk else None
        leads = [l.time_to_threshold for l in hits if l.time_to_threshold is not None]
        results.append(
            FlagEvaluation(
                flag=name, part=part, eligible=len(chunk), flagged=len(flagged),
                events=len(events), true_positives=len(hits), precision=precision,
                recall=len(hits) / len(events) if events else None,
                miss_rate=1 - len(hits) / len(events) if events else None,
                base_rate=base,
                lift=precision / base if precision is not None and base else None,
                false_alarms=len(flagged) - len(hits),
                false_alarms_per_hour=(len(flagged) - len(hits)) / hours if hours else None,
                median_lead_time=median(leads) if leads else None,
            )
        )
    return results[0], results[1]


def run_expansion_research(
    path: str | Path, config: ExpansionConfig | None = None
) -> tuple[ExpansionResearch, tuple[ExpansionSummary, ...]]:
    """Read-only: loads and validates a CSV, never writes or connects anywhere."""

    config = config or ExpansionConfig()
    research = analyze_expansion(load_candles_csv(path), config)
    return research, summarize_labels(research.labels, config)
