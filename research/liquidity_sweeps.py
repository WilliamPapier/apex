"""Neutral candidate liquidity-sweep detection against confirmed swing levels.

Definitions (strict inequalities; an exact touch is never a sweep):

* High-side sweep: candle high > swing-high price AND candle close < that price.
* Low-side sweep: candle low < swing-low price AND candle close > that price.
* ``min_penetration`` (default 0.0) additionally requires the penetration
  distance (high minus level, or level minus low) to be strictly greater than
  it. The default does not weaken the definition above.
* The reference swing must be confirmed before the sweep candle begins:
  sweep candle open >= swing ``confirmed_at`` (UTC). Swings come from
  ``research.market_structure`` and are never recomputed here.
* The event is available when the sweep candle closes (open + the structure
  config's ``candle_duration``), returned in UTC.
* One event per (reference swing, sweep candle). By default a swing can be
  swept again by later candles; no expiry or invalidation rule is assumed.
  ``first_sweep_only=True`` keeps only the first sweep of each swing.
* Order: sweep candle index, then high-side before low-side, then swing candle
  index.

Events are observations only: not entries, reversals, MSS/BMS/BOS/CHOCH, and
not evidence of profitability. Gaps are not filled; no schedule is assumed.
Output for a candle prefix never changes when candles are appended.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from core.market_data import Candle
from data.historical_data_loader import load_candles_csv
from research.market_structure import (
    StructureConfig,
    SwingPoint,
    _closes_at,
    _prepare,
    analyze_market_structure,
)


SweepSide = Literal["high", "low"]


@dataclass(frozen=True)
class SweepConfig:
    min_penetration: float = 0.0
    first_sweep_only: bool = False

    def __post_init__(self) -> None:
        value = self.min_penetration
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not value >= 0
            or value == float("inf")
        ):
            raise ValueError("min_penetration must be a finite number of at least 0.")
        if not isinstance(self.first_sweep_only, bool):
            raise ValueError("first_sweep_only must be a bool.")


@dataclass(frozen=True)
class LiquiditySweep:
    side: SweepSide
    sweep_index: int
    sweep_timestamp: datetime
    available_at: datetime
    level: float
    sweep_extreme: float
    close: float
    penetration: float
    swing: SwingPoint


def _check_swings(
    candles: Sequence[Candle], swings: Sequence[SwingPoint], kind: SweepSide
) -> None:
    for swing in swings:
        if not isinstance(swing, SwingPoint) or swing.kind != kind:
            raise ValueError(f"Expected confirmed swing {kind}s from market_structure.")
        index = swing.candle_index
        if not (0 <= index < swing.confirmation_index < len(candles)):
            raise ValueError("Swing indices do not match the supplied candles.")
        candle = candles[index]
        price = candle.high if kind == "high" else candle.low
        if candle.timestamp != swing.timestamp or price != swing.price:
            raise ValueError("Swing does not match the supplied candles.")


def detect_liquidity_sweeps(
    candles: Sequence[Candle],
    swing_highs: Sequence[SwingPoint],
    swing_lows: Sequence[SwingPoint],
    structure_config: StructureConfig,
    config: SweepConfig | None = None,
) -> tuple[LiquiditySweep, ...]:
    """Detect candidate sweeps of already-confirmed swings (see module doc)."""

    config = SweepConfig() if config is None else config
    if not isinstance(config, SweepConfig):
        raise ValueError("config must be a SweepConfig.")
    _prepare(candles, structure_config)
    _check_swings(candles, swing_highs, "high")
    _check_swings(candles, swing_lows, "low")

    events: list[LiquiditySweep] = []
    for side, swings in (("high", swing_highs), ("low", swing_lows)):
        for swing in swings:
            for index in range(swing.confirmation_index + 1, len(candles)):
                candle = candles[index]
                if candle.timestamp.astimezone(timezone.utc) < swing.confirmed_at:
                    continue
                if side == "high":
                    extreme = candle.high
                    penetration = extreme - swing.price
                    swept = penetration > 0 and candle.close < swing.price
                else:
                    extreme = candle.low
                    penetration = swing.price - extreme
                    swept = penetration > 0 and candle.close > swing.price
                if not swept or penetration <= config.min_penetration:
                    continue
                events.append(
                    LiquiditySweep(
                        side,
                        index,
                        candle.timestamp,
                        _closes_at(candle, structure_config),
                        swing.price,
                        extreme,
                        candle.close,
                        penetration,
                        swing,
                    )
                )
                if config.first_sweep_only:
                    break
    events.sort(key=lambda e: (e.sweep_index, e.side != "high", e.swing.candle_index))
    return tuple(events)


def analyze_liquidity_sweeps(
    candles: Sequence[Candle],
    structure_config: StructureConfig | None = None,
    config: SweepConfig | None = None,
) -> tuple[LiquiditySweep, ...]:
    """Confirm swings with market_structure, then detect sweeps of them."""

    structure_config = structure_config or StructureConfig()
    structure = analyze_market_structure(candles, structure_config)
    return detect_liquidity_sweeps(
        candles, structure.swing_highs, structure.swing_lows, structure_config, config
    )


@dataclass(frozen=True)
class SweepDiagnostics:
    candle_count: int
    high_side_sweeps: int
    low_side_sweeps: int
    high_side_by_utc_day: tuple[tuple[date, int], ...]
    low_side_by_utc_day: tuple[tuple[date, int], ...]
    sample_events: tuple[LiquiditySweep, ...]


def _by_day(events: Sequence[LiquiditySweep]) -> tuple[tuple[date, int], ...]:
    counts = Counter(e.sweep_timestamp.astimezone(timezone.utc).date() for e in events)
    return tuple(sorted(counts.items()))


def diagnose_liquidity_sweeps(
    candles: Sequence[Candle],
    structure_config: StructureConfig | None = None,
    config: SweepConfig | None = None,
    sample_size: int = 5,
) -> SweepDiagnostics:
    """Descriptive counts by sweep-candle UTC day plus a leading sample."""

    if type(sample_size) is not int or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer.")
    events = analyze_liquidity_sweeps(candles, structure_config, config)
    high = [e for e in events if e.side == "high"]
    low = [e for e in events if e.side == "low"]
    return SweepDiagnostics(
        len(candles), len(high), len(low), _by_day(high), _by_day(low), events[:sample_size]
    )


def diagnose_historical_csv(
    path: str | Path,
    structure_config: StructureConfig | None = None,
    config: SweepConfig | None = None,
    sample_size: int = 5,
) -> SweepDiagnostics:
    """Load an existing APEX CSV (read-only, no MT5) and diagnose sweeps."""

    return diagnose_liquidity_sweeps(
        load_candles_csv(path), structure_config, config, sample_size
    )
