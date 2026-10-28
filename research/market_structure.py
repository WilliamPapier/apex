"""Neutral, causal market-structure detection for historical candles.

Definitions (all comparisons use candle counts, never clock time, so gaps in
the data are neither filled nor interpreted):

* Swing-high candidate at index i: i >= left_lookback and the candle's high
  exceeds the highs of the ``left_lookback`` preceding candles (see
  ``equal_policy``). It is knowable when candle i closes.
* Confirmed swing high at index i: a candidate whose high also exceeds the
  highs of the ``right_lookback`` following candles. It becomes knowable only
  when candle ``i + right_lookback`` closes (``confirmed_at``).
* Swing lows mirror this using lows and "below".
* ``equal_policy="strict"``: neighbours must be strictly lower (highs) or
  strictly higher (lows); equal highs/lows therefore produce no swing.
  ``"inclusive"``: neighbours may be equal, so each candle of a flat plateau
  qualifies as its own swing.
* A structural break of a confirmed swing happens on the first candle j with
  ``j > confirmation_index`` whose close (``break_basis="close"``) or high/low
  (``"wick"``) is strictly beyond the swing price. Touching the level exactly
  is not a break. Each swing is broken at most once; one candle may break
  several swings, giving one event per swing.
* Candle timestamps are candle-open times; a candle is knowable at its open
  plus ``candle_duration``, which must equal the true timeframe of the data
  (the module cannot detect a mismatch). ``available_at`` and ``confirmed_at``
  are returned in UTC; the swing/break candle ``timestamp`` fields keep the
  input candle's timezone. Gaps are never filled: lookbacks count candles, and
  availability depends only on the closing candle's own timestamp.

Events are neutral observations. They are not MSS, BMS, BOS or CHOCH, and no
trading rule is implied. Output is a pure function of the input prefix:
appending candles never changes earlier events.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Sequence

from core.market_data import Candle, validate_candle_sequence
from data.historical_data_loader import load_candles_csv
from data.mt5_history_collector import M5_INTERVAL


SwingKind = Literal["high", "low"]
BreakDirection = Literal["up", "down"]
EQUAL_POLICIES = ("strict", "inclusive")
BREAK_BASES = ("close", "wick")


@dataclass(frozen=True)
class StructureConfig:
    left_lookback: int = 2
    right_lookback: int = 2
    equal_policy: str = "strict"
    break_basis: str = "close"
    candle_duration: timedelta = M5_INTERVAL

    def __post_init__(self) -> None:
        for name in ("left_lookback", "right_lookback"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")
        if self.equal_policy not in EQUAL_POLICIES:
            raise ValueError(f"equal_policy must be one of {EQUAL_POLICIES}.")
        if self.break_basis not in BREAK_BASES:
            raise ValueError(f"break_basis must be one of {BREAK_BASES}.")
        if (
            not isinstance(self.candle_duration, timedelta)
            or self.candle_duration <= timedelta(0)
        ):
            raise ValueError("candle_duration must be a positive timedelta.")


@dataclass(frozen=True)
class SwingCandidate:
    kind: SwingKind
    candle_index: int
    timestamp: datetime
    price: float
    available_at: datetime


@dataclass(frozen=True)
class SwingPoint:
    kind: SwingKind
    candle_index: int
    timestamp: datetime
    price: float
    confirmation_index: int
    confirmed_at: datetime


@dataclass(frozen=True)
class StructuralBreak:
    direction: BreakDirection
    break_index: int
    break_timestamp: datetime
    available_at: datetime
    level: float
    break_price: float
    swing: SwingPoint


@dataclass(frozen=True)
class MarketStructure:
    swing_highs: tuple[SwingPoint, ...]
    swing_lows: tuple[SwingPoint, ...]
    breaks: tuple[StructuralBreak, ...]


def _prepare(candles: Sequence[Candle], config: StructureConfig) -> None:
    if not isinstance(config, StructureConfig):
        raise ValueError("config must be a StructureConfig.")
    validate_candle_sequence(candles)
    if any(candle.timestamp.utcoffset() is None for candle in candles):
        raise ValueError("Candle timestamps must be timezone-aware.")


def _closes_at(candle: Candle, config: StructureConfig) -> datetime:
    """Instant the candle closes, as a UTC-aware time.

    Arithmetic is done in UTC because adding a timedelta to a non-fixed-offset
    aware datetime is wall-clock arithmetic and is wrong around DST folds.
    """

    return candle.timestamp.astimezone(timezone.utc) + config.candle_duration


def _beats(value: float, others: Sequence[float], kind: SwingKind, policy: str) -> bool:
    if kind == "high":
        return all(value >= o if policy == "inclusive" else value > o for o in others)
    return all(value <= o if policy == "inclusive" else value < o for o in others)


def _price(candle: Candle, kind: SwingKind) -> float:
    return candle.high if kind == "high" else candle.low


def _candidates(
    candles: Sequence[Candle], config: StructureConfig, kind: SwingKind
) -> tuple[SwingCandidate, ...]:
    _prepare(candles, config)
    prices = [_price(candle, kind) for candle in candles]
    left = config.left_lookback
    return tuple(
        SwingCandidate(
            kind,
            index,
            candles[index].timestamp,
            prices[index],
            _closes_at(candles[index], config),
        )
        for index in range(left, len(candles))
        if _beats(prices[index], prices[index - left : index], kind, config.equal_policy)
    )


def _confirmed(
    candles: Sequence[Candle], config: StructureConfig, kind: SwingKind
) -> tuple[SwingPoint, ...]:
    _prepare(candles, config)
    right = config.right_lookback
    prices = [_price(candle, kind) for candle in candles]
    swings = []
    for candidate in _candidates(candles, config, kind):
        index = candidate.candle_index
        confirmation_index = index + right
        if confirmation_index >= len(candles):
            continue
        if _beats(
            candidate.price,
            prices[index + 1 : confirmation_index + 1],
            kind,
            config.equal_policy,
        ):
            swings.append(
                SwingPoint(
                    kind,
                    index,
                    candidate.timestamp,
                    candidate.price,
                    confirmation_index,
                    _closes_at(candles[confirmation_index], config),
                )
            )
    return tuple(swings)


def find_swing_high_candidates(
    candles: Sequence[Candle], config: StructureConfig
) -> tuple[SwingCandidate, ...]:
    """Highs satisfying only the left-side condition; not yet confirmed."""

    return _candidates(candles, config, "high")


def find_swing_low_candidates(
    candles: Sequence[Candle], config: StructureConfig
) -> tuple[SwingCandidate, ...]:
    """Lows satisfying only the left-side condition; not yet confirmed."""

    return _candidates(candles, config, "low")


def find_confirmed_swing_highs(
    candles: Sequence[Candle], config: StructureConfig
) -> tuple[SwingPoint, ...]:
    """Swing highs with both sides satisfied, ordered by candle index."""

    return _confirmed(candles, config, "high")


def find_confirmed_swing_lows(
    candles: Sequence[Candle], config: StructureConfig
) -> tuple[SwingPoint, ...]:
    """Swing lows with both sides satisfied, ordered by candle index."""

    return _confirmed(candles, config, "low")


def find_structural_breaks(
    candles: Sequence[Candle],
    swing_highs: Sequence[SwingPoint],
    swing_lows: Sequence[SwingPoint],
    config: StructureConfig,
) -> tuple[StructuralBreak, ...]:
    """Find first breaks of swings, using a swing only after it is confirmed.

    Events are ordered by break candle, then swing candle, with up breaks
    before down breaks.
    """

    _prepare(candles, config)
    events: list[StructuralBreak] = []
    for direction, swings in (("up", swing_highs), ("down", swing_lows)):
        for swing in swings:
            for index in range(swing.confirmation_index + 1, len(candles)):
                candle = candles[index]
                if direction == "up":
                    price = candle.close if config.break_basis == "close" else candle.high
                    broke = price > swing.price
                else:
                    price = candle.close if config.break_basis == "close" else candle.low
                    broke = price < swing.price
                if broke:
                    events.append(
                        StructuralBreak(
                            direction,
                            index,
                            candle.timestamp,
                            _closes_at(candle, config),
                            swing.price,
                            price,
                            swing,
                        )
                    )
                    break
    events.sort(
        key=lambda event: (
            event.break_index,
            event.swing.candle_index,
            event.direction != "up",
        )
    )
    return tuple(events)


def analyze_market_structure(
    candles: Sequence[Candle], config: StructureConfig | None = None
) -> MarketStructure:
    """Detect confirmed swings and structural breaks in one call."""

    config = config or StructureConfig()
    highs = find_confirmed_swing_highs(candles, config)
    lows = find_confirmed_swing_lows(candles, config)
    return MarketStructure(highs, lows, find_structural_breaks(candles, highs, lows, config))


@dataclass(frozen=True)
class StructureDiagnostics:
    candle_count: int
    confirmed_swing_highs: int
    confirmed_swing_lows: int
    upward_breaks: int
    downward_breaks: int
    swing_highs_by_utc_day: tuple[tuple[date, int], ...]
    swing_lows_by_utc_day: tuple[tuple[date, int], ...]
    upward_breaks_by_utc_day: tuple[tuple[date, int], ...]
    downward_breaks_by_utc_day: tuple[tuple[date, int], ...]
    sample_swings: tuple[SwingPoint, ...]
    sample_breaks: tuple[StructuralBreak, ...]


def _by_day(timestamps: Sequence[datetime]) -> tuple[tuple[date, int], ...]:
    counts = Counter(timestamp.astimezone(timezone.utc).date() for timestamp in timestamps)
    return tuple(sorted(counts.items()))


def diagnose_market_structure(
    candles: Sequence[Candle],
    config: StructureConfig | None = None,
    sample_size: int = 5,
) -> StructureDiagnostics:
    """Descriptive counts only. Swings are counted by swing-candle UTC day and
    breaks by break-candle UTC day. Counts carry no performance meaning."""

    if type(sample_size) is not int or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer.")
    structure = analyze_market_structure(candles, config)
    up = [event for event in structure.breaks if event.direction == "up"]
    down = [event for event in structure.breaks if event.direction == "down"]
    swings = sorted(
        structure.swing_highs + structure.swing_lows,
        key=lambda swing: (swing.candle_index, swing.kind != "high"),
    )
    return StructureDiagnostics(
        candle_count=len(candles),
        confirmed_swing_highs=len(structure.swing_highs),
        confirmed_swing_lows=len(structure.swing_lows),
        upward_breaks=len(up),
        downward_breaks=len(down),
        swing_highs_by_utc_day=_by_day([s.timestamp for s in structure.swing_highs]),
        swing_lows_by_utc_day=_by_day([s.timestamp for s in structure.swing_lows]),
        upward_breaks_by_utc_day=_by_day([e.break_timestamp for e in up]),
        downward_breaks_by_utc_day=_by_day([e.break_timestamp for e in down]),
        sample_swings=tuple(swings[:sample_size]),
        sample_breaks=structure.breaks[:sample_size],
    )


def diagnose_historical_csv(
    path: str | Path,
    config: StructureConfig | None = None,
    sample_size: int = 5,
) -> StructureDiagnostics:
    """Load an existing APEX CSV (read-only, no MT5) and diagnose structure."""

    return diagnose_market_structure(load_candles_csv(path), config, sample_size)
