"""Deterministic validation for historical or live market candles."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from typing import Sequence


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False

    try:
        return isfinite(value)
    except (OverflowError, TypeError):
        return False


def validate_candle(candle: Candle) -> None:
    """Raise ValueError when a candle has invalid timestamps, prices, or volume."""

    if not isinstance(candle.timestamp, datetime):
        raise ValueError("Timestamp must be a datetime value.")

    prices = {
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
    }
    for name, price in prices.items():
        if not _is_finite_number(price):
            raise ValueError(f"{name.capitalize()} price must be a finite number.")
        if price <= 0:
            raise ValueError(f"{name.capitalize()} price must be greater than 0.")

    if candle.high < candle.open or candle.high < candle.close:
        raise ValueError("High price must be at least as high as open and close.")
    if candle.low > candle.open or candle.low > candle.close:
        raise ValueError("Low price must be at most as low as open and close.")
    if candle.high < candle.low:
        raise ValueError("High price must be greater than or equal to low price.")

    if isinstance(candle.tick_volume, bool) or not isinstance(candle.tick_volume, int):
        raise ValueError("Tick volume must be an integer.")
    if candle.tick_volume < 0:
        raise ValueError("Tick volume cannot be negative.")


def validate_candle_sequence(candles: Sequence[Candle]) -> None:
    """Validate candles in their given order; an empty sequence is valid.

    Candles are never sorted, removed, or changed. Duplicate timestamps and
    timestamps earlier than the preceding candle raise ValueError. Naive
    timestamps remain timezone-unspecified and are accepted only alongside
    other naive timestamps; they are never assumed to be UTC. Aware timestamps
    are compared as UTC instants, so UTC-aware timestamps are preferred.
    """

    seen_timestamps: set[datetime] = set()
    previous_timestamp: datetime | None = None
    sequence_is_aware: bool | None = None

    for index, candle in enumerate(candles):
        validate_candle(candle)
        timestamp = candle.timestamp
        timestamp_is_aware = timestamp.utcoffset() is not None

        if sequence_is_aware is None:
            sequence_is_aware = timestamp_is_aware
        elif timestamp_is_aware != sequence_is_aware:
            raise ValueError(
                "All candle timestamps must be either timezone-naive or "
                "timezone-aware; do not mix them."
            )

        comparable_timestamp = (
            timestamp.astimezone(timezone.utc) if timestamp_is_aware else timestamp
        )

        if comparable_timestamp in seen_timestamps:
            raise ValueError(f"Candle at index {index} has a duplicate timestamp.")
        seen_timestamps.add(comparable_timestamp)

        if previous_timestamp is not None:
            if comparable_timestamp < previous_timestamp:
                raise ValueError(
                    f"Candle at index {index} is out of order; timestamps must ascend."
                )

        previous_timestamp = comparable_timestamp