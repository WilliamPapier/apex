"""Offline structural and descriptive audits for historical candles."""

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import fmean, median
from typing import Sequence

from core.market_data import Candle, validate_candle
from data.mt5_history_collector import M5_INTERVAL, PotentialGap


@dataclass(frozen=True)
class AuditIssue:
    candle_index: int
    message: str


@dataclass(frozen=True)
class TimestampOrderIssue:
    previous_index: int
    candle_index: int
    previous_timestamp: datetime
    timestamp: datetime
    elapsed: timedelta


@dataclass(frozen=True)
class DuplicateTimestampIssue:
    timestamp: datetime
    candle_indices: tuple[int, ...]


@dataclass(frozen=True)
class NumericSummary:
    count: int
    minimum: float | None
    median: float | None
    mean: float | None
    maximum: float | None


@dataclass(frozen=True)
class DatasetAudit:
    total_candles: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    counts_by_utc_date: tuple[tuple[date, int], ...]
    counts_by_utc_hour: tuple[tuple[int, int], ...]
    five_minute_intervals: int
    potential_gaps: tuple[PotentialGap, ...]
    duplicate_timestamps: tuple[DuplicateTimestampIssue, ...]
    out_of_order_pairs: tuple[TimestampOrderIssue, ...]
    invalid_timestamps: tuple[AuditIssue, ...]
    invalid_candles: tuple[AuditIssue, ...]
    candle_range_statistics: NumericSummary
    absolute_body_statistics: NumericSummary


def _summarize(values: list[float]) -> NumericSummary:
    if not values:
        return NumericSummary(0, None, None, None, None)
    return NumericSummary(
        count=len(values),
        minimum=min(values),
        median=float(median(values)),
        mean=fmean(values),
        maximum=max(values),
    )


def audit_candles(candles: Sequence[Candle]) -> DatasetAudit:
    """Audit candles without mutating, sorting, dropping, or filling them.

    Long intervals are reported as PotentialGap observations. No broker
    schedule is assumed. Date/hour counts include rows with valid aware
    timestamps even if that candle has an OHLC or volume validation issue.
    Descriptive price statistics include only individually valid candles.
    """

    normalized_timestamps: list[datetime | None] = []
    invalid_timestamps: list[AuditIssue] = []
    invalid_candles: list[AuditIssue] = []
    valid_ranges: list[float] = []
    valid_bodies: list[float] = []
    daily_counts: Counter[date] = Counter()
    hourly_counts: Counter[int] = Counter()

    for index, candle in enumerate(candles):
        try:
            validate_candle(candle)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            invalid_candles.append(AuditIssue(index, str(error)))
        else:
            valid_ranges.append(candle.high - candle.low)
            valid_bodies.append(abs(candle.close - candle.open))

        timestamp = getattr(candle, "timestamp", None)
        if not isinstance(timestamp, datetime):
            invalid_timestamps.append(
                AuditIssue(index, "Timestamp must be a datetime value.")
            )
            normalized_timestamps.append(None)
            continue
        try:
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("Timestamp must be timezone-aware.")
            utc_timestamp = timestamp.astimezone(timezone.utc)
        except (OverflowError, TypeError, ValueError) as error:
            invalid_timestamps.append(AuditIssue(index, str(error)))
            normalized_timestamps.append(None)
            continue

        normalized_timestamps.append(utc_timestamp)
        daily_counts[utc_timestamp.date()] += 1
        hourly_counts[utc_timestamp.hour] += 1

    indices_by_timestamp: dict[datetime, list[int]] = {}
    for index, timestamp in enumerate(normalized_timestamps):
        if timestamp is not None:
            indices_by_timestamp.setdefault(timestamp, []).append(index)
    duplicates = tuple(
        DuplicateTimestampIssue(timestamp, tuple(indices))
        for timestamp, indices in sorted(indices_by_timestamp.items())
        if len(indices) > 1
    )

    five_minute_intervals = 0
    potential_gaps: list[PotentialGap] = []
    out_of_order_pairs: list[TimestampOrderIssue] = []
    for index, (previous, current) in enumerate(
        zip(normalized_timestamps, normalized_timestamps[1:]), start=1
    ):
        if previous is None or current is None:
            continue
        elapsed = current - previous
        if elapsed <= timedelta(0):
            out_of_order_pairs.append(
                TimestampOrderIssue(index - 1, index, previous, current, elapsed)
            )
        elif elapsed == M5_INTERVAL:
            five_minute_intervals += 1
        elif elapsed > M5_INTERVAL:
            potential_gaps.append(PotentialGap(previous, current, elapsed))

    valid_timestamps = [
        timestamp for timestamp in normalized_timestamps if timestamp is not None
    ]
    return DatasetAudit(
        total_candles=len(candles),
        first_timestamp=valid_timestamps[0] if valid_timestamps else None,
        last_timestamp=valid_timestamps[-1] if valid_timestamps else None,
        counts_by_utc_date=tuple(sorted(daily_counts.items())),
        counts_by_utc_hour=tuple((hour, hourly_counts[hour]) for hour in range(24)),
        five_minute_intervals=five_minute_intervals,
        potential_gaps=tuple(potential_gaps),
        duplicate_timestamps=duplicates,
        out_of_order_pairs=tuple(out_of_order_pairs),
        invalid_timestamps=tuple(invalid_timestamps),
        invalid_candles=tuple(invalid_candles),
        candle_range_statistics=_summarize(valid_ranges),
        absolute_body_statistics=_summarize(valid_bodies),
    )