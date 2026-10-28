"""Bounded, read-only collection of MT5 M5 history."""

import csv
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol, Sequence

from core.market_data import Candle, validate_candle_sequence


DEFAULT_MAX_CHUNK_SIZE = 500
DEFAULT_MAX_TOTAL_BARS = 100_000
DEFAULT_MAX_SEARCH_POSITION = 1_000_000
M5_INTERVAL = timedelta(minutes=5)


class RatesApi(Protocol):
    TIMEFRAME_M5: int

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> Any: ...


class HistoryCollectionError(ValueError):
    """Raised when history cannot be retrieved or validated safely."""


class HistoryRequestError(HistoryCollectionError):
    """Raised when an MT5 positional request returns no usable result."""


class InconsistentPositionHistoryError(HistoryCollectionError):
    """Raised when MT5 position indices do not map consistently to time."""


@dataclass(frozen=True)
class PotentialGap:
    previous_timestamp: datetime
    next_timestamp: datetime
    elapsed: timedelta

    @property
    def intervening_m5_slots(self) -> int:
        """Count five-minute spacing slots strictly between the endpoints.

        This calendar-neutral diagnostic is derived from elapsed time only.
        It does not say how many candles should have existed or confirm that
        any candle is missing during a market closure.
        """

        elapsed = self.next_timestamp - self.previous_timestamp
        elapsed_microseconds = (
            (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000
            + elapsed.microseconds
        )
        interval_microseconds = (
            (M5_INTERVAL.days * 86_400 + M5_INTERVAL.seconds) * 1_000_000
            + M5_INTERVAL.microseconds
        )
        if elapsed_microseconds <= 0:
            return 0
        return max(0, (elapsed_microseconds - 1) // interval_microseconds)


@dataclass(frozen=True)
class CollectionResult:
    candles: tuple[Candle, ...]
    potential_gaps: tuple[PotentialGap, ...]
    output_path: Path


def to_utc(value: datetime, name: str = "timestamp") -> datetime:
    """Require an aware datetime and convert it to UTC."""

    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime value.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware; naive times are ambiguous.")
    return value.astimezone(timezone.utc)


def _rate_value(rate: Any, field: str) -> Any:
    try:
        return rate[field]
    except (KeyError, IndexError, TypeError) as error:
        raise HistoryCollectionError(f"MT5 bar is missing the {field!r} field.") from error


def _rate_timestamp(rate: Any) -> datetime:
    raw_timestamp = _rate_value(rate, "time")
    if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, Integral):
        raise HistoryCollectionError("MT5 bar timestamp must be integer Unix seconds.")
    try:
        return datetime.fromtimestamp(int(raw_timestamp), timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise HistoryCollectionError("MT5 bar timestamp is outside the supported range.") from error


def _price_value(rate: Any, field: str) -> float:
    value = _rate_value(rate, field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HistoryCollectionError(f"MT5 {field} price must be numeric.")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise HistoryCollectionError(f"MT5 {field} price must be finite.") from error
    if not isfinite(converted):
        raise HistoryCollectionError(f"MT5 {field} price must be finite.")
    return converted


def rate_to_candle(rate: Any) -> Candle:
    """Convert one MT5 structured rate to a validated UTC Candle."""

    raw_volume = _rate_value(rate, "tick_volume")
    if isinstance(raw_volume, bool) or not isinstance(raw_volume, Integral):
        raise HistoryCollectionError("MT5 tick volume must be an integer.")

    candle = Candle(
        timestamp=_rate_timestamp(rate),
        open=_price_value(rate, "open"),
        high=_price_value(rate, "high"),
        low=_price_value(rate, "low"),
        close=_price_value(rate, "close"),
        tick_volume=int(raw_volume),
    )
    try:
        validate_candle_sequence((candle,))
    except ValueError as error:
        raise HistoryCollectionError(str(error)) from error
    return candle


def convert_rates_to_candles(rates: Any) -> tuple[Candle, ...]:
    """Convert a nonempty MT5 result without changing its order."""

    if rates is None:
        raise HistoryRequestError("MT5 returned None instead of a rate array.")
    if len(rates) == 0:
        raise HistoryRequestError("MT5 returned an empty rate array.")
    return tuple(rate_to_candle(rate) for rate in rates)


def validate_candles_in_range(
    candles: Sequence[Candle], start_utc: datetime, end_utc: datetime
) -> None:
    """Validate candles and enforce the half-open candle-open interval."""

    start = to_utc(start_utc, "start_utc")
    end = to_utc(end_utc, "end_utc")
    if start >= end:
        raise ValueError("start_utc must be earlier than end_utc.")
    if not candles:
        raise HistoryRequestError("No candles were returned for the requested interval.")

    try:
        validate_candle_sequence(candles)
    except ValueError as error:
        raise HistoryCollectionError(str(error)) from error

    for index, candle in enumerate(candles):
        timestamp = to_utc(candle.timestamp, f"candle {index} timestamp")
        if not start <= timestamp < end:
            raise HistoryCollectionError(
                f"Candle at index {index} falls outside [start_utc, end_utc)."
            )


def find_potential_gaps(
    candles: Sequence[Candle], expected_interval: timedelta = M5_INTERVAL
) -> tuple[PotentialGap, ...]:
    """Report intervals longer than expected without labeling them missing."""

    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive.")
    gaps = []
    for previous, following in zip(candles, candles[1:]):
        elapsed = following.timestamp - previous.timestamp
        if elapsed > expected_interval:
            gaps.append(
                PotentialGap(previous.timestamp, following.timestamp, elapsed)
            )
    return tuple(gaps)


def _sanitized_error_code(api: RatesApi) -> str:
    try:
        error = api.last_error()  # type: ignore[attr-defined]
        code = error[0]
    except (AttributeError, IndexError, TypeError):
        return "unavailable"
    if isinstance(code, bool) or not isinstance(code, Integral):
        return "unavailable"
    return str(int(code))


class _PositionLocator:
    """Find position boundaries while checking every observed mapping."""

    def __init__(
        self,
        api: RatesApi,
        symbol: str,
        timeframe: int,
        max_search_position: int,
    ) -> None:
        self.api = api
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_search_position = max_search_position
        self.observed: dict[int, datetime] = {}
        self.empty_positions: set[int] = set()

    def observe(self, position: int, timestamp: datetime) -> None:
        existing = self.observed.get(position)
        if existing is not None and existing != timestamp:
            raise InconsistentPositionHistoryError(
                f"MT5 returned different timestamps for position {position}."
            )
        for other_position, other_timestamp in self.observed.items():
            if other_position < position and not other_timestamp > timestamp:
                raise InconsistentPositionHistoryError(
                    "MT5 position/time mapping is not strictly older at larger positions."
                )
            if other_position > position and not other_timestamp < timestamp:
                raise InconsistentPositionHistoryError(
                    "MT5 position/time mapping is not strictly older at larger positions."
                )
        self.observed[position] = timestamp

    def timestamp_at_or_none(self, position: int) -> datetime | None:
        if position in self.observed:
            return self.observed[position]
        if position in self.empty_positions:
            return None
        rates = self.api.copy_rates_from_pos(
            self.symbol, self.timeframe, position, 1
        )
        if rates is None:
            raise HistoryRequestError(
                f"MT5 returned None while probing position {position} "
                f"(error code {_sanitized_error_code(self.api)})."
            )
        if len(rates) == 0:
            self.empty_positions.add(position)
            return None
        if len(rates) != 1:
            raise HistoryRequestError(
                f"MT5 returned {len(rates)} bars for a one-bar position probe."
            )
        timestamp = _rate_timestamp(rates[0])
        self.observe(position, timestamp)
        return timestamp

    def timestamp_at(self, position: int) -> datetime:
        timestamp = self.timestamp_at_or_none(position)
        if timestamp is None:
            raise HistoryRequestError(
                f"MT5 returned no bar while probing position {position}."
            )
        return timestamp

    def last_available_position(self, valid_position: int, empty_position: int) -> int:
        """Narrow an empty tail to its last valid position."""

        while empty_position - valid_position > 1:
            middle_position = (valid_position + empty_position) // 2
            if self.timestamp_at_or_none(middle_position) is None:
                empty_position = middle_position
            else:
                valid_position = middle_position
        return valid_position

    def first_position_older_than(self, boundary: datetime) -> int:
        """Find the first shift whose candle-open time is before boundary."""

        if self.timestamp_at(1) < boundary:
            return 1

        lower_position = 1
        upper_position = min(2, self.max_search_position)
        if upper_position == lower_position:
            raise InconsistentPositionHistoryError(
                "max_search_position is too small to bracket the boundary."
            )

        while True:
            upper_timestamp = self.timestamp_at_or_none(upper_position)
            if upper_timestamp is None:
                upper_position = self.last_available_position(
                    lower_position, upper_position
                )
                if self.timestamp_at(upper_position) >= boundary:
                    return upper_position + 1
                break
            if upper_timestamp < boundary:
                break
            if upper_position >= self.max_search_position:
                raise HistoryRequestError(
                    "The requested boundary is older than the configured search limit."
                )
            lower_position = upper_position
            upper_position = min(upper_position * 2, self.max_search_position)

        while upper_position - lower_position > 1:
            middle_position = (lower_position + upper_position) // 2
            middle_timestamp = self.timestamp_at_or_none(middle_position)
            if middle_timestamp is None:
                raise InconsistentPositionHistoryError(
                    "MT5 returned an empty position inside a verified history bracket."
                )
            if middle_timestamp < boundary:
                upper_position = middle_position
            else:
                lower_position = middle_position
        return upper_position


def _request_error(api: RatesApi, message: str) -> HistoryRequestError:
    return HistoryRequestError(
        f"{message} (MT5 error code {_sanitized_error_code(api)})."
    )


def default_output_path(
    symbol: str, start_utc: datetime, end_utc: datetime
) -> Path:
    start = to_utc(start_utc, "start_utc")
    end = to_utc(end_utc, "end_utc")
    safe_symbol = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in symbol
    )
    filename = (
        f"{safe_symbol}_M5_{start:%Y%m%dT%H%M%SZ}_"
        f"{end:%Y%m%dT%H%M%SZ}.csv"
    )
    return Path(__file__).resolve().parent / "historical" / filename


def write_candles_csv(candles: Sequence[Candle], output_path: Path) -> Path:
    """Atomically publish validated candles without overwriting an existing file."""

    if not candles:
        raise HistoryRequestError("Cannot write an empty candle sequence.")
    try:
        validate_candle_sequence(candles)
        for index, candle in enumerate(candles):
            to_utc(candle.timestamp, f"candle {index} timestamp")
    except ValueError as error:
        raise HistoryCollectionError(str(error)) from error

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            writer = csv.writer(output_file)
            writer.writerow(
                ("timestamp_utc", "open", "high", "low", "close", "tick_volume")
            )
            for candle in candles:
                writer.writerow(
                    (
                        to_utc(candle.timestamp).isoformat(),
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.tick_volume,
                    )
                )
            output_file.flush()
            os.fsync(output_file.fileno())

        # Linking publishes the completed file atomically and fails if the
        # destination already exists on the same filesystem.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return path


def collect_m5_history(
    api: RatesApi,
    start_utc: datetime,
    end_utc: datetime,
    *,
    symbol: str = "XAUUSDm",
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    max_total_bars: int = DEFAULT_MAX_TOTAL_BARS,
    max_search_position: int = DEFAULT_MAX_SEARCH_POSITION,
    output_path: Path | None = None,
) -> CollectionResult:
    """Collect a bounded UTC interval using positional MT5 requests only.

    The caller owns MT5 initialization and shutdown. This function only reads
    rates and writes the validated result to a new local CSV file.
    """

    start = to_utc(start_utc, "start_utc")
    end = to_utc(end_utc, "end_utc")
    if start >= end:
        raise ValueError("start_utc must be earlier than end_utc.")
    if max_chunk_size < 1 or max_total_bars < 1 or max_search_position < 1:
        raise ValueError("Chunk, total-bar, and search-position limits must be positive.")

    timeframe = api.TIMEFRAME_M5
    locator = _PositionLocator(api, symbol, timeframe, max_search_position)
    newest_included_position = locator.first_position_older_than(end)
    first_older_than_start = locator.first_position_older_than(start)
    oldest_included_position = first_older_than_start - 1

    if oldest_included_position < newest_included_position:
        raise HistoryRequestError("No candles were returned for the requested interval.")

    total_bars = oldest_included_position - newest_included_position + 1
    if total_bars > max_total_bars:
        raise HistoryRequestError(
            f"Requested interval contains {total_bars} bars, exceeding "
            f"max_total_bars={max_total_bars}."
        )

    chunks: list[tuple[Candle, ...]] = []
    current_position = newest_included_position
    while current_position <= oldest_included_position:
        count = min(
            max_chunk_size,
            oldest_included_position - current_position + 1,
        )
        rates = api.copy_rates_from_pos(
            symbol, timeframe, current_position, count
        )
        if rates is None:
            raise _request_error(
                api, f"MT5 returned None for position {current_position}"
            )
        if len(rates) == 0:
            raise HistoryRequestError(
                f"MT5 returned an empty array for position {current_position}."
            )
        if len(rates) != count:
            raise HistoryRequestError(
                f"MT5 returned {len(rates)} bars for a {count}-bar request at "
                f"position {current_position}."
            )

        batch = tuple(rate_to_candle(rate) for rate in rates)
        validate_candles_in_range(batch, start, end)
        returned_positions = range(
            current_position + count - 1, current_position - 1, -1
        )
        for position, candle in zip(returned_positions, batch):
            locator.observe(position, candle.timestamp)
        chunks.append(batch)
        current_position += count

    candles = tuple(candle for batch in reversed(chunks) for candle in batch)
    validate_candles_in_range(candles, start, end)
    gaps = find_potential_gaps(candles)
    destination = output_path or default_output_path(symbol, start, end)
    written_path = write_candles_csv(candles, destination)
    return CollectionResult(candles, gaps, written_path)