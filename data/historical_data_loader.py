"""Strict, offline loading of APEX historical candle CSV files."""

import csv
from datetime import datetime
from pathlib import Path

from core.market_data import Candle, validate_candle, validate_candle_sequence
from data.mt5_history_collector import to_utc, validate_candles_in_range


CANDLE_CSV_COLUMNS = (
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
)


class HistoricalDataError(ValueError):
    """Raised when an APEX historical CSV cannot be loaded safely."""


def _parse_row(row: list[str], row_number: int, source: Path) -> Candle:
    if len(row) != len(CANDLE_CSV_COLUMNS):
        raise HistoricalDataError(
            f"{source}: row {row_number} has {len(row)} fields; "
            f"expected {len(CANDLE_CSV_COLUMNS)}."
        )

    try:
        timestamp = to_utc(datetime.fromisoformat(row[0].strip()), "CSV timestamp")
        candle = Candle(
            timestamp=timestamp,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            tick_volume=int(row[5].strip()),
        )
        validate_candle(candle)
    except (TypeError, ValueError, OverflowError, OSError) as error:
        raise HistoricalDataError(
            f"{source}: invalid candle at row {row_number}: {error}"
        ) from error
    return candle


def load_candles_csv(
    path: str | Path,
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> tuple[Candle, ...]:
    """Load the existing APEX CSV schema without dropping or repairing rows.

    Timestamps must be timezone-aware and are normalized to UTC. Optional
    bounds must be provided together and use [start_utc, end_utc), based on
    candle-open timestamps. Invalid records, duplicates, and disorder raise
    HistoricalDataError; the source file is only read.
    """

    source = Path(path)
    if (start_utc is None) != (end_utc is None):
        raise ValueError("start_utc and end_utc must be supplied together.")

    candles: list[Candle] = []
    try:
        with source.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            columns = next(reader, None)
            if columns is None:
                raise HistoricalDataError(f"{source}: CSV file is empty.")
            if tuple(columns) != CANDLE_CSV_COLUMNS:
                raise HistoricalDataError(
                    f"{source}: columns must be {CANDLE_CSV_COLUMNS}; got {tuple(columns)}."
                )

            for row in reader:
                row_number = reader.line_num
                if not row:
                    raise HistoricalDataError(
                        f"{source}: blank record at row {row_number}."
                    )
                candles.append(_parse_row(row, row_number, source))
    except csv.Error as error:
        raise HistoricalDataError(f"{source}: malformed CSV: {error}") from error

    if not candles:
        raise HistoricalDataError(f"{source}: CSV contains no candle rows.")

    try:
        validate_candle_sequence(candles)
        if start_utc is not None and end_utc is not None:
            validate_candles_in_range(candles, start_utc, end_utc)
    except ValueError as error:
        raise HistoricalDataError(
            f"{source}: invalid candle sequence or range: {error}"
        ) from error

    return tuple(candles)