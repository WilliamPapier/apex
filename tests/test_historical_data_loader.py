import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from data.historical_data_loader import (
    CANDLE_CSV_COLUMNS,
    HistoricalDataError,
    load_candles_csv,
)


class TestHistoricalDataLoader(unittest.TestCase):

    def write_csv(self, directory, rows, columns=CANDLE_CSV_COLUMNS, filename="sample.csv"):
        path = Path(directory) / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def valid_row(self, timestamp="2025-09-01T00:00:00+00:00"):
        return [timestamp, "10", "12", "9", "11", "25"]

    def test_loads_schema_and_normalizes_timestamps_to_utc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                [
                    self.valid_row("2025-09-01T02:00:00+02:00"),
                    self.valid_row("2025-09-01T02:05:00+02:00"),
                ],
            )
            candles = load_candles_csv(path)

        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].timestamp, datetime(2025, 9, 1, tzinfo=timezone.utc))
        self.assertEqual(candles[1].timestamp, datetime(2025, 9, 1, 0, 5, tzinfo=timezone.utc))

    def test_rejects_empty_header_only_and_missing_column_files(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(HistoricalDataError, "empty"):
                load_candles_csv(empty)

            header_only = self.write_csv(directory, [])
            with self.assertRaisesRegex(HistoricalDataError, "no candle rows"):
                load_candles_csv(header_only)

            missing_column = self.write_csv(
                directory,
                [self.valid_row()],
                columns=CANDLE_CSV_COLUMNS[:-1],
                filename="missing-column.csv",
            )
            with self.assertRaisesRegex(HistoricalDataError, "columns"):
                load_candles_csv(missing_column)

    def test_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_field = self.write_csv(
                directory, [["2025-09-01T00:00:00+00:00", "10"]]
            )
            with self.assertRaisesRegex(HistoricalDataError, "row 2 has 2 fields"):
                load_candles_csv(missing_field)

            extra_field = self.write_csv(
                directory,
                [self.valid_row() + ["unexpected"]],
                filename="extra-field.csv",
            )
            with self.assertRaisesRegex(HistoricalDataError, "row 2 has 7 fields"):
                load_candles_csv(extra_field)

    def test_rejects_malformed_csv_quoting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.csv"
            path.write_text(
                "timestamp_utc,open,high,low,close,tick_volume\n"
                '"2025-09-01T00:00:00+00:00,10,12,9,11,25\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(HistoricalDataError, "malformed CSV"):
                load_candles_csv(path)

    def test_rejects_naive_timestamp_invalid_ohlc_and_invalid_values(self):
        cases = (
            (self.valid_row("2025-09-01T00:00:00"), "timezone-aware"),
            (["2025-09-01T00:00:00+00:00", "10", "9", "8", "9", "25"], "High price"),
            (["2025-09-01T00:00:00+00:00", "10", "12", "9", "11", "1.5"], "invalid literal"),
            (["2025-09-01T00:00:00+00:00", "nan", "12", "9", "11", "25"], "finite"),
        )
        for row, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(HistoricalDataError, message):
                    load_candles_csv(self.write_csv(directory, [row]))

    def test_rejects_duplicate_and_out_of_order_timestamps(self):
        first = self.valid_row("2025-09-01T00:00:00+00:00")
        duplicate = self.valid_row("2025-09-01T00:00:00+00:00")
        later = self.valid_row("2025-09-01T00:05:00+00:00")
        cases = (([first, duplicate], "duplicate"), ([later, first], "out of order"))

        for rows, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(HistoricalDataError, message):
                    load_candles_csv(self.write_csv(directory, rows))

    def test_enforces_optional_half_open_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                [
                    self.valid_row("2025-09-01T00:00:00+00:00"),
                    self.valid_row("2025-09-01T00:05:00+00:00"),
                ],
            )
            start = datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)
            end = datetime(2025, 9, 1, 0, 5, tzinfo=timezone.utc)
            with self.assertRaisesRegex(HistoricalDataError, "outside"):
                load_candles_csv(path, start_utc=start, end_utc=end)
            with self.assertRaisesRegex(ValueError, "supplied together"):
                load_candles_csv(path, start_utc=start)


if __name__ == "__main__":
    unittest.main()