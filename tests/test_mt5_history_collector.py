import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from core.market_data import Candle
from data.mt5_history_collector import (
    HistoryCollectionError,
    HistoryRequestError,
    InconsistentPositionHistoryError,
    collect_m5_history,
    convert_rates_to_candles,
    default_output_path,
    find_potential_gaps,
    PotentialGap,
    rate_to_candle,
    to_utc,
    validate_candles_in_range,
    write_candles_csv,
)


UTC = timezone.utc


def make_rate(timestamp: datetime) -> dict[str, object]:
    return {
        "time": int(timestamp.timestamp()),
        "open": 2000.0,
        "high": 2010.0,
        "low": 1990.0,
        "close": 2005.0,
        "tick_volume": 100,
    }


class FakeRatesApi:
    TIMEFRAME_M5 = 5

    def __init__(
        self,
        rates,
        *,
        inconsistent_position=None,
        duplicate_position=None,
        none_at=None,
        empty_at=None,
        none_chunk_at=None,
        empty_chunk_at=None,
        short_chunk_at=None,
    ):
        self.rates = list(rates)
        self.inconsistent_position = inconsistent_position
        self.duplicate_position = duplicate_position
        self.none_at = set(none_at or ())
        self.empty_at = set(empty_at or ())
        self.none_chunk_at = set(none_chunk_at or ())
        self.empty_chunk_at = set(empty_chunk_at or ())
        self.short_chunk_at = set(short_chunk_at or ())
        self.requests = []

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        self.requests.append((symbol, timeframe, start_pos, count))
        if count > 1 and start_pos in self.none_chunk_at:
            return None
        if count > 1 and start_pos in self.empty_chunk_at:
            return []
        if start_pos in self.none_at:
            return None
        if start_pos in self.empty_at:
            return []

        result = []
        for position in range(start_pos + count - 1, start_pos - 1, -1):
            if position < 1 or position > len(self.rates):
                return []
            rate = dict(self.rates[-position])
            if position == self.inconsistent_position:
                rate["time"] = int(self.rates[-1]["time"]) + 60
            if position == self.duplicate_position:
                rate["time"] = int(self.rates[-(position + 1)]["time"])
            result.append(rate)
        if count > 1 and start_pos in self.short_chunk_at:
            return result[:-1]
        return result

    def last_error(self):
        return (-4, "details intentionally omitted")


class TestRateConversion(unittest.TestCase):

    def test_to_utc_converts_offset_aware_time(self):
        offset_time = datetime(
            2021, 10, 26, 5, tzinfo=timezone(timedelta(hours=2))
        )

        self.assertEqual(to_utc(offset_time), datetime(2021, 10, 26, 3, tzinfo=UTC))

    def test_to_utc_rejects_naive_time(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            to_utc(datetime(2021, 10, 26, 3))

    def test_rate_conversion_creates_utc_candle(self):
        candle = rate_to_candle(make_rate(datetime(2021, 10, 26, 3, tzinfo=UTC)))

        self.assertIsInstance(candle, Candle)
        self.assertEqual(candle.timestamp, datetime(2021, 10, 26, 3, tzinfo=UTC))
        self.assertEqual(candle.tick_volume, 100)

    def test_conversion_distinguishes_none_and_empty(self):
        with self.assertRaisesRegex(HistoryRequestError, "None"):
            convert_rates_to_candles(None)
        with self.assertRaisesRegex(HistoryRequestError, "empty"):
            convert_rates_to_candles([])

    def test_conversion_rejects_invalid_ohlc_and_fractional_volume(self):
        timestamp = datetime(2021, 10, 26, 3, tzinfo=UTC)
        invalid_high = make_rate(timestamp)
        invalid_high["high"] = 1999.0
        invalid_volume = make_rate(timestamp)
        invalid_volume["tick_volume"] = 1.5

        for rate in (invalid_high, invalid_volume):
            with self.subTest(rate=rate):
                with self.assertRaises(HistoryCollectionError):
                    rate_to_candle(rate)


class TestCandleValidationAndGaps(unittest.TestCase):

    def make_candle(self, minute, **changes):
        candle = Candle(
            datetime(2021, 10, 26, 3, minute, tzinfo=UTC),
            2000.0,
            2010.0,
            1990.0,
            2005.0,
            100,
        )
        return Candle(**(candle.__dict__ | changes))

    def test_validates_half_open_range(self):
        start = datetime(2021, 10, 26, 3, 0, tzinfo=UTC)
        end = datetime(2021, 10, 26, 3, 10, tzinfo=UTC)
        candles = (
            self.make_candle(0),
            self.make_candle(5),
        )

        validate_candles_in_range(candles, start, end)
        with self.assertRaisesRegex(HistoryCollectionError, "outside"):
            validate_candles_in_range((*candles, self.make_candle(10)), start, end)

    def test_rejects_duplicate_and_disordered_timestamps(self):
        start = datetime(2021, 10, 26, 3, 0, tzinfo=UTC)
        end = datetime(2021, 10, 26, 4, 0, tzinfo=UTC)
        first = self.make_candle(0)
        duplicate = self.make_candle(0)
        later = self.make_candle(10)

        with self.assertRaisesRegex(HistoryCollectionError, "duplicate"):
            validate_candles_in_range((first, duplicate), start, end)
        with self.assertRaisesRegex(HistoryCollectionError, "out of order"):
            validate_candles_in_range((later, first), start, end)

    def test_reports_potential_gaps_without_filling(self):
        candles = (self.make_candle(0), self.make_candle(15))

        gaps = find_potential_gaps(candles)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].elapsed, timedelta(minutes=15))
        self.assertEqual(gaps[0].intervening_m5_slots, 2)
        self.assertEqual(candles[0].timestamp, datetime(2021, 10, 26, 3, 0, tzinfo=UTC))

    def test_intervening_m5_slots_are_calendar_neutral(self):
        cases = (
            (
                "consecutive",
                datetime(2025, 10, 20, 12, 0, tzinfo=UTC),
                datetime(2025, 10, 20, 12, 5, tzinfo=UTC),
                0,
            ),
            (
                "15-minute interval",
                datetime(2025, 10, 20, 12, 0, tzinfo=UTC),
                datetime(2025, 10, 20, 12, 15, tzinfo=UTC),
                2,
            ),
            (
                "65-minute interval",
                datetime(2025, 10, 20, 20, 55, tzinfo=UTC),
                datetime(2025, 10, 20, 22, 0, tzinfo=UTC),
                12,
            ),
            (
                "weekend-shaped interval",
                datetime(2025, 10, 24, 20, 55, tzinfo=UTC),
                datetime(2025, 10, 26, 22, 5, tzinfo=UTC),
                589,
            ),
        )

        template = self.make_candle(0)
        for label, previous, following, expected_slots in cases:
            with self.subTest(label=label):
                gap = PotentialGap(previous, following, following - previous)
                self.assertEqual(gap.intervening_m5_slots, expected_slots)

                candles = (
                    replace(template, timestamp=previous),
                    replace(template, timestamp=following),
                )
                potential_gaps = find_potential_gaps(candles)
                if following - previous <= timedelta(minutes=5):
                    self.assertEqual(potential_gaps, ())
                else:
                    self.assertEqual(len(potential_gaps), 1)
                    self.assertEqual(
                        potential_gaps[0].intervening_m5_slots, expected_slots
                    )
                self.assertEqual(candles[0].timestamp, previous)
                self.assertEqual(candles[1].timestamp, following)


class TestPositionalCollection(unittest.TestCase):

    def setUp(self):
        self.first_time = datetime(2021, 10, 26, 3, 0, tzinfo=UTC)
        self.rates = [
            make_rate(self.first_time + timedelta(minutes=5 * index))
            for index in range(30)
        ]
        self.start = self.first_time + timedelta(minutes=50)
        self.end = self.first_time + timedelta(minutes=110)

    def collect(self, api, output_path, **kwargs):
        return collect_m5_history(
            api,
            self.start,
            self.end,
            max_chunk_size=3,
            output_path=output_path,
            **kwargs,
        )

    def test_locates_boundaries_and_obeys_chunk_limit(self):
        api = FakeRatesApi(self.rates)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "sample.csv"
            result = self.collect(api, output_path)

        expected_timestamps = tuple(
            self.start + timedelta(minutes=5 * index) for index in range(12)
        )
        self.assertEqual(
            tuple(candle.timestamp for candle in result.candles), expected_timestamps
        )
        chunk_requests = [
            (start_pos, count)
            for _, _, start_pos, count in api.requests
            if count > 1
        ]
        self.assertEqual(chunk_requests, [(9, 3), (12, 3), (15, 3), (18, 3)])
        requested_positions = [
            position
            for start_pos, count in chunk_requests
            for position in range(start_pos, start_pos + count)
        ]
        self.assertEqual(requested_positions, list(range(9, 21)))
        self.assertTrue(all(count <= 3 for _, count in chunk_requests))
        self.assertTrue(
            all(
                symbol == "XAUUSDm" and timeframe == 5
                for symbol, timeframe, _, _ in api.requests
            )
        )

    def test_includes_oldest_candle_when_start_equals_oldest_timestamp(self):
        oldest_timestamp = self.rates[0]["time"]
        start = datetime.fromtimestamp(oldest_timestamp, UTC)
        end = start + timedelta(minutes=5)
        api = FakeRatesApi(self.rates)

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = collect_m5_history(
                api,
                start,
                end,
                max_chunk_size=3,
                output_path=Path(temporary_directory) / "oldest.csv",
            )

        self.assertEqual(len(result.candles), 1)
        self.assertEqual(result.candles[0].timestamp, start)

    def test_rejects_interval_strictly_older_than_available_history(self):
        oldest_timestamp = self.rates[0]["time"]
        oldest = datetime.fromtimestamp(oldest_timestamp, UTC)
        start = oldest - timedelta(minutes=10)
        end = oldest - timedelta(minutes=5)
        api = FakeRatesApi(self.rates)

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(HistoryRequestError, "No candles"):
                collect_m5_history(
                    api,
                    start,
                    end,
                    output_path=Path(temporary_directory) / "older.csv",
                )

    def test_aborts_inconsistent_position_time_mapping(self):
        api = FakeRatesApi(self.rates, inconsistent_position=4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "must-not-exist.csv"
            with self.assertRaises(InconsistentPositionHistoryError):
                self.collect(api, output_path)
            self.assertFalse(output_path.exists())

    def test_aborts_in_chunk_mapping_anomaly_at_unprobed_position(self):
        api = FakeRatesApi(self.rates, duplicate_position=13)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "misindexed.csv"
            with self.assertRaises(HistoryCollectionError):
                self.collect(api, output_path)

            boundary_probes = {
                start_pos
                for _, _, start_pos, count in api.requests
                if count == 1
            }
            self.assertNotIn(13, boundary_probes)
            self.assertFalse(output_path.exists())

    def test_distinguishes_none_from_empty_boundary_probe(self):
        for api, message in (
            (FakeRatesApi(self.rates, none_at={1}), "None.*-4"),
            (FakeRatesApi(self.rates, empty_at={1}), "no bar"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary_directory:
                output_path = Path(temporary_directory) / "must-not-exist.csv"
                with self.assertRaisesRegex(HistoryRequestError, message):
                    self.collect(api, output_path)
                self.assertFalse(output_path.exists())

    def test_rejects_empty_requested_interval(self):
        api = FakeRatesApi(self.rates)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(HistoryRequestError, "No candles"):
                collect_m5_history(
                    api,
                    self.first_time - timedelta(days=1),
                    self.first_time - timedelta(hours=12),
                    max_search_position=32,
                    output_path=Path(temporary_directory) / "empty.csv",
                )

    def test_rejects_invalid_chunk_data_before_writing(self):
        invalid_rates = [dict(rate) for rate in self.rates]
        invalid_rates[-15]["high"] = 1.0
        api = FakeRatesApi(invalid_rates)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "invalid.csv"
            with self.assertRaises(HistoryCollectionError):
                self.collect(api, output_path)
            self.assertFalse(output_path.exists())

    def test_rejects_none_empty_and_short_chunk_responses(self):
        cases = (
            (FakeRatesApi(self.rates, none_chunk_at={9}), "None.*-4"),
            (FakeRatesApi(self.rates, empty_chunk_at={9}), "empty array"),
            (FakeRatesApi(self.rates, short_chunk_at={9}), "returned 2 bars"),
        )
        for api, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary_directory:
                output_path = Path(temporary_directory) / "incomplete.csv"
                with self.assertRaisesRegex(HistoryRequestError, message):
                    self.collect(api, output_path)
                self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_refuses_to_overwrite_existing_csv(self):
        first = rate_to_candle(self.rates[10])
        second = rate_to_candle(self.rates[11])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "existing.csv"
            output_path.write_text("keep this content", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_candles_csv((first, second), output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep this content")

    def test_interrupted_csv_write_leaves_no_final_or_temporary_file(self):
        class FailingWriter:
            def __init__(self):
                self.rows_written = 0

            def writerow(self, row):
                self.rows_written += 1
                if self.rows_written == 2:
                    raise OSError("simulated interrupted write")

        candles = (
            rate_to_candle(self.rates[10]),
            rate_to_candle(self.rates[11]),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "interrupted.csv"
            with patch(
                "data.mt5_history_collector.csv.writer", return_value=FailingWriter()
            ):
                with self.assertRaisesRegex(OSError, "simulated interrupted write"):
                    write_candles_csv(candles, output_path)

            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_csv_written_after_valid_collection(self):
        api = FakeRatesApi(self.rates)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "sample.csv"
            result = self.collect(api, output_path)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(result.output_path, output_path)
        self.assertEqual(len(rows), 13)
        self.assertEqual(rows[0][0], "timestamp_utc")

    def test_default_path_is_under_data_historical(self):
        path = default_output_path("XAUUSDm", self.start, self.end)

        self.assertEqual(path.parent.name, "historical")
        self.assertEqual(path.parent.parent.name, "data")

    def test_rejects_interval_over_total_bar_limit_before_writing(self):
        api = FakeRatesApi(self.rates)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "too-many.csv"
            with self.assertRaisesRegex(HistoryRequestError, "max_total_bars"):
                self.collect(api, output_path, max_total_bars=2)
            self.assertFalse(output_path.exists())
            self.assertTrue(api.requests)
            self.assertTrue(all(count == 1 for _, _, _, count in api.requests))

    def test_collection_result_exposes_gaps_only_as_potential(self):
        rates_with_gap = self.rates[:15] + self.rates[16:]
        api = FakeRatesApi(rates_with_gap)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.collect(
                api, Path(temporary_directory) / "gap-sample.csv"
            )

        self.assertEqual(len(result.potential_gaps), 1)
        gap = result.potential_gaps[0]
        self.assertIsInstance(gap, PotentialGap)
        self.assertEqual(gap.elapsed, timedelta(minutes=10))
        self.assertFalse(hasattr(gap, "confirmed_missing"))

    def test_link_failure_preserves_existing_file_and_attempts_temp_cleanup(self):
        first = rate_to_candle(self.rates[10])
        second = rate_to_candle(self.rates[11])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "existing.csv"
            output_path.write_text("keep this content", encoding="utf-8")
            original_unlink = Path.unlink
            cleanup_attempts = []

            def tracked_unlink(path, *args, **kwargs):
                cleanup_attempts.append(path)
                return original_unlink(path, *args, **kwargs)

            with patch(
                "data.mt5_history_collector.os.link",
                side_effect=OSError("simulated link failure"),
            ), patch.object(Path, "unlink", tracked_unlink):
                with self.assertRaisesRegex(OSError, "simulated link failure"):
                    write_candles_csv((first, second), output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep this content")
            self.assertEqual(len(cleanup_attempts), 1)
            self.assertNotEqual(cleanup_attempts[0], output_path)
            self.assertFalse(cleanup_attempts[0].exists())
            self.assertEqual(list(Path(temporary_directory).iterdir()), [output_path])


if __name__ == "__main__":
    unittest.main()