import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo

from core.market_data import Candle, validate_candle, validate_candle_sequence


class FoldTimezone(tzinfo):

    def utcoffset(self, value):
        return timedelta(hours=-4 if value.fold == 0 else -5)

    def dst(self, value):
        return timedelta(0)


class TestCandleValidation(unittest.TestCase):

    def make_candle(self, **changes):
        candle = Candle(
            timestamp=datetime(2026, 1, 1, 12, 0),
            open=2000.0,
            high=2010.0,
            low=1990.0,
            close=2005.0,
            tick_volume=100,
        )
        return replace(candle, **changes)

    def test_accepts_valid_candle(self):
        validate_candle(self.make_candle())

    def test_rejects_non_numeric_prices(self):
        for field in ("open", "high", "low", "close"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be a finite number"):
                    validate_candle(self.make_candle(**{field: "not numeric"}))

    def test_rejects_non_finite_prices(self):
        for field in ("open", "high", "low", "close"):
            for invalid_price in (float("nan"), float("inf")):
                with self.subTest(field=field, invalid_price=invalid_price):
                    with self.assertRaisesRegex(ValueError, "must be a finite number"):
                        validate_candle(self.make_candle(**{field: invalid_price}))

    def test_rejects_boolean_prices(self):
        for field in ("open", "high", "low", "close"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be a finite number"):
                    validate_candle(self.make_candle(**{field: True}))

    def test_rejects_extremely_large_integer_prices(self):
        huge_integer = 10**1000

        for field in ("open", "high", "low", "close"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be a finite number"):
                    validate_candle(self.make_candle(**{field: huge_integer}))

    def test_rejects_zero_or_negative_prices(self):
        for field in ("open", "high", "low", "close"):
            for invalid_price in (0, -1):
                with self.subTest(field=field, invalid_price=invalid_price):
                    with self.assertRaisesRegex(ValueError, "must be greater than 0"):
                        validate_candle(self.make_candle(**{field: invalid_price}))

    def test_rejects_inconsistent_high_and_low_values(self):
        inconsistent_candles = (
            self.make_candle(open=2011.0),
            self.make_candle(close=2011.0),
            self.make_candle(open=1989.0),
            self.make_candle(close=1989.0),
        )

        for candle in inconsistent_candles:
            with self.subTest(candle=candle):
                with self.assertRaisesRegex(ValueError, "High price|Low price"):
                    validate_candle(candle)

    def test_rejects_negative_tick_volume(self):
        with self.assertRaisesRegex(ValueError, "Tick volume cannot be negative"):
            validate_candle(self.make_candle(tick_volume=-1))

    def test_rejects_fractional_or_boolean_tick_volume(self):
        for tick_volume in (1.5, True, False, float("nan"), float("inf")):
            with self.subTest(tick_volume=tick_volume):
                with self.assertRaisesRegex(ValueError, "Tick volume must be an integer"):
                    validate_candle(self.make_candle(tick_volume=tick_volume))

    def test_accepts_zero_tick_volume(self):
        validate_candle(self.make_candle(tick_volume=0))

    def test_rejects_duplicate_timestamps(self):
        candle = self.make_candle()

        with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
            validate_candle_sequence([candle, candle])

    def test_rejects_timestamps_out_of_order(self):
        first = self.make_candle(timestamp=datetime(2026, 1, 1, 12, 1))
        second = self.make_candle(timestamp=datetime(2026, 1, 1, 12, 0))

        with self.assertRaisesRegex(ValueError, "out of order"):
            validate_candle_sequence([first, second])

    def test_rejects_mixed_naive_and_aware_timestamps(self):
        naive = self.make_candle()
        aware = self.make_candle(
            timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)
        )

        with self.assertRaisesRegex(ValueError, "do not mix"):
            validate_candle_sequence([naive, aware])

    def test_accepts_both_dst_folds_as_distinct_ascending_instants(self):
        fold_timezone = FoldTimezone()
        first_fold = self.make_candle(
            timestamp=datetime(2025, 11, 2, 1, 30, tzinfo=fold_timezone, fold=0)
        )
        second_fold = self.make_candle(
            timestamp=datetime(2025, 11, 2, 1, 30, tzinfo=fold_timezone, fold=1)
        )

        validate_candle_sequence([first_fold, second_fold])

    def test_detects_aware_timestamps_for_the_same_instant_as_duplicates(self):
        utc_timestamp = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        offset_timestamp = datetime(
            2026, 1, 1, 13, tzinfo=timezone(timedelta(hours=1))
        )
        first = self.make_candle(timestamp=utc_timestamp)
        second = self.make_candle(timestamp=offset_timestamp)

        with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
            validate_candle_sequence([first, second])

    def test_accepts_empty_sequence(self):
        validate_candle_sequence([])

    def test_accepts_ascending_sequence_without_changing_order(self):
        first = self.make_candle()
        second = self.make_candle(timestamp=first.timestamp + timedelta(minutes=1))
        candles = [first, second]

        validate_candle_sequence(candles)

        self.assertEqual(candles, [first, second])


if __name__ == "__main__":
    unittest.main()