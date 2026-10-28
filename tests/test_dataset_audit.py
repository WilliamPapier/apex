import unittest
from datetime import datetime, timedelta, timezone

from core.market_data import Candle
from data.dataset_audit import audit_candles


UTC = timezone.utc


def make_candle(timestamp, open_price=10.0, high=12.0, low=9.0, close=11.0, volume=5):
    return Candle(timestamp, open_price, high, low, close, volume)


class TestDatasetAudit(unittest.TestCase):

    def test_counts_regular_intervals_gaps_and_descriptive_statistics(self):
        candles = (
            make_candle(datetime(2025, 9, 1, 23, 55, tzinfo=UTC), 10, 12, 9, 11),
            make_candle(datetime(2025, 9, 2, 0, 0, tzinfo=UTC), 11, 14, 10, 13),
            make_candle(datetime(2025, 9, 2, 0, 15, tzinfo=UTC), 13, 15, 12, 14),
        )

        audit = audit_candles(candles)

        self.assertEqual(audit.total_candles, 3)
        self.assertEqual(audit.first_timestamp, candles[0].timestamp)
        self.assertEqual(audit.last_timestamp, candles[-1].timestamp)
        self.assertEqual(
            audit.counts_by_utc_date,
            ((datetime(2025, 9, 1, tzinfo=UTC).date(), 1), (datetime(2025, 9, 2, tzinfo=UTC).date(), 2)),
        )
        self.assertEqual(dict(audit.counts_by_utc_hour)[23], 1)
        self.assertEqual(dict(audit.counts_by_utc_hour)[0], 2)
        self.assertEqual(audit.five_minute_intervals, 1)
        self.assertEqual(len(audit.potential_gaps), 1)
        self.assertEqual(audit.potential_gaps[0].elapsed, timedelta(minutes=15))
        self.assertEqual(audit.potential_gaps[0].intervening_m5_slots, 2)
        self.assertEqual(audit.candle_range_statistics.minimum, 3.0)
        self.assertEqual(audit.candle_range_statistics.maximum, 4.0)
        self.assertAlmostEqual(audit.candle_range_statistics.mean, 10 / 3)
        self.assertAlmostEqual(audit.absolute_body_statistics.mean, 4 / 3)
        self.assertEqual(candles[0].timestamp, datetime(2025, 9, 1, 23, 55, tzinfo=UTC))

    def test_reports_duplicate_disordered_naive_and_invalid_candles(self):
        first = make_candle(datetime(2025, 9, 1, 0, 0, tzinfo=UTC))
        duplicate = make_candle(datetime(2025, 9, 1, 0, 0, tzinfo=UTC))
        out_of_order = make_candle(datetime(2025, 9, 1, 0, 5, tzinfo=UTC))
        invalid_ohlc = make_candle(datetime(2025, 9, 1, 0, 10, tzinfo=UTC), high=8.0)
        naive = make_candle(datetime(2025, 9, 1, 0, 15))

        audit = audit_candles((first, duplicate, out_of_order, invalid_ohlc, naive))

        self.assertEqual(audit.duplicate_timestamps[0].candle_indices, (0, 1))
        self.assertEqual(len(audit.out_of_order_pairs), 1)
        self.assertEqual(len(audit.invalid_timestamps), 1)
        self.assertEqual(len(audit.invalid_candles), 1)
        self.assertIn("High price", audit.invalid_candles[0].message)

    def test_empty_input_returns_empty_summaries(self):
        audit = audit_candles(())

        self.assertEqual(audit.total_candles, 0)
        self.assertIsNone(audit.first_timestamp)
        self.assertIsNone(audit.last_timestamp)
        self.assertEqual(audit.candle_range_statistics.count, 0)
        self.assertEqual(audit.absolute_body_statistics.count, 0)
        self.assertEqual(audit.counts_by_utc_hour, tuple((hour, 0) for hour in range(24)))


if __name__ == "__main__":
    unittest.main()