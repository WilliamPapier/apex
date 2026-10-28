import unittest
from datetime import datetime, timedelta, timezone

from core.market_data import Candle
from research.market_features import calculate_market_features


UTC = timezone.utc


def make_candle(
    minute: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=datetime(2025, 9, 1, tzinfo=UTC) + timedelta(minutes=minute),
        open=open_price,
        high=high,
        low=low,
        close=close,
        tick_volume=1,
    )


class TestMarketFeatures(unittest.TestCase):

    def test_calculates_ohlc_features_and_candle_direction(self):
        candles = (
            make_candle(0, 10, 12, 9, 11),
            make_candle(5, 11, 14, 10, 13),
            make_candle(10, 14, 15, 12, 13),
        )

        features = calculate_market_features(
            candles, volatility_periods=(), atr_period=2
        )

        self.assertEqual(features[0].absolute_range, 3)
        self.assertEqual(features[0].absolute_body_size, 1)
        self.assertEqual(features[0].upper_wick_size, 1)
        self.assertEqual(features[0].lower_wick_size, 1)
        self.assertAlmostEqual(features[0].body_to_range_ratio, 1 / 3)
        self.assertEqual(features[0].direction, "bullish")
        self.assertEqual(features[1].direction, "bullish")
        self.assertEqual(features[2].direction, "bearish")
        self.assertEqual(features[1].absolute_range, 4)
        self.assertEqual(features[1].absolute_body_size, 2)
        self.assertEqual(features[1].upper_wick_size, 1)
        self.assertEqual(features[1].lower_wick_size, 1)

    def test_neutral_candle_and_zero_range_are_safe(self):
        candle = make_candle(0, 10, 10, 10, 10)

        feature = calculate_market_features(
            (candle,), volatility_periods=(), atr_period=1
        )[0]

        self.assertEqual(feature.direction, "neutral")
        self.assertEqual(feature.absolute_range, 0)
        self.assertEqual(feature.absolute_body_size, 0)
        self.assertIsNone(feature.body_to_range_ratio)
        self.assertEqual(feature.average_true_range, 0)

    def test_close_changes_returns_and_realized_volatility_warmup(self):
        candles = (
            make_candle(0, 10, 11, 9, 10),
            make_candle(5, 10, 12, 10, 11),
            make_candle(10, 11, 11, 9, 10),
        )

        features = calculate_market_features(
            candles, volatility_periods=(1, 2), atr_period=5
        )

        self.assertIsNone(features[0].close_to_close_change)
        self.assertIsNone(features[0].close_to_close_percentage_return)
        self.assertEqual(features[0].rolling_realized_volatility, ((1, None), (2, None)))
        self.assertEqual(features[1].close_to_close_change, 1)
        self.assertAlmostEqual(features[1].close_to_close_percentage_return, 0.1)
        self.assertAlmostEqual(
            features[1].rolling_realized_volatility[0][1], 0.1
        )
        self.assertIsNone(features[1].rolling_realized_volatility[1][1])
        self.assertEqual(features[2].close_to_close_change, -1)
        self.assertAlmostEqual(
            features[2].close_to_close_percentage_return, -1 / 11
        )
        self.assertAlmostEqual(
            features[2].rolling_realized_volatility[1][1],
            (0.1**2 + (-1 / 11) ** 2) ** 0.5,
        )
        self.assertTrue(
            all(feature.average_true_range is None for feature in features)
        )

    def test_atr_uses_true_range_mean_initialization_and_wilder_smoothing(self):
        candles = (
            make_candle(0, 10, 12, 9, 11),
            make_candle(5, 11, 14, 10, 13),
            make_candle(10, 13, 15, 12, 14),
            make_candle(15, 14, 16, 13, 13),
        )

        features = calculate_market_features(
            candles, volatility_periods=(), atr_period=2
        )

        self.assertIsNone(features[0].average_true_range)
        self.assertEqual(features[1].average_true_range, 3.5)
        self.assertEqual(features[2].average_true_range, 3.25)
        self.assertEqual(features[3].average_true_range, 3.125)

    def test_future_candles_do_not_change_prior_features(self):
        candles = (
            make_candle(0, 10, 12, 9, 11),
            make_candle(5, 11, 14, 10, 13),
            make_candle(10, 13, 100, 1, 90),
        )
        kwargs = {"volatility_periods": (1, 2), "atr_period": 2}

        prefix_features = calculate_market_features(candles[:2], **kwargs)
        full_features = calculate_market_features(candles, **kwargs)

        self.assertEqual(full_features[:2], prefix_features)

    def test_empty_and_insufficient_inputs_and_bad_periods(self):
        self.assertEqual(
            calculate_market_features((), volatility_periods=(1,), atr_period=1),
            (),
        )
        one_candle = (make_candle(0, 10, 11, 9, 10),)
        feature = calculate_market_features(
            one_candle, volatility_periods=(2,), atr_period=2
        )[0]
        self.assertIsNone(feature.rolling_realized_volatility[0][1])
        self.assertIsNone(feature.average_true_range)

        for periods, atr_period in (((0,), 2), ((1, 1), 2), ((1,), 0)):
            with self.subTest(periods=periods, atr_period=atr_period):
                with self.assertRaises(ValueError):
                    calculate_market_features(
                        one_candle,
                        volatility_periods=periods,
                        atr_period=atr_period,
                    )

    def test_rejects_invalid_or_disordered_candles(self):
        invalid = make_candle(0, 10, 9, 8, 10)
        with self.assertRaisesRegex(ValueError, "High price"):
            calculate_market_features((invalid,))

        zero_close = make_candle(0, 1, 1, 1, 0)
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            calculate_market_features((zero_close,))

        later = make_candle(5, 10, 11, 9, 10)
        earlier = make_candle(0, 10, 11, 9, 10)
        with self.assertRaisesRegex(ValueError, "out of order"):
            calculate_market_features((later, earlier))


if __name__ == "__main__":
    unittest.main()
