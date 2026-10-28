
import unittest

from config.settings import SETTINGS


class TestApexSettings(unittest.TestCase):

    def test_live_trading_is_disabled(self):
        self.assertFalse(SETTINGS.live_trading_enabled)

    def test_initial_risk_is_half_a_percent(self):
        self.assertEqual(SETTINGS.risk_per_trade, 0.005)

    def test_only_one_open_position_is_allowed(self):
        self.assertEqual(SETTINGS.max_open_positions, 1)

    def test_initial_instrument_is_gold(self):
        self.assertEqual(SETTINGS.symbol, "XAUUSDm")


if __name__ == "__main__":
    unittest.main()
