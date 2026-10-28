import unittest
from typing import cast

from risk.risk_manager import TradeRisk, calculate_position_risk


class TestRiskManager(unittest.TestCase):

    def test_calculates_half_percent_risk(self):
        trade = TradeRisk(10000, 0.005, 2000, 1990, 100)
        result = calculate_position_risk(trade)

        self.assertTrue(result.approved)
        self.assertAlmostEqual(result.risk_amount, 50)
        self.assertAlmostEqual(result.stop_distance, 10)
        self.assertAlmostEqual(result.position_size_lots, 0.05)

    def test_rejects_risk_above_half_percent(self):
        trade = TradeRisk(10000, 0.01, 2000, 1990, 100)
        result = calculate_position_risk(trade)

        self.assertFalse(result.approved)

    def test_rejects_zero_or_negative_risk_fraction(self):
        for risk_fraction in (0, -0.001):
            with self.subTest(risk_fraction=risk_fraction):
                trade = TradeRisk(10000, risk_fraction, 2000, 1990, 100)
                result = calculate_position_risk(trade)

                self.assertFalse(result.approved)

    def test_rejects_risk_fraction_just_above_half_percent(self):
        trade = TradeRisk(10000, 0.005001, 2000, 1990, 100)
        result = calculate_position_risk(trade)

        self.assertFalse(result.approved)

    def test_rejects_zero_balance(self):
        trade = TradeRisk(0, 0.005, 2000, 1990, 100)
        result = calculate_position_risk(trade)

        self.assertFalse(result.approved)

    def test_rejects_negative_balance(self):
        trade = TradeRisk(-10000, 0.005, 2000, 1990, 100)
        result = calculate_position_risk(trade)

        self.assertFalse(result.approved)

    def test_rejects_zero_or_negative_instrument_value(self):
        for instrument_value in (0, -100):
            with self.subTest(instrument_value=instrument_value):
                trade = TradeRisk(10000, 0.005, 2000, 1990, instrument_value)
                result = calculate_position_risk(trade)

                self.assertFalse(result.approved)

    def test_rejects_zero_or_negative_entry_price(self):
        for entry_price in (0, -2000):
            with self.subTest(entry_price=entry_price):
                trade = TradeRisk(10000, 0.005, entry_price, 1990, 100)
                result = calculate_position_risk(trade)

                self.assertFalse(result.approved)

    def test_rejects_zero_or_negative_stop_price(self):
        for stop_loss in (0, -1990):
            with self.subTest(stop_loss=stop_loss):
                trade = TradeRisk(10000, 0.005, 2000, stop_loss, 100)
                result = calculate_position_risk(trade)

                self.assertFalse(result.approved)

    def test_rejects_non_numeric_inputs(self):
        valid_inputs = {
            "account_balance": 10000,
            "risk_fraction": 0.005,
            "entry_price": 2000,
            "stop_loss": 1990,
            "value_per_price_unit_per_lot": 100,
        }

        for field in valid_inputs:
            with self.subTest(field=field):
                inputs = valid_inputs.copy()
                inputs[field] = cast(float, "not numeric")
                result = calculate_position_risk(TradeRisk(**inputs))

                self.assertFalse(result.approved)

    def test_rejects_non_finite_inputs(self):
        valid_inputs = {
            "account_balance": 10000,
            "risk_fraction": 0.005,
            "entry_price": 2000,
            "stop_loss": 1990,
            "value_per_price_unit_per_lot": 100,
        }

        for field in valid_inputs:
            for invalid_value in (float("nan"), float("inf")):
                with self.subTest(field=field, invalid_value=invalid_value):
                    inputs = valid_inputs.copy()
                    inputs[field] = invalid_value
                    result = calculate_position_risk(TradeRisk(**inputs))

                    self.assertFalse(result.approved)

    def test_rejects_equal_entry_and_stop(self):
        trade = TradeRisk(10000, 0.005, 2000, 2000, 100)
        result = calculate_position_risk(trade)

        self.assertFalse(result.approved)


if __name__ == "__main__":
    unittest.main()
