"""Causal, strategy-independent features calculated from historical candles."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from statistics import fmean
from typing import Literal, Sequence

from core.market_data import Candle, validate_candle_sequence


CandleDirection = Literal["bullish", "bearish", "neutral"]
DEFAULT_VOLATILITY_PERIODS = (5, 20)
DEFAULT_ATR_PERIOD = 14


@dataclass(frozen=True)
class CandleFeatures:
    timestamp: datetime
    absolute_range: float
    absolute_body_size: float
    upper_wick_size: float
    lower_wick_size: float
    body_to_range_ratio: float | None
    direction: CandleDirection
    close_to_close_change: float | None
    close_to_close_percentage_return: float | None
    rolling_realized_volatility: tuple[tuple[int, float | None], ...]
    average_true_range: float | None


def _validate_periods(
    volatility_periods: Sequence[int], atr_period: int
) -> tuple[int, ...]:
    periods = tuple(volatility_periods)
    if any(type(period) is not int or period <= 0 for period in periods):
        raise ValueError("Volatility periods must be positive integers.")
    if len(set(periods)) != len(periods):
        raise ValueError("Volatility periods must be unique.")
    if type(atr_period) is not int or atr_period <= 0:
        raise ValueError("atr_period must be a positive integer.")
    return periods


def calculate_market_features(
    candles: Sequence[Candle],
    *,
    volatility_periods: Sequence[int] = DEFAULT_VOLATILITY_PERIODS,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> tuple[CandleFeatures, ...]:
    """Calculate per-candle features using only data available at that candle.

    Volatility is the unannualized root sum of squared fractional
    close-to-close returns over the configured number of returns. It remains
    ``None`` until a full window is available. The first candle has no
    close-to-close change or return.

    True range is high-low for the first candle and thereafter the maximum of
    high-low, high minus the previous close, and previous close minus low. ATR
    is unavailable until ``atr_period`` true ranges exist; it initializes to
    their arithmetic mean, then uses Wilder's recursive smoothing. No values
    are backfilled, and empty input returns an empty tuple.

    Inputs must be valid candles in strictly ascending timestamp order.
    Percentage returns are fractional (0.01 means 1%). The existing Candle
    validator rejects nonpositive/nonfinite closes; the calculation also
    guards against an unavailable return if a denominator is invalid.
    """

    periods = _validate_periods(volatility_periods, atr_period)
    validate_candle_sequence(candles)

    features: list[CandleFeatures] = []
    close_returns: list[float | None] = []
    true_ranges: list[float] = []
    previous_close: float | None = None
    current_atr: float | None = None

    for candle in candles:
        candle_range = candle.high - candle.low
        body_size = abs(candle.close - candle.open)
        close_change: float | None = None
        percentage_return: float | None = None

        if previous_close is not None:
            close_change = candle.close - previous_close
            if previous_close != 0 and isfinite(previous_close):
                calculated_return = close_change / previous_close
                if isfinite(calculated_return):
                    percentage_return = calculated_return

        if previous_close is not None:
            close_returns.append(percentage_return)

        volatility_values: list[tuple[int, float | None]] = []
        for period in periods:
            value = None
            if len(close_returns) >= period:
                window = close_returns[-period:]
                valid_returns = [item for item in window if item is not None]
                if len(valid_returns) == period:
                    value = sqrt(sum(return_value**2 for return_value in valid_returns))
            volatility_values.append((period, value))

        if previous_close is None:
            true_range = candle_range
        else:
            true_range = max(
                candle_range,
                abs(candle.high - previous_close),
                abs(previous_close - candle.low),
            )
        true_ranges.append(true_range)

        if len(true_ranges) == atr_period:
            current_atr = fmean(true_ranges)
        elif len(true_ranges) > atr_period and current_atr is not None:
            current_atr = (
                current_atr * (atr_period - 1) + true_range
            ) / atr_period

        if candle.close > candle.open:
            direction: CandleDirection = "bullish"
        elif candle.close < candle.open:
            direction = "bearish"
        else:
            direction = "neutral"

        features.append(
            CandleFeatures(
                timestamp=candle.timestamp,
                absolute_range=candle_range,
                absolute_body_size=body_size,
                upper_wick_size=candle.high - max(candle.open, candle.close),
                lower_wick_size=min(candle.open, candle.close) - candle.low,
                body_to_range_ratio=(
                    body_size / candle_range if candle_range != 0 else None
                ),
                direction=direction,
                close_to_close_change=close_change,
                close_to_close_percentage_return=percentage_return,
                rolling_realized_volatility=tuple(volatility_values),
                average_true_range=current_atr,
            )
        )
        previous_close = candle.close

    return tuple(features)
