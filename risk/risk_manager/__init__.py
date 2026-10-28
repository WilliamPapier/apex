
"""APEX risk calculations. This module does not place trades."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeRisk:
    account_balance: float
    risk_fraction: float
    entry_price: float
    stop_loss: float
    value_per_price_unit_per_lot: float


@dataclass(frozen=True)
class RiskResult:
    approved: bool
    risk_amount: float
    stop_distance: float
    position_size_lots: float
    reason: str


def calculate_position_risk(trade: TradeRisk) -> RiskResult:
    """Calculate theoretical position size from account risk and stop distance."""

    values = (
        trade.account_balance,
        trade.risk_fraction,
        trade.entry_price,
        trade.stop_loss,
        trade.value_per_price_unit_per_lot,
    )

    if not all(isinstance(value, (int, float)) for value in values):
        return RiskResult(False, 0, 0, 0, "Inputs must be numeric.")

    if not all(__import__("math").isfinite(value) for value in values):
        return RiskResult(False, 0, 0, 0, "Inputs must be finite.")

    if trade.account_balance <= 0:
        return RiskResult(False, 0, 0, 0, "Balance must be positive.")

    if not 0 < trade.risk_fraction <= 0.005:
        return RiskResult(False, 0, 0, 0, "Risk must be greater than 0 and no more than 0.5%.")

    if trade.entry_price <= 0 or trade.stop_loss <= 0:
        return RiskResult(False, 0, 0, 0, "Entry and stop prices must be positive.")

    if trade.entry_price == trade.stop_loss:
        return RiskResult(False, 0, 0, 0, "Stop loss must differ from entry.")

    if trade.value_per_price_unit_per_lot <= 0:
        return RiskResult(False, 0, 0, 0, "Instrument value must be positive.")

    risk_amount = trade.account_balance * trade.risk_fraction
    stop_distance = abs(trade.entry_price - trade.stop_loss)

    lots = risk_amount / (
        stop_distance * trade.value_per_price_unit_per_lot
    )

    return RiskResult(
        approved=True,
        risk_amount=risk_amount,
        stop_distance=stop_distance,
        position_size_lots=lots,
        reason="Calculation passed. Broker volume rules and costs still need checking.",
    )
