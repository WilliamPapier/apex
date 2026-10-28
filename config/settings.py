
"""Central configuration for the APEX trading system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApexSettings:
    # Application
    app_name: str = "APEX"
    environment: str = "development"

    # Broker and instrument
    broker_name: str = "Exness"
    symbol: str = "XAUUSDm"
    timeframe: str = "M5"

    # Risk policy
    risk_per_trade: float = 0.005  # 0.5%
    max_open_positions: int = 1

    # Safety: live execution stays disabled during development
    live_trading_enabled: bool = False


SETTINGS = ApexSettings()
