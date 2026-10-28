"""Lightweight research coordinator: run specialist agents on one validated dataset.

Every agent receives the same immutable ``ResearchContext`` (validated candles
plus the causal predictors and outcome labels computed once) and returns an
``AgentResult``. The coordinator runs agents in the order given, so output is
deterministic, and converts an agent's exception into an ``error`` result so one
failing specialist never hides the others. Nothing here connects to a broker,
reads an account, places orders or writes to historical data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, Sequence

from core.market_data import Candle, validate_candle_sequence
from data.historical_data_loader import load_candles_csv
from research.backtest import CostModel
from research.candidates import CandidateGrid
from research.liquidity_structure import LiquidityStudyConfig
from research.price_expansion import ExpansionConfig, ExpansionResearch, FlagConfig, analyze_expansion
from research.regimes import RegimeStudyConfig
from research.walk_forward import WalkForwardConfig


@dataclass(frozen=True)
class RegimeConfig:
    """Reporting buckets from the causal short/long realized-volatility ratio.

    Realized volatility here is a sum over the window, so with periods 5 and 20
    an unchanged volatility gives a ratio near sqrt(5/20) = 0.5 (observed median
    0.46 on the 2025-26 data). The bounds are rounded values near that
    distribution's quartiles. They only slice results for reporting; they are
    not a validated regime model.
    """

    contracting_below: float = 0.4
    expanding_above: float = 0.65

    def __post_init__(self) -> None:
        if not 0 < self.contracting_below < self.expanding_above:
            raise ValueError("Require 0 < contracting_below < expanding_above.")

    def classify(self, volatility_ratio: float | None) -> str:
        if volatility_ratio is None:
            return "unavailable"
        if volatility_ratio < self.contracting_below:
            return "contracting"
        if volatility_ratio > self.expanding_above:
            return "expanding"
        return "normal"


@dataclass(frozen=True)
class SuiteConfig:
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    flags: FlagConfig = field(default_factory=FlagConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    grid: CandidateGrid = field(default_factory=CandidateGrid)
    walk_forward: WalkForwardConfig = field(
        default_factory=lambda: WalkForwardConfig(
            train_span=timedelta(days=90),
            test_span=timedelta(days=30),
            step=timedelta(days=30),
            embargo=timedelta(hours=2),
            min_train_trades=30,
        )
    )
    # The CSVs carry no spread data: these are placeholders, not measured costs.
    costs: CostModel = field(default_factory=lambda: CostModel(spread=0.30, slippage_per_side=0.05))
    min_regime_trades: int = 20
    focus_atr_multiple: float = 2.0
    spec_output_dir: str | None = None
    regime_study: RegimeStudyConfig = field(default_factory=RegimeStudyConfig)
    liquidity_study: LiquidityStudyConfig = field(default_factory=LiquidityStudyConfig)


@dataclass(frozen=True)
class ResearchContext:
    source: str
    candles: tuple[Candle, ...]
    expansion: ExpansionResearch
    config: SuiteConfig


@dataclass(frozen=True)
class AgentResult:
    agent: str
    status: str  # "ok" or "error"
    headline: str
    lines: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    data: dict = field(default_factory=dict)
    error: str | None = None


class ResearchAgent(Protocol):
    name: str

    def run(self, context: ResearchContext) -> AgentResult: ...


@dataclass(frozen=True)
class CoordinatorReport:
    source: str
    candle_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    results: tuple[AgentResult, ...]

    @property
    def all_ok(self) -> bool:
        return all(r.status == "ok" for r in self.results)

    def to_text(self) -> str:
        out = [
            "APEX offline research suite (research only; no trading, no broker connection)",
            f"source: {self.source}",
            f"candles: {self.candle_count}  range: {self.first_timestamp.isoformat()} -> "
            f"{self.last_timestamp.isoformat()}",
            "",
        ]
        for result in self.results:
            out.append(f"== {result.agent} [{result.status}] ==")
            out.append(result.headline)
            if result.error:
                out.append(f"ERROR: {result.error}")
            out.extend(result.lines)
            if result.limitations:
                out.append("limitations:")
                out.extend(f"  - {item}" for item in result.limitations)
            out.append("")
        return "\n".join(out)


def build_context(candles: Sequence[Candle], config: SuiteConfig | None = None, source: str = "<memory>") -> ResearchContext:
    """Validate candles and compute shared predictors/labels once."""

    config = config or SuiteConfig()
    if not isinstance(config, SuiteConfig):
        raise ValueError("config must be a SuiteConfig.")
    validate_candle_sequence(candles)
    if not candles:
        raise ValueError("Cannot run research on an empty dataset.")
    return ResearchContext(source, tuple(candles), analyze_expansion(candles, config.expansion), config)


def load_context(path: str | Path, config: SuiteConfig | None = None, limit_candles: int | None = None) -> ResearchContext:
    """Read-only: loads and validates a CSV through the existing loader."""

    candles = load_candles_csv(path)
    source = str(path)
    if limit_candles is not None:
        if type(limit_candles) is not int or limit_candles < 1:
            raise ValueError("limit_candles must be a positive integer.")
        candles = candles[:limit_candles]
        source += f" (first {len(candles)} candles only)"
    return build_context(candles, config, source)


def run_agents(context: ResearchContext, agents: Sequence[ResearchAgent]) -> CoordinatorReport:
    names = [agent.name for agent in agents]
    if not agents or len(set(names)) != len(names):
        raise ValueError("agents must be non-empty with unique names.")
    results: list[AgentResult] = []
    for agent in agents:
        try:
            result = agent.run(context)
            if not isinstance(result, AgentResult):
                raise TypeError("agent did not return an AgentResult")
        except Exception as exc:  # one failing specialist must not hide the others
            result = AgentResult(
                agent=agent.name, status="error", headline="Agent failed; no result produced.",
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
    candles = context.candles
    return CoordinatorReport(
        context.source, len(candles), candles[0].timestamp, candles[-1].timestamp, tuple(results)
    )
