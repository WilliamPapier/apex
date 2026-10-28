"""Versioned, immutable rule specifications with evidence and promotion gates.

A ``RuleSpec`` records exactly what a candidate rule is, why it was proposed and
what evidence supports it. Status is a ladder:

    experimental  any rule, including the user's screenshot-based hypotheses
    candidate     the evidence record passes the promotion gate (statistical bar only)
    approved      candidate plus a named human approver

Nothing here can enable trading: ``live_trading_allowed`` is always False.
Even an approved rule must still pass a future approved-strategy review, the risk
manager and the execution layer. The default gate thresholds are research policy
placeholders awaiting the owner's sign-off, not claims of sufficiency.

Specs are written exclusively (never overwritten) as ``<rule_id>_v<version>.json``
into a caller-chosen directory that must not be a historical-data directory.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Sequence

from config.settings import SETTINGS
from core.market_data import Candle
from research.backtest import BacktestConfig
from research.walk_forward import WalkForwardResult


STATUSES = ("experimental", "candidate", "approved")
SOURCES = ("user_hypothesis", "data_discovered")


@dataclass(frozen=True)
class EvidenceRecord:
    data_start: datetime
    data_end: datetime
    distinct_months: int
    folds: int
    folds_with_selection: int
    oos_trades: int
    candidates_evaluated: int
    oos_expectancy_r: float | None
    oos_profit_factor: float | None
    oos_max_drawdown_fraction: float
    spread: float
    slippage_per_side: float
    commission_per_side: float
    regimes_tested: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionGate:
    min_months: int = 6
    min_oos_trades: int = 100
    min_folds_with_selection: int = 3
    min_oos_profit_factor: float = 1.0
    max_drawdown_fraction: float = 0.2
    min_regimes: int = 2


def evidence_from_walk_forward(
    result: WalkForwardResult,
    candles: Sequence[Candle],
    config: BacktestConfig,
    regimes_tested: Sequence[str] = (),
    limitations: Sequence[str] = (),
) -> EvidenceRecord:
    if not candles:
        raise ValueError("candles must not be empty.")
    months = {(c.timestamp.year, c.timestamp.month) for c in candles}
    metrics = result.oos_metrics
    return EvidenceRecord(
        data_start=candles[0].timestamp,
        data_end=candles[-1].timestamp,
        distinct_months=len(months),
        folds=len(result.folds),
        folds_with_selection=result.folds_with_selection,
        oos_trades=metrics.trades,
        candidates_evaluated=result.candidates_evaluated,
        oos_expectancy_r=metrics.expectancy_r,
        oos_profit_factor=metrics.profit_factor,
        oos_max_drawdown_fraction=metrics.max_drawdown_fraction,
        spread=config.costs.spread,
        slippage_per_side=config.costs.slippage_per_side,
        commission_per_side=config.costs.commission_per_side,
        regimes_tested=tuple(regimes_tested),
        limitations=tuple(limitations),
    )


def check_promotion(evidence: EvidenceRecord, gate: PromotionGate | None = None) -> tuple[str, ...]:
    """Every reason the evidence is insufficient; an empty tuple means the gate passes."""

    gate = gate or PromotionGate()
    failures = []
    if evidence.distinct_months < gate.min_months:
        failures.append(f"only {evidence.distinct_months} calendar month(s); need {gate.min_months}")
    if evidence.folds_with_selection < gate.min_folds_with_selection:
        failures.append(
            f"only {evidence.folds_with_selection} walk-forward fold(s) with a selected rule; "
            f"need {gate.min_folds_with_selection}"
        )
    if evidence.oos_trades < gate.min_oos_trades:
        failures.append(f"only {evidence.oos_trades} out-of-sample trades; need {gate.min_oos_trades}")
    if evidence.candidates_evaluated < 1:
        failures.append("number of candidates evaluated is not recorded")
    if evidence.spread <= 0:
        failures.append("no spread was modelled")
    if evidence.oos_expectancy_r is None or evidence.oos_expectancy_r <= 0:
        failures.append("out-of-sample expectancy after costs is not positive")
    if evidence.oos_profit_factor is None or evidence.oos_profit_factor < gate.min_oos_profit_factor:
        failures.append("out-of-sample profit factor is missing or below the gate")
    if evidence.oos_max_drawdown_fraction > gate.max_drawdown_fraction:
        failures.append("out-of-sample drawdown exceeds the gate")
    if len(evidence.regimes_tested) < gate.min_regimes:
        failures.append(f"fewer than {gate.min_regimes} market regimes evaluated")
    return tuple(failures)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    version: int
    status: str
    source: str
    hypothesis: str
    rationale: str
    entry_conditions: tuple[str, ...]
    exit_conditions: tuple[str, ...]
    parameters: tuple[tuple[str, float | int | str], ...]
    instrument: str = SETTINGS.symbol
    timeframe: str = SETTINGS.timeframe
    risk_fraction: float = SETTINGS.risk_per_trade
    max_open_positions: int = SETTINGS.max_open_positions
    evidence: EvidenceRecord | None = None
    approved_by: str = ""
    gate: PromotionGate = field(default_factory=PromotionGate)

    def __post_init__(self) -> None:
        if not self.rule_id or not self.rule_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("rule_id must be a non-empty alphanumeric identifier.")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be an integer of at least 1.")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}.")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}.")
        if not self.hypothesis.strip() or not self.entry_conditions or not self.exit_conditions:
            raise ValueError("hypothesis, entry_conditions and exit_conditions are required.")
        if not 0 < self.risk_fraction <= SETTINGS.risk_per_trade or self.max_open_positions != 1:
            raise ValueError("Risk must not exceed the configured limit and one position is allowed.")
        if self.status != "experimental":
            if self.evidence is None:
                raise ValueError(f"A {self.status} rule requires an evidence record.")
            failures = check_promotion(self.evidence, self.gate)
            if failures:
                raise ValueError(f"Promotion gate not met: {'; '.join(failures)}")
        if self.status == "approved" and not self.approved_by.strip():
            raise ValueError("An approved rule requires a named human approver.")
        if self.status != "approved" and self.approved_by:
            raise ValueError("Only an approved rule may name an approver.")

    @property
    def live_trading_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=_encode, sort_keys=True))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def content_hash(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


def _encode(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    raise TypeError(f"Cannot serialise {type(value).__name__}.")


def save_spec(spec: RuleSpec, directory: str | Path) -> Path:
    """Write ``<rule_id>_v<version>.json`` exclusively; an existing version is never overwritten."""

    folder = Path(directory)
    if "historical" in (part.lower() for part in folder.resolve().parts):
        raise ValueError("Specifications must not be written into a historical data directory.")
    if not folder.is_dir():
        raise ValueError("directory must be an existing folder.")
    path = folder / f"{spec.rule_id}_v{spec.version}.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(spec.to_json() + "\n")
    return path
