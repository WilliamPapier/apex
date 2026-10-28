"""Offline research runner: CSV -> candles -> features -> structure -> sweeps.

This is a diagnostic pipeline, not a trading strategy. ``ResearchReport``
keeps raw observations (counts, timestamps, gap facts) separate from
``warnings`` and ``pending_strategy_decisions``, which record configuration
choices and APEX rules that are still unformalized. Nothing here selects
sessions, timezones, key levels, MSS/BMS/BOS/CHOCH confirmation, IFVG/FVG
patterns, entries, stops or targets.

The runner only reads the given CSV, never connects to MT5, and writes no
files. Counts by date use the UTC date of the event's candle-open timestamp.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

from core.market_data import Candle
from data.dataset_audit import DatasetAudit, audit_candles
from data.historical_data_loader import load_candles_csv
from research.liquidity_sweeps import (
    LiquiditySweep,
    SweepConfig,
    detect_liquidity_sweeps,
)
from research.market_features import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_VOLATILITY_PERIODS,
    calculate_market_features,
)
from research.market_structure import (
    MarketStructure,
    StructureConfig,
    analyze_market_structure,
)


PENDING_STRATEGY_DECISIONS = (
    "Session boundaries, timezone rules and the New York 15:00 key-level definition",
    "Timing-window interpretation",
    "MSS/BMS/BOS/CHOCH confirmation, including reference-candle selection",
    "FVG/IFVG formation and key-level retest/touch rules",
    "Sweep expiry/invalidation and which swings count as relevant liquidity",
    "Entry timing, stop-loss and take-profit rules",
)


@dataclass(frozen=True)
class PipelineConfig:
    structure: StructureConfig = field(default_factory=StructureConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    volatility_periods: tuple[int, ...] = DEFAULT_VOLATILITY_PERIODS
    atr_period: int = DEFAULT_ATR_PERIOD

    def __post_init__(self) -> None:
        if not isinstance(self.structure, StructureConfig):
            raise ValueError("structure must be a StructureConfig.")
        if not isinstance(self.sweep, SweepConfig):
            raise ValueError("sweep must be a SweepConfig.")
        object.__setattr__(self, "volatility_periods", tuple(self.volatility_periods))


@dataclass(frozen=True)
class DataSummary:
    source: str
    candle_count: int
    first_open: datetime
    last_open: datetime
    strictly_ascending_unique_utc: bool
    five_minute_intervals: int
    longer_intervals: int
    longest_interval_seconds: int
    invalid_candles: int


@dataclass(frozen=True)
class EventAvailability:
    count: int
    first_available_at: datetime | None
    last_available_at: datetime | None


@dataclass(frozen=True)
class FeatureObservations:
    candles_with_atr: int
    candles_with_full_volatility_window: int
    sweeps_with_atr: int
    median_penetration_to_atr: float | None


@dataclass(frozen=True)
class ResearchReport:
    config: PipelineConfig
    data: DataSummary
    features: FeatureObservations
    swing_highs: int
    swing_lows: int
    upward_breaks: int
    downward_breaks: int
    high_side_sweeps: int
    low_side_sweeps: int
    swept_swings: int
    swing_availability: EventAvailability
    break_availability: EventAvailability
    sweep_availability: EventAvailability
    counts_by_utc_date: tuple[tuple[date, int, int, int, int, int, int], ...]
    swings_spanning_long_interval: int
    breaks_spanning_long_interval: int
    sweeps_spanning_long_interval: int
    sample_sweeps: tuple[LiquiditySweep, ...]
    warnings: tuple[str, ...]
    pending_strategy_decisions: tuple[str, ...]


DATE_COLUMNS = (
    "swing_highs",
    "swing_lows",
    "up_breaks",
    "down_breaks",
    "high_sweeps",
    "low_sweeps",
)


def _availability(times: Sequence[datetime]) -> EventAvailability:
    if not times:
        return EventAvailability(0, None, None)
    return EventAvailability(len(times), min(times), max(times))


def _utc_date(timestamp: datetime) -> date:
    return timestamp.astimezone(timezone.utc).date()


def _warnings(config: PipelineConfig, structure: StructureConfig, sweeps: int,
              swings: int) -> tuple[str, ...]:
    notes = [
        "Candle duration is assumed from configuration "
        f"({structure.candle_duration}); it is not detected from the data.",
        "Lookbacks count candles, so a window can span a market closure or data gap.",
        "No session, timezone or trading-schedule rule is applied; UTC dates are labels only.",
    ]
    if structure.equal_policy == "inclusive":
        notes.append(
            "equal_policy='inclusive' makes each candle of an equal-price plateau its own swing."
        )
    if structure.break_basis == "wick":
        notes.append(
            "break_basis='wick' counts a wick beyond a swing as a break; 'close' needs a close."
        )
    if not config.sweep.first_sweep_only:
        notes.append(
            "first_sweep_only=False: every confirmed swing stays sweepable indefinitely "
            "(no expiry), so one swing can generate many sweep events."
        )
    if sweeps > 10 * max(swings, 1):
        notes.append(
            f"{sweeps} sweep events from {swings} swings: counts are dominated by "
            "repeat sweeps of old swings."
        )
    if config.sweep.min_penetration == 0.0:
        notes.append("min_penetration=0: any strictly positive wick penetration qualifies.")
    notes.append("Breaks and sweeps are neutral candidate events, not MSS/BMS/BOS/CHOCH or setups.")
    return tuple(notes)


def analyze_candles(
    candles: Sequence[Candle], config: PipelineConfig | None = None, source: str = "<memory>"
) -> ResearchReport:
    """Run features, structure and sweep detection over validated candles."""

    config = config or PipelineConfig()
    if not candles:
        raise ValueError("At least one candle is required.")
    structure_config = config.structure

    audit: DatasetAudit = audit_candles(candles)
    features = calculate_market_features(
        candles, volatility_periods=config.volatility_periods, atr_period=config.atr_period
    )
    structure: MarketStructure = analyze_market_structure(candles, structure_config)
    sweeps = detect_liquidity_sweeps(
        candles, structure.swing_highs, structure.swing_lows, structure_config, config.sweep
    )

    long_prefix = [0]
    for previous, current in zip(candles, candles[1:]):
        long_prefix.append(
            long_prefix[-1] + (1 if current.timestamp - previous.timestamp > structure_config.candle_duration else 0)
        )

    def crosses(start: int, end: int) -> bool:
        return long_prefix[end] - long_prefix[start] > 0

    all_swings = structure.swing_highs + structure.swing_lows
    left = structure_config.left_lookback
    ratios = []
    for sweep in sweeps:
        atr = features[sweep.sweep_index].average_true_range
        if atr:
            ratios.append(sweep.penetration / atr)

    by_date: dict[date, Counter[str]] = {}

    def tally(timestamp: datetime, column: str) -> None:
        by_date.setdefault(_utc_date(timestamp), Counter())[column] += 1

    for swing in structure.swing_highs:
        tally(swing.timestamp, "swing_highs")
    for swing in structure.swing_lows:
        tally(swing.timestamp, "swing_lows")
    for event in structure.breaks:
        tally(event.break_timestamp, "up_breaks" if event.direction == "up" else "down_breaks")
    for sweep in sweeps:
        tally(sweep.sweep_timestamp, "high_sweeps" if sweep.side == "high" else "low_sweeps")

    high_sweeps = sum(1 for s in sweeps if s.side == "high")
    intervals = [
        (b.timestamp - a.timestamp).total_seconds()
        for a, b in zip(candles, candles[1:])
    ]
    return ResearchReport(
        config=config,
        data=DataSummary(
            source=source,
            candle_count=len(candles),
            first_open=audit.first_timestamp,  # type: ignore[arg-type]
            last_open=audit.last_timestamp,  # type: ignore[arg-type]
            strictly_ascending_unique_utc=not (
                audit.duplicate_timestamps or audit.out_of_order_pairs or audit.invalid_timestamps
            ),
            five_minute_intervals=audit.five_minute_intervals,
            longer_intervals=len(audit.potential_gaps),
            longest_interval_seconds=int(max(intervals, default=0)),
            invalid_candles=len(audit.invalid_candles),
        ),
        features=FeatureObservations(
            candles_with_atr=sum(f.average_true_range is not None for f in features),
            candles_with_full_volatility_window=sum(
                all(v is not None for _, v in f.rolling_realized_volatility) for f in features
            ),
            sweeps_with_atr=len(ratios),
            median_penetration_to_atr=float(median(ratios)) if ratios else None,
        ),
        swing_highs=len(structure.swing_highs),
        swing_lows=len(structure.swing_lows),
        upward_breaks=sum(e.direction == "up" for e in structure.breaks),
        downward_breaks=sum(e.direction == "down" for e in structure.breaks),
        high_side_sweeps=high_sweeps,
        low_side_sweeps=len(sweeps) - high_sweeps,
        swept_swings=len({(s.side, s.swing.candle_index) for s in sweeps}),
        swing_availability=_availability([s.confirmed_at for s in all_swings]),
        break_availability=_availability(
            [e.available_at.astimezone(timezone.utc) for e in structure.breaks]
        ),
        sweep_availability=_availability([s.available_at for s in sweeps]),
        counts_by_utc_date=tuple(
            (day, *(counter[column] for column in DATE_COLUMNS))
            for day, counter in sorted(by_date.items())
        ),
        swings_spanning_long_interval=sum(
            crosses(max(s.candle_index - left, 0), s.confirmation_index) for s in all_swings
        ),
        breaks_spanning_long_interval=sum(
            crosses(e.swing.candle_index, e.break_index) for e in structure.breaks
        ),
        sweeps_spanning_long_interval=sum(
            crosses(s.swing.candle_index, s.sweep_index) for s in sweeps
        ),
        sample_sweeps=sweeps[:5],
        warnings=_warnings(config, structure_config, len(sweeps), len(all_swings)),
        pending_strategy_decisions=PENDING_STRATEGY_DECISIONS,
    )


def run_research(path: str | Path, config: PipelineConfig | None = None) -> ResearchReport:
    """Load a CSV (read-only; invalid data raises HistoricalDataError) and analyze it."""

    source = Path(path)
    return analyze_candles(load_candles_csv(source), config, str(source))


def _iso(value: datetime | None) -> str:
    return "n/a" if value is None else value.astimezone(timezone.utc).isoformat()


def render_report(report: ResearchReport) -> str:
    """Deterministic plain-text report: observations first, then notes."""

    c, d, f = report.config, report.data, report.features
    s = c.structure
    lines = [
        "APEX OFFLINE RESEARCH REPORT (diagnostic only; not a trading strategy)",
        "",
        "== CONFIGURATION ==",
        f"structure: left={s.left_lookback} right={s.right_lookback} equal_policy={s.equal_policy} "
        f"break_basis={s.break_basis} candle_duration={s.candle_duration}",
        f"sweeps: min_penetration={c.sweep.min_penetration} first_sweep_only={c.sweep.first_sweep_only}",
        f"features: volatility_periods={c.volatility_periods} atr_period={c.atr_period}",
        "",
        "== OBSERVATIONS: DATA ==",
        f"source: {d.source}",
        f"candles: {d.candle_count}  first_open: {_iso(d.first_open)}  last_open: {_iso(d.last_open)}",
        f"validation: passed (invalid_candles={d.invalid_candles}, "
        f"ascending_unique_utc={d.strictly_ascending_unique_utc})",
        f"intervals: exactly_5min={d.five_minute_intervals} longer_than_5min={d.longer_intervals} "
        f"longest_seconds={d.longest_interval_seconds} (observations; not proof of missing candles)",
        f"features: candles_with_atr={f.candles_with_atr} "
        f"candles_with_full_volatility_window={f.candles_with_full_volatility_window}",
        "",
        "== OBSERVATIONS: EVENTS ==",
        f"confirmed swings: highs={report.swing_highs} lows={report.swing_lows}",
        f"structural breaks: up={report.upward_breaks} down={report.downward_breaks}",
        f"candidate sweeps: high_side={report.high_side_sweeps} low_side={report.low_side_sweeps} "
        f"distinct_swept_swings={report.swept_swings}",
        f"sweep penetration/ATR at sweep candle: n={f.sweeps_with_atr} median={f.median_penetration_to_atr}",
        "availability (first..last): "
        f"swings {_iso(report.swing_availability.first_available_at)}..{_iso(report.swing_availability.last_available_at)}; "
        f"breaks {_iso(report.break_availability.first_available_at)}..{_iso(report.break_availability.last_available_at)}; "
        f"sweeps {_iso(report.sweep_availability.first_available_at)}..{_iso(report.sweep_availability.last_available_at)}",
        "gap-spanning events (swing window / swing-to-break / swing-to-sweep cross a >duration interval): "
        f"{report.swings_spanning_long_interval}/{report.swing_highs + report.swing_lows} swings, "
        f"{report.breaks_spanning_long_interval}/{report.upward_breaks + report.downward_breaks} breaks, "
        f"{report.sweeps_spanning_long_interval}/{report.high_side_sweeps + report.low_side_sweeps} sweeps",
        "",
        "counts by UTC date (event candle open): date " + " ".join(DATE_COLUMNS),
    ]
    lines += [f"{row[0].isoformat()} " + " ".join(str(v) for v in row[1:]) for row in report.counts_by_utc_date]
    lines += ["", "sample sweeps (first 5):"]
    for e in report.sample_sweeps:
        lines.append(
            f"  {e.side} level={e.level} extreme={e.sweep_extreme} close={e.close} "
            f"penetration={e.penetration:.6g} swing_open={_iso(e.swing.timestamp)} "
            f"swing_confirmed={_iso(e.swing.confirmed_at)} sweep_open={_iso(e.sweep_timestamp)} "
            f"available={_iso(e.available_at)}"
        )
    lines += ["", "== INTERPRETATION NOTES (not observations) =="]
    lines += [f"warning: {w}" for w in report.warnings]
    lines += [f"unresolved: {p}" for p in report.pending_strategy_decisions]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline APEX research report (read-only).")
    parser.add_argument("csv_path")
    parser.add_argument("--left", type=int, default=2)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--equal-policy", choices=("strict", "inclusive"), default="strict")
    parser.add_argument("--break-basis", choices=("close", "wick"), default="close")
    parser.add_argument("--min-penetration", type=float, default=0.0)
    parser.add_argument("--first-sweep-only", action="store_true")
    args = parser.parse_args(argv)
    config = PipelineConfig(
        structure=StructureConfig(args.left, args.right, args.equal_policy, args.break_basis),
        sweep=SweepConfig(args.min_penetration, args.first_sweep_only),
    )
    print(render_report(run_research(args.csv_path, config)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
