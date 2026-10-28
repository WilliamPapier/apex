"""Explicit, deterministic strategy candidates for offline research. Never trades.

A candidate is only runnable when everything needed for a deterministic
backtest is stated: a known signal family, every signal parameter, and an
exit rule (stop, target, maximum holding time). Anything not stated is listed
in ``unresolved`` and the candidate is REJECTED with a reason. Nothing is
guessed or defaulted on the owner's behalf.

Families
    momentum_continuation      independent baseline: trade with recent momentum
    mean_reversion_fade        independent baseline: fade an unusually large candle
    sweep_reversal_baseline    simple, generic sweep reaction (NOT the owner's setup)
    control_always_long/short  null benchmarks: a rule that beats these is not
                               merely collecting the market's drift or costs
    user_sweep_mss_ifvg        the owner's hypothesis, kept as a spec with its
                               unresolved definitions; it is rejected until they
                               are formalised and a signal builder exists

All signals are pure functions of one causal ``PredictorSnapshot`` (candle i has
closed), so they cannot see later candles. ``variants_generated`` counts every
candidate considered, accepted or not, so selection bias stays visible.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable

from research.backtest import ExitRule, SignalRule
from research.price_expansion import PredictorSnapshot
from research.rule_spec import RuleSpec
from research.walk_forward import RuleCandidate


Param = float | int | str | None
SOURCES = ("user_hypothesis", "data_discovered", "control")

USER_HYPOTHESIS_UNRESOLVED = (
    "MSS/BMS/BOS/CHOCH confirmation rule",
    "original FVG to IFVG transition definition",
    "new FVG formation definition",
    "key level definition and timezone/session boundaries (e.g. 15:00 New York open)",
    "wick/body touch and retest rule",
    "sweep expiry and invalidation rule",
    "entry timing",
    "stop-loss rule",
    "target rule",
    "trade-management rule",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    hypothesis: str
    source: str
    params: tuple[tuple[str, Param], ...] = ()
    exit: ExitRule | None = None
    unresolved: tuple[str, ...] = ()

    def param(self, key: str) -> Param:
        return dict(self.params).get(key)


def _side_from_momentum(spec: CandidateSpec) -> SignalRule:
    threshold = float(spec.param("min_momentum_atr"))
    volume = float(spec.param("min_tick_volume_ratio"))

    def signal(s: PredictorSnapshot):
        if s.momentum_to_atr is None:
            return None
        if volume > 0 and (s.tick_volume_ratio is None or s.tick_volume_ratio < volume):
            return None
        if s.momentum_to_atr >= threshold:
            return "long"
        if s.momentum_to_atr <= -threshold:
            return "short"
        return None

    return signal


def _side_from_fade(spec: CandidateSpec) -> SignalRule:
    threshold = float(spec.param("min_range_atr"))
    volume = float(spec.param("min_tick_volume_ratio"))

    def signal(s: PredictorSnapshot):
        if s.range_to_atr is None or s.range_to_atr < threshold:
            return None
        if volume > 0 and (s.tick_volume_ratio is None or s.tick_volume_ratio < volume):
            return None
        return {"bullish": "short", "bearish": "long"}.get(s.direction)

    return signal


def _side_from_sweep(spec: CandidateSpec) -> SignalRule:
    confirm = int(spec.param("require_confirming_candle"))

    def signal(s: PredictorSnapshot):
        # A swept low is a candidate long reaction; a swept high a candidate short.
        if s.recent_low_sweeps > 0 and s.recent_high_sweeps == 0:
            return "long" if not confirm or s.direction == "bullish" else None
        if s.recent_high_sweeps > 0 and s.recent_low_sweeps == 0:
            return "short" if not confirm or s.direction == "bearish" else None
        return None

    return signal


def _always(side: str) -> Callable[[CandidateSpec], SignalRule]:
    return lambda spec: (lambda s: side)


SIGNAL_BUILDERS: dict[str, tuple[tuple[str, ...], Callable[[CandidateSpec], SignalRule]]] = {
    "momentum_continuation": (("min_momentum_atr", "min_tick_volume_ratio"), _side_from_momentum),
    "mean_reversion_fade": (("min_range_atr", "min_tick_volume_ratio"), _side_from_fade),
    "sweep_reversal_baseline": (("require_confirming_candle",), _side_from_sweep),
    "control_always_long": ((), _always("long")),
    "control_always_short": ((), _always("short")),
}


def rejection_reasons(spec: CandidateSpec) -> tuple[str, ...]:
    """Every reason a candidate cannot be backtested deterministically; empty means runnable."""

    reasons: list[str] = []
    if spec.source not in SOURCES:
        reasons.append(f"unknown source {spec.source!r}")
    reasons.extend(f"unresolved: {item}" for item in spec.unresolved)
    if spec.exit is None:
        reasons.append("no exit rule (stop, target, maximum holding time) defined")
    elif not isinstance(spec.exit, ExitRule):
        reasons.append("exit is not an ExitRule")
    if spec.family not in SIGNAL_BUILDERS:
        reasons.append(f"no signal builder implemented for family {spec.family!r}")
    else:
        values = dict(spec.params)
        for key in SIGNAL_BUILDERS[spec.family][0]:
            value = values.get(key)
            if value is None:
                reasons.append(f"missing signal parameter {key!r}")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                reasons.append(f"signal parameter {key!r} must be a non-negative number")
    return tuple(reasons)


def build_signal(spec: CandidateSpec) -> SignalRule:
    reasons = rejection_reasons(spec)
    if reasons:
        raise ValueError(f"Candidate {spec.name!r} is incomplete: {'; '.join(reasons)}")
    return SIGNAL_BUILDERS[spec.family][1](spec)


@dataclass(frozen=True)
class CandidateGrid:
    """Explicit search space. Every combination counts as a tested variant."""

    momentum_atr: tuple[float, ...] = (0.5, 1.0, 1.5)
    fade_range_atr: tuple[float, ...] = (1.5, 2.5)
    volume_ratios: tuple[float, ...] = (0.0, 1.5)
    stop_atr_multiples: tuple[float, ...] = (1.0, 2.0)
    target_rs: tuple[float, ...] = (1.0, 2.0)
    max_hold_candles: tuple[int, ...] = (12,)
    include_controls: bool = True
    include_sweep_baseline: bool = True
    include_user_hypothesis: bool = True

    def __post_init__(self) -> None:
        groups = (
            self.momentum_atr, self.fade_range_atr, self.volume_ratios,
            self.stop_atr_multiples, self.target_rs, self.max_hold_candles,
        )
        if any(not group for group in groups):
            raise ValueError("Every grid dimension needs at least one value.")


@dataclass(frozen=True)
class GeneratedCandidate:
    spec: CandidateSpec
    candidate: RuleCandidate


@dataclass(frozen=True)
class CandidateGeneration:
    accepted: tuple[GeneratedCandidate, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...]
    variants_generated: int


def _g(value: float) -> str:
    """Compact number for identifiers (rule ids may not contain a dot)."""
    return f"{value:g}".replace(".", "p")


def _exit_label(exit_rule: ExitRule) -> str:
    return f"s{_g(exit_rule.stop_atr_multiple)}_t{_g(exit_rule.target_r)}_h{exit_rule.max_hold_candles}"


def candidate_specs(grid: CandidateGrid) -> tuple[CandidateSpec, ...]:
    exits = [ExitRule(s, t, h) for s, t, h in product(grid.stop_atr_multiples, grid.target_rs, grid.max_hold_candles)]
    specs: list[CandidateSpec] = []
    for exit_rule, volume in product(exits, grid.volume_ratios):
        for threshold in grid.momentum_atr:
            specs.append(CandidateSpec(
                name=f"momentum_m{_g(threshold)}_v{_g(volume)}_{_exit_label(exit_rule)}",
                family="momentum_continuation", source="data_discovered",
                hypothesis="Recent directional momentum (in ATRs) continues over the holding period.",
                params=(("min_momentum_atr", threshold), ("min_tick_volume_ratio", volume)),
                exit=exit_rule,
            ))
        for threshold in grid.fade_range_atr:
            specs.append(CandidateSpec(
                name=f"fade_r{_g(threshold)}_v{_g(volume)}_{_exit_label(exit_rule)}",
                family="mean_reversion_fade", source="data_discovered",
                hypothesis="An unusually large candle is partly retraced; trade against its direction.",
                params=(("min_range_atr", threshold), ("min_tick_volume_ratio", volume)),
                exit=exit_rule,
            ))
    if grid.include_sweep_baseline:
        for exit_rule, confirm in product(exits, (0, 1)):
            specs.append(CandidateSpec(
                name=f"sweep_reversal_c{confirm}_{_exit_label(exit_rule)}",
                family="sweep_reversal_baseline", source="data_discovered",
                hypothesis="Price reverses after a candidate liquidity sweep (generic baseline, not the owner's setup).",
                params=(("require_confirming_candle", confirm),),
                exit=exit_rule,
            ))
    if grid.include_controls:
        for exit_rule in exits:
            for family in ("control_always_long", "control_always_short"):
                specs.append(CandidateSpec(
                    name=f"{family}_{_exit_label(exit_rule)}", family=family, source="control",
                    hypothesis="Null benchmark: enter every eligible candle; isolates drift and costs from signal value.",
                    exit=exit_rule,
                ))
    if grid.include_user_hypothesis:
        specs.append(CandidateSpec(
            name="user_sweep_mss_ifvg_keylevel",
            family="user_sweep_mss_ifvg", source="user_hypothesis",
            hypothesis=(
                "Liquidity sweep, then MSS/BMS/BOS/CHOCH, then FVG becomes IFVG, then a new FVG, "
                "then a key-level retest, then potential expansion."
            ),
            unresolved=USER_HYPOTHESIS_UNRESOLVED,
        ))
    return tuple(specs)


def generate_candidates(grid: CandidateGrid | None = None) -> CandidateGeneration:
    grid = grid or CandidateGrid()
    specs = candidate_specs(grid)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique.")
    accepted: list[GeneratedCandidate] = []
    rejected: list[tuple[str, tuple[str, ...]]] = []
    for spec in specs:
        reasons = rejection_reasons(spec)
        if reasons:
            rejected.append((spec.name, reasons))
        else:
            accepted.append(GeneratedCandidate(spec, RuleCandidate(spec.name, build_signal(spec), spec.exit)))
    return CandidateGeneration(tuple(accepted), tuple(rejected), len(specs))


def to_experimental_rule_spec(spec: CandidateSpec, version: int = 1) -> RuleSpec:
    """An experimental (never approved, never live) record of a runnable candidate."""

    reasons = rejection_reasons(spec)
    if reasons:
        raise ValueError(f"Candidate {spec.name!r} is incomplete: {'; '.join(reasons)}")
    exit_rule = spec.exit
    params = [(k, v) for k, v in spec.params if v is not None]
    params += [
        ("stop_atr_multiple", exit_rule.stop_atr_multiple),
        ("target_r", exit_rule.target_r),
        ("max_hold_candles", exit_rule.max_hold_candles),
    ]
    return RuleSpec(
        rule_id=spec.name, version=version, status="experimental",
        source="user_hypothesis" if spec.source == "user_hypothesis" else "data_discovered",
        hypothesis=spec.hypothesis,
        rationale="Generated by the offline candidate grid; awaiting walk-forward evidence.",
        entry_conditions=(
            f"family={spec.family}",
            "decision at candle close; entry at next candle open if it follows directly",
        ),
        exit_conditions=(
            f"stop {exit_rule.stop_atr_multiple:g} x ATR; target {exit_rule.target_r:g} R; "
            f"time exit after {exit_rule.max_hold_candles} candles",
        ),
        parameters=tuple(sorted(params)),
    )
