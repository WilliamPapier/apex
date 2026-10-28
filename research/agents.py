"""Specialist research agents run by ``research.coordinator``. Research only.

PriceExpansionAgent
    Describes expansion events and evaluates transparent, untuned baseline
    flags against them (precision, recall, false alarms, lift, lead time),
    split into chronological halves and sliced by calendar month and by a
    reporting-only volatility regime. Predictors are the causal
    ``PredictorSnapshot`` values; the expansion labels use future candles and
    are outcomes only.

CandidateStrategyAgent
    Generates explicit candidates (rejecting incomplete ones), evaluates them
    with the existing cost-aware backtester and walk-forward validation, runs
    null-benchmark controls through the same folds, and applies the promotion
    gate. A candidate that passes is still only eligible for human review; no
    agent can approve a rule or enable trading.
"""

from collections import Counter, defaultdict
from typing import Sequence

from data.mt5_history_collector import M5_INTERVAL
from research.backtest import BacktestConfig, Trade
from research.candidates import (
    GeneratedCandidate,
    generate_candidates,
    to_experimental_rule_spec,
)
from research.coordinator import AgentResult, ResearchContext
from research.price_expansion import baseline_flags, evaluate_flag, summarize_labels
from research.rule_spec import check_promotion, evidence_from_walk_forward, save_spec
from research.walk_forward import WalkForwardResult, walk_forward_evaluate


CONSISTENT_LIFT_MIN = 1.1
SIGNIFICANCE_T = 2.0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


class PriceExpansionAgent:
    name = "price_expansion_momentum"

    def run(self, context: ResearchContext) -> AgentResult:
        config = context.config
        exp = config.expansion
        research = context.expansion
        summaries = summarize_labels(research.labels, exp)
        lines = ["Expansion event base rates (labels use FUTURE candles: outcomes only, never predictors):"]
        for s in summaries:
            minutes = int(s.horizon.total_seconds() // 60)
            lines.append(
                f"  {minutes:>2} min, {s.atr_multiple:g} ATR: labelled={s.labelled} expanded={s.expanded} "
                f"base_rate={_pct(s.base_rate)} non_overlapping={s.non_overlapping_events} "
                f"continuation={s.continuation} retracement={s.retracement} reversal={s.reversal} "
                f"| excluded: atr_unavailable={s.atr_unavailable} incomplete={s.incomplete} gap={s.gap}"
            )

        multiple = config.focus_atr_multiple if config.focus_atr_multiple in exp.atr_multiples else exp.atr_multiples[0]
        flags = baseline_flags(config.flags)
        evaluations = []
        lines.append("")
        lines.append(
            f"Baseline flags at {multiple:g} ATR, first vs second chronological half "
            "(untuned thresholds; the second half is later data, not a tuned out-of-sample test):"
        )
        for horizon in exp.horizons:
            minutes = int(horizon.total_seconds() // 60)
            for name, predicate in flags.items():
                first, second = evaluate_flag(
                    research.snapshots, research.labels, name, predicate, horizon, multiple, M5_INTERVAL
                )
                evaluations.extend((first, second))
                consistent = (
                    first.lift is not None and second.lift is not None
                    and first.lift >= CONSISTENT_LIFT_MIN and second.lift >= CONSISTENT_LIFT_MIN
                    and first.flagged >= 30 and second.flagged >= 30
                )
                lines.append(
                    f"  {minutes:>2} min {name:<20} "
                    f"precision {_pct(first.precision)}/{_pct(second.precision)}  "
                    f"recall {_pct(first.recall)}/{_pct(second.recall)}  "
                    f"lift {_num(first.lift)}/{_num(second.lift)}  "
                    f"false_alarms/h {_num(first.false_alarms_per_hour)}/{_num(second.false_alarms_per_hour)}  "
                    f"median_lead {second.median_lead_time}  "
                    f"{'lift >= 1.1 in both halves' if consistent else 'not consistent across halves'}"
                )

        horizon = exp.horizons[len(exp.horizons) // 2]
        rows = [
            l for l in research.labels
            if l.horizon == horizon and l.atr_multiple == multiple and l.expanded is not None
        ]
        snapshots = research.snapshots
        by_month: dict[str, list[bool]] = defaultdict(list)
        by_regime: dict[str, list[bool]] = defaultdict(list)
        for label in rows:
            by_month[f"{label.timestamp.year}-{label.timestamp.month:02d}"].append(label.expanded)
            by_regime[config.regimes.classify(snapshots[label.index].volatility_ratio)].append(label.expanded)
        minutes = int(horizon.total_seconds() // 60)
        lines.append("")
        lines.append(f"Base rate by calendar month ({minutes} min, {multiple:g} ATR):")
        lines.append("  " + "  ".join(f"{k}:{_pct(sum(v) / len(v))}(n={len(v)})" for k, v in sorted(by_month.items())))
        lines.append("Base rate by reporting-only volatility regime (causal short/long realized-vol ratio):")
        lines.append("  " + "  ".join(f"{k}:{_pct(sum(v) / len(v))}(n={len(v)})" for k, v in sorted(by_regime.items())))

        lines.append("Flag lift by regime (flagged count in brackets):")
        regime_of = {l.index: config.regimes.classify(snapshots[l.index].volatility_ratio) for l in rows}
        for name, predicate in flags.items():
            parts = []
            for regime in sorted(by_regime):
                group = [l for l in rows if regime_of[l.index] == regime]
                flagged = [l for l in group if predicate(snapshots[l.index])]
                base = sum(l.expanded for l in group) / len(group) if group else None
                precision = sum(l.expanded for l in flagged) / len(flagged) if flagged else None
                lift = precision / base if precision is not None and base else None
                parts.append(f"{regime}:{_num(lift)}[{len(flagged)}]")
            lines.append(f"  {name:<20} " + "  ".join(parts))

        limitations = [
            "Overlapping windows: precision/recall count every eligible candle; use the non-overlapping counts for independent events.",
            f"Only {len(by_month)} calendar month(s) of data; lift estimates are descriptive, not proof of a stable signal.",
            "Flag thresholds are fixed, untuned defaults; a lift above 1 is a lead for investigation, not an edge.",
            "Closed-candle flags are available only at the candle close (available_at), never before it.",
            "Key levels, FVG/IFVG, session and news context are not yet modelled; no session boundaries are assumed.",
            "No spread data in the CSVs; excursions are measured on the raw candle prices.",
        ]
        return AgentResult(
            agent=self.name, status="ok",
            headline=f"{len(research.labels)} labels over {len(snapshots)} candles; descriptive findings only.",
            lines=tuple(lines), limitations=tuple(limitations),
            data={"summaries": summaries, "flag_evaluations": tuple(evaluations)},
        )


def _breakdown(trades: Sequence[Trade], key) -> dict[str, tuple[int, float, float | None]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        groups[key(trade)].append(trade.net_r)
    out = {}
    for name, values in sorted(groups.items()):
        wins = sum(v for v in values if v > 0)
        losses = -sum(v for v in values if v < 0)
        out[name] = (len(values), sum(values) / len(values), wins / losses if losses else None)
    return out


def _format_breakdown(label: str, table: dict[str, tuple[int, float, float | None]]) -> str:
    cells = "  ".join(f"{k}: n={n} exp={_num(e)}R pf={_num(p)}" for k, (n, e, p) in table.items())
    return f"  {label}: {cells or 'no trades'}"


class CandidateStrategyAgent:
    name = "candidate_strategy"

    def run(self, context: ResearchContext) -> AgentResult:
        config = context.config
        generation = generate_candidates(config.grid)
        candles = context.candles
        snapshots = context.expansion.snapshots
        tradable = [g for g in generation.accepted if g.spec.source != "control"]
        controls = [g for g in generation.accepted if g.spec.source == "control"]

        lines = [
            f"Variants generated: {generation.variants_generated}  accepted: {len(generation.accepted)} "
            f"(selection pool {len(tradable)}, null controls {len(controls)})  rejected: {len(generation.rejected)}",
            "Rejected candidates (incomplete rules are never filled with assumptions):",
        ]
        for name, reasons in generation.rejected:
            lines.append(f"  {name}: {len(reasons)} issue(s), e.g. {'; '.join(reasons[:3])}")
        if not tradable:
            return AgentResult(
                agent=self.name, status="ok", headline="No runnable candidates; nothing was evaluated.",
                lines=tuple(lines), data={"generation": generation},
            )

        template = BacktestConfig(exit=tradable[0].spec.exit, costs=config.costs)
        wf = walk_forward_evaluate(candles, snapshots, [g.candidate for g in tradable], config.walk_forward, template)
        benchmark = (
            walk_forward_evaluate(candles, snapshots, [g.candidate for g in controls], config.walk_forward, template)
            if controls else None
        )

        m = wf.oos_metrics
        lines.append("")
        lines.append(
            f"Walk-forward: {len(wf.folds)} fold(s), train {config.walk_forward.train_span.days}d / "
            f"test {config.walk_forward.test_span.days}d / step {config.walk_forward.step.days}d, "
            f"embargo {config.walk_forward.embargo}, min train trades {config.walk_forward.min_train_trades}."
        )
        lines.append(
            f"Costs (placeholders): spread {config.costs.spread:g}, slippage/side {config.costs.slippage_per_side:g}, "
            f"commission/side {config.costs.commission_per_side:g}; risk 0.5%, one position."
        )
        for fr in wf.folds:
            chosen = next((mt for n, mt in fr.train_metrics if n == fr.selected), None)
            test = fr.test.metrics if fr.test else None
            lines.append(
                f"  fold {fr.fold.number} test {fr.fold.test_from.date()}..{fr.fold.test_until.date()} "
                f"selected={fr.selected or 'none'} "
                f"train_exp={_num(chosen.expectancy_r) if chosen else 'n/a'}R "
                f"test_trades={test.trades if test else 0} test_exp={_num(test.expectancy_r) if test else 'n/a'}R"
            )
        lines.append(
            f"Pooled out-of-sample: trades={m.trades} win_rate={_pct(m.win_rate)} expectancy={_num(m.expectancy_r)}R "
            f"profit_factor={_num(m.profit_factor)} avg_cost={_num(m.average_cost_r)}R "
            f"max_drawdown={_pct(m.max_drawdown_fraction)} max_losing_streak={m.max_consecutive_losses} "
            f"trades_spanning_gap={m.trades_spanning_gap}"
        )
        selected = Counter(fr.selected for fr in wf.folds if fr.selected)
        lines.append(f"Selection stability: {dict(selected) or 'no candidate selected in any fold'}")

        classify = config.regimes.classify
        by_regime = _breakdown(wf.oos_trades, lambda t: classify(snapshots[t.signal_index].volatility_ratio))
        by_month = _breakdown(wf.oos_trades, lambda t: f"{t.entry_time.year}-{t.entry_time.month:02d}")
        lines.append("Out-of-sample breakdown (net of placeholder costs):")
        lines.append(_format_breakdown("by month", by_month))
        lines.append(_format_breakdown("by regime", by_regime))

        if benchmark is not None:
            b = benchmark.oos_metrics
            lines.append(
                f"Null controls through the same folds: trades={b.trades} expectancy={_num(b.expectancy_r)}R "
                f"profit_factor={_num(b.profit_factor)} max_drawdown={_pct(b.max_drawdown_fraction)}"
            )

        regimes_tested = tuple(
            regime for regime, (count, _, _) in by_regime.items()
            if regime != "unavailable" and count >= config.min_regime_trades
        )
        limitations = [
            "Spread, slippage and commission are placeholders; the CSVs contain no spread.",
            "Entries fill at the next candle open; intrabar order is unknown (stop assumed first when both are touched).",
            "Drawdown uses closed-trade balances only; lot rounding, swaps and margin are not modelled.",
            f"{generation.variants_generated} variants were considered: expect the best pooled result to be optimistic.",
            "The owner's sweep/MSS/IFVG/key-level hypothesis is not testable until its unresolved definitions are supplied.",
            "Regimes are reporting buckets from a causal volatility ratio, not a validated regime model.",
        ]
        evidence = evidence_from_walk_forward(wf, candles, template, regimes_tested, limitations)
        failures = check_promotion(evidence)
        returns = [t.net_r for t in wf.oos_trades]
        t_stat = None
        if len(returns) > 1:
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
            t_stat = mean / (variance / len(returns)) ** 0.5 if variance > 0 else None
        positive_folds = sum(1 for fr in wf.folds if fr.test and fr.test.metrics.expectancy_r and fr.test.metrics.expectancy_r > 0)
        traded_folds = sum(1 for fr in wf.folds if fr.test and fr.test.metrics.trades)
        distinguishable = t_stat is not None and t_stat >= SIGNIFICANCE_T
        lines.append(
            f"Robustness: expectancy t-statistic {_num(t_stat)} (needs >= {SIGNIFICANCE_T:g} to be distinguishable from "
            f"zero, ignoring the {generation.variants_generated} variants tried); "
            f"{positive_folds}/{traded_folds} traded test folds had positive expectancy."
        )
        lines.append("")
        if failures:
            lines.append("Promotion gate: NOT met (candidates stay experimental):")
            lines.extend(f"  - {reason}" for reason in failures)
        elif not distinguishable:
            lines.append(
                "Promotion gate: placeholder statistical bar met, but the result is NOT distinguishable from zero "
                "after costs; treat as inconclusive (candidates stay experimental)."
            )
        else:
            lines.append("Promotion gate: statistical bar met; eligible for HUMAN review only (not approved, not live).")

        saved = self._save_specs(config.spec_output_dir, wf, tradable)
        if saved:
            lines.append(f"Experimental specs written: {', '.join(saved)}")
        if failures:
            verdict = "no evidence of a robust edge"
        elif not distinguishable:
            verdict = "gate met but not distinguishable from zero (inconclusive)"
        else:
            verdict = "passed the statistical gate; needs human review"
        return AgentResult(
            agent=self.name, status="ok",
            headline=f"{len(wf.folds)} fold(s), {m.trades} out-of-sample trades; {verdict}.",
            lines=tuple(lines), limitations=tuple(limitations),
            data={
                "generation": generation, "walk_forward": wf, "benchmark": benchmark,
                "evidence": evidence, "gate_failures": failures, "variants_generated": generation.variants_generated,
                "t_statistic": t_stat,
            },
        )

    @staticmethod
    def _save_specs(directory: str | None, wf: WalkForwardResult, tradable: list[GeneratedCandidate]) -> list[str]:
        if not directory:
            return []
        by_name = {g.spec.name: g.spec for g in tradable}
        written = []
        for name in sorted({fr.selected for fr in wf.folds if fr.selected}):
            try:
                written.append(save_spec(to_experimental_rule_spec(by_name[name]), directory).name)
            except FileExistsError:
                continue
        return written
