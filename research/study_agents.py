"""Market-regime and liquidity/structure research agents. Research only.

Both agents are thin: all measurement lives in ``research.regimes`` and
``research.liquidity_structure`` (pure, tested functions). The agents run those
studies on the shared ``ResearchContext`` and format a compact report. Neither
can approve a rule, place an order or connect to a broker, and a "justifies
further research" verdict only means the measured separation persisted out of
sample under the stated criteria; it is not evidence of profitability.
"""

from datetime import timedelta

from research.coordinator import AgentResult, ResearchContext
from research.liquidity_structure import DetailTable, HorizonStat, run_liquidity_study
from research.regimes import STATES, VariantDescription, run_regime_study


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}pp"


def _z(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _minutes(delta: timedelta) -> int:
    return int(delta.total_seconds() // 60)


def _focus(context: ResearchContext) -> tuple[timedelta, float]:
    exp = context.config.expansion
    multiple = context.config.focus_atr_multiple
    if multiple not in exp.atr_multiples:
        multiple = exp.atr_multiples[0]
    return exp.horizons[len(exp.horizons) // 2], multiple


class MarketRegimeAgent:
    name = "market_regime"

    def run(self, context: ResearchContext) -> AgentResult:
        config = context.config
        horizon, multiple = _focus(context)
        study = config.regime_study
        result = run_regime_study(
            context.candles, context.expansion, config.expansion, study, config.walk_forward,
            horizon, multiple, config.flags.large_candle_atr_min,
        )
        minutes = _minutes(horizon)
        excluded = ", ".join(f"{k}={v}" for k, v in result.excluded_by_status) or "none"
        lines = [
            f"Outcome = expansion of {multiple:g} ATR within {minutes} min (FUTURE label, evaluation only). "
            "Regime states use only candles up to each close: percentile of the feature among strictly earlier values "
            f"(min history {study.min_history} values; no full-sample thresholds).",
            f"Rows: candles={result.total_candles} labelled={result.labelled_rows} excluded by label status: {excluded}. "
            f"Variants tried: {result.variants_tried} (+1 fixed baseline flag).",
            "Variants, full sample, DESCRIPTIVE (low/mid/high): frequency | median run (candles) | expansion rate | "
            "unclassified rows | state changes per 100:",
        ]
        for d in result.descriptions:
            lines.append("  " + self._variant_line(d))
        lines.append("  " + self._variant_line(result.baseline) + "  (fixed flag: high = large closed candle)")

        lines.append(
            f"Chronological walk-forward ({len(result.folds)} fold(s); variant chosen on TRAIN rows only; hot = state "
            "with the higher TRAIN expansion rate):"
        )
        for f in result.folds:
            if f.selected is None:
                lines.append(f"  fold {f.number}: no variant had >= {study.min_state_rows} rows in both extreme states")
                continue
            lines.append(
                f"  fold {f.number} test from {f.test_from.date()}: {f.selected} train spread {_pp(f.train_spread)} -> "
                f"test spread {_pp(f.test_spread)} (hot n={f.test_hot.n}, cold n={f.test_cold.n})"
            )
        p, b = result.pooled, result.baseline_pooled
        lines.append(
            f"Pooled out-of-sample: unconditional rate {_pct(result.base_rate)} (n={result.unconditional_oos.n}); "
            f"selected hot {_pct(p.hot.expansion_rate)} vs cold {_pct(p.cold.expansion_rate)} spread {_pp(p.spread)} "
            f"(thinned n={p.hot_thin.n}+{p.cold_thin.n}, z {_z(p.z)}); baseline flag {_pct(b.hot.expansion_rate)} vs "
            f"{_pct(b.cold.expansion_rate)} spread {_pp(b.spread)} (z {_z(b.z)})."
        )
        if result.selection_counts:
            lines.append(
                "Selection stability: "
                + ", ".join(f"{name} x{count}" for name, count in result.selection_counts)
                + f" ({len(result.selection_counts)} distinct variant(s) over {len(result.folds)} folds)"
            )
        if result.most_selected:
            lines.extend(self._detail_lines(result.most_selected, result.descriptions))
        lines.append("Criteria (all must pass to justify further regime research):")
        for c in result.criteria:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        verdict = (
            "Regime labelling passed the out-of-sample criteria; worth further (still offline) experiments."
            if result.justifies_further_research
            else "Regime labelling did NOT pass every criterion; it is not yet justified as a basis for strategy research."
        )
        lines.append(verdict)
        limitations = [
            "Single instrument (XAUUSDm), single timeframe (M5), roughly one year; regimes may not generalise.",
            "No spread or execution data, so nothing here concerns tradeable outcomes.",
            "Expansion is ATR-normalised, so ATR-level regimes interact with the outcome definition itself.",
            "Neighbouring rows share forward windows; significance uses rows thinned to one per horizon, folds are few "
            "and dependent, and several variants were tried (selection is repeated per fold).",
            "States are relative to each candle's own past (expanding percentiles), not absolute market regimes; "
            "a separation of expansion rates does not show a genuine regime or a tradeable edge.",
            "Rolling windows require contiguous candles (gaps are not filled); features taken from the existing "
            "predictors inherit their gap behaviour.",
        ]
        return AgentResult(
            agent=self.name, status="ok",
            headline=(
                f"{result.variants_tried} variants, {len(result.folds)} fold(s); pooled OOS spread {_pp(p.spread)} "
                f"(z {_z(p.z)}); " + ("passes" if result.justifies_further_research else "does not pass") + " the stated criteria."
            ),
            lines=tuple(lines), limitations=tuple(limitations), data={"study": result},
        )

    @staticmethod
    def _variant_line(d: VariantDescription) -> str:
        freq = "/".join(f"{v * 100:.0f}" for _, v in d.frequency)
        runs = "/".join("n/a" if v is None else f"{v:g}" for _, v in d.median_run)
        rates = "/".join(_pct(v, 1).rstrip("%") if v is not None else "n/a" for _, v, _ in d.expansion_rate)
        change = "n/a" if d.changes_per_100 is None else f"{d.changes_per_100:.1f}"
        return f"{d.name:<30} freq {freq:<9} run {runs:<10} exp% {rates:<16} unclass {d.unclassified_rows:<6} chg/100 {change}"

    @staticmethod
    def _detail_lines(name: str, descriptions) -> list[str]:
        d = next(x for x in descriptions if x.name == name)
        lines = [f"Most-selected variant {name} (descriptive, full sample):"]
        for state, rate, n in d.expansion_rate:
            reversal = dict(d.reversal_rate)[state]
            cont = dict(d.continuation_rate)[state]
            excursion = dict(d.mean_excursion_atr)[state]
            lines.append(
                f"  {state:<4} n={n:<6} expansion {_pct(rate)}  reversal-among-expansions {_pct(reversal)}  "
                f"prior-momentum continued {_pct(cont)}  mean excursion {'n/a' if excursion is None else f'{excursion:.2f}'} ATR"
            )
        months = "  ".join(
            f"{month}:" + "/".join(_pct(rate, 0).rstrip("%") if rate is not None else "-" for _, rate, _ in cells)
            for month, cells in d.monthly
        )
        lines.append(f"  expansion % by month (low/mid/high): {months}")
        return lines


class LiquidityStructureAgent:
    name = "liquidity_structure"

    def run(self, context: ResearchContext) -> AgentResult:
        config = context.config
        horizon, multiple = _focus(context)
        study = config.liquidity_study
        regimes = [config.regimes.classify(s.volatility_ratio) for s in context.expansion.snapshots]
        result = run_liquidity_study(
            context.candles, context.expansion, config.expansion, study, config.walk_forward,
            horizon, multiple, regimes,
        )
        minutes = _minutes(horizon)
        ref = study.reference
        summaries = {s.name: s for s in result.summaries}
        reference = summaries.get(ref.name)
        lines = [
            f"Outcome = expansion of {multiple:g} ATR; 'aligned' = expansion in the event's implied direction; "
            "'excess' = aligned minus the unconditional rate of the same direction (FUTURE labels, evaluation only). "
            f"Definitions tried: {result.variants_tried}.",
        ]
        if result.reference_matches_existing is None:
            lines.append("Cross-check against the existing detector: skipped (non-default sweep/structure config).")
        else:
            lines.append(
                f"Cross-check, reference definition vs research.liquidity_sweeps: "
                f"{'IDENTICAL' if result.reference_matches_existing else 'MISMATCH'} ({result.existing_event_count} events)."
            )
        if reference is not None:
            c = reference.counts
            lines.append(
                "Reference (existing) definition, candle x live-swing pairs, never merged: "
                f"wick crossings={c.wick_crossings} close-beyond={c.close_beyond} exact touches={c.exact_touches} "
                f"sweeps(wick cross, close back inside)={c.unfiltered_sweeps}; "
                f"candles with a sweep in the last {study.activity_window}: {_pct(reference.active_rate)}; "
                f"mean live swings per candle {reference.mean_live_swings:.1f}."
            )
        lines.append(f"Swing lifetime and repeat policy, structure {ref.left}x{ref.right} {ref.equal_policy}, no penetration filter:")
        lines.append(
            f"  lifetime  repeats  events  active%  live/candle  excess({minutes} min)  z"
        )
        for s in result.summaries:
            d = next(x for x in study.definitions() if x.name == s.name)
            if d.structure_key != ref.structure_key or d.min_penetration_atr != 0:
                continue
            life = "unlimited" if d.lifetime is None else f"{int(d.lifetime.total_seconds() // 3600)}h"
            lines.append(
                f"  {life:<9} {'first' if d.first_only else 'all':<7} {s.events:<7} {_pct(s.active_rate, 0):<8} "
                f"{s.mean_live_swings:<12.1f} {_pp(s.focus.excess):<14} {_z(s.focus.z)}"
            )
        lines.append("Sensitivity of full-sample excess across all tested definitions, by swing structure (descriptive):")
        by_structure: dict[tuple, list] = {}
        for d in study.definitions():
            by_structure.setdefault(d.structure_key, []).append(summaries[d.name])
        for key, items in by_structure.items():
            excess = sorted(s.focus.excess for s in items if s.focus.excess is not None)
            if excess:
                lines.append(
                    f"  {key[0]}x{key[1]} {key[2]:<9} definitions={len(items)} events {min(s.events for s in items)}"
                    f"-{max(s.events for s in items)} excess min {_pp(excess[0])} median {_pp(excess[len(excess) // 2])} "
                    f"max {_pp(excess[-1])}"
                )
        lines.extend(self._detail_lines("Existing definition", result.reference, minutes))
        if result.selected_detail is not None:
            lines.extend(self._detail_lines("Most-selected definition", result.selected_detail, minutes))

        lines.append(
            f"Chronological walk-forward ({len(result.folds)} fold(s); definition and direction chosen on TRAIN events only):"
        )
        for f in result.folds:
            if f.selected is None:
                lines.append(f"  fold {f.number}: no definition had >= {study.min_events} thinned train events")
                continue
            lines.append(
                f"  fold {f.number}: {f.selected} sign {f.sign:+d} train excess {_pp(f.train_excess)} -> "
                f"test oriented excess {_pp(f.oriented_test_excess)} (test events {f.test_raw.n}, thinned {f.test_thin.n})"
            )
        s, r = result.pooled_selected, result.pooled_reference
        lines.append(
            f"Pooled out-of-sample, selected: events {s.raw_n} (thinned {s.thin_n}) aligned {_pct(s.aligned_rate_thin)} vs "
            f"expected {_pct(s.expected_rate_thin)} z {_z(s.z)}; existing definition (reversal-aligned, no selection): "
            f"events {r.raw_n} (thinned {r.thin_n}) aligned {_pct(r.aligned_rate_thin)} vs expected {_pct(r.expected_rate_thin)} "
            f"z {_z(r.z)}."
        )
        if result.selection_counts:
            lines.append(
                "Selection stability: "
                + ", ".join(f"{name} x{count}" for name, count in result.selection_counts)
            )
        lines.append("Criteria (all must pass to justify further sweep/structure experiments):")
        for c in result.criteria:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        lines.append(
            "Sweep/structure association persisted out of sample under the stated criteria; worth further offline "
            "experiments (not a tradeable rule)."
            if result.justifies_further_research
            else "No sweep/structure definition passed every criterion; these events are not yet justified as a "
            "basis for a trade candidate."
        )
        lines.append(
            "Unresolved and NOT tested: sessions/timezone, key levels (e.g. 15:00 New York open), MSS/BMS/BOS/CHOCH, "
            "FVG/IFVG, sweep expiry/invalidation, entry, stop, target and trade management. "
            "The owner's full sequence has not been tested."
        )
        limitations = [
            "A statistical association with later expansion is not a trade candidate: no executable entry, stop, "
            "target, cost or risk rule is defined here.",
            "Single instrument, single timeframe, roughly one year; no spread data.",
            "Many events share candles and forward windows; significance uses events thinned to one per horizon, "
            "and 64 definitions were tried, so the best full-sample numbers are optimistically biased.",
            "Break categories use the existing close-basis structural-break definition, which is not MSS/CHOCH.",
            "Swing lifetime is wall-clock time since confirmation; gaps are not filled and no schedule is assumed.",
            "Regime split uses the reporting-only short/long realized-volatility buckets, not a validated model.",
        ]
        return AgentResult(
            agent=self.name, status="ok",
            headline=(
                f"{result.variants_tried} definitions, {len(result.folds)} fold(s); pooled OOS aligned-excess z "
                f"{_z(s.z)}; " + ("passes" if result.justifies_further_research else "does not pass") + " the stated criteria."
            ),
            lines=tuple(lines), limitations=tuple(limitations), data={"study": result},
        )

    @staticmethod
    def _stat_line(label: str, h: HorizonStat) -> str:
        return (
            f"  {label:<28} n={h.labelled:<7} thin={h.thinned:<6} expansion {_pct(h.expansion_rate)} "
            f"(base {_pct(h.base_expansion_rate)})  aligned {_pct(h.aligned_rate)} vs expected "
            f"{_pct(h.expected_aligned_rate)} against {_pct(h.against_rate)} both {_pct(h.both_rate)} "
            f"excess {_pp(h.excess)} z {_z(h.z)}  outcomes c/r/rev {h.continuation}/{h.retracement}/{h.reversal}"
        )

    def _detail_lines(self, title: str, detail: DetailTable, focus_minutes: int) -> list[str]:
        lines = [f"{title} ({detail.name}), full sample, descriptive:"]
        for h in detail.horizons:
            lines.append(self._stat_line(f"{_minutes(h.horizon)} min", h))
        lines.append(f"  event combinations at {focus_minutes} min (neutral categories, not MSS/CHOCH):")
        for key, h in detail.families:
            lines.append(self._stat_line(key, h))
        for dimension, cells in detail.bins:
            lines.append(
                f"  by {dimension} (excess, n): "
                + " | ".join(f"{label} {_pp(h.excess)} ({h.labelled})" for label, h in cells)
            )
        lines.append("  by month (excess, n): " + " ".join(f"{m}:{_pp(e)}({n})" for m, n, e in detail.by_month))
        if detail.by_regime:
            lines.append("  by volatility regime (excess, n): " + " ".join(f"{m}:{_pp(e)}({n})" for m, n, e in detail.by_regime))
        return lines
