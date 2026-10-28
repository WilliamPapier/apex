"""Causal market-regime research: do observable conditions separate forward expansion?

Research only. Nothing here is a trading rule, signal or model, and nothing
touches a broker.

Classification (uses information available at each candle close only)
    Every candidate regime variant is a feature, a lookback window and a pair
    of quantiles. The feature value of candle ``i`` is turned into a state
    ``low`` / ``mid`` / ``high`` by its mid-rank percentile within the feature
    values of strictly EARLIER candles (an expanding history, minimum
    ``min_history`` values). No threshold is ever fitted on the full dataset,
    and appending candles never changes an earlier candle's state.

    Features: ``atr_level`` (ATR over its trailing mean), ``vol_change``
    (existing short/long realized-volatility ratio), ``efficiency`` (Kaufman
    efficiency ratio: |net move| / path length; high = directional, low =
    range-bound),     ``compression`` (existing short/long mean-range ratio; LOW = compressed,
    HIGH = recent ranges large relative to the longer window),
    ``break_rate`` (confirmed structural breaks in the recent window).
    Own rolling windows are only used when the candles in the window are
    contiguous at the candle duration; gaps are never filled, no schedule is
    assumed, and rows that fail this are counted as unclassified. Features
    taken from the existing predictor snapshots inherit that module's
    behaviour (including its ATR across gaps).

Evaluation (outcomes use FUTURE candles and are never inputs)
    Outcome rows come from the existing expansion labels at one horizon and
    ATR multiple: expanded?, reversal among expansions, forward excursion in
    ATR, and whether the forward move continued the prior momentum direction.
    Variants are compared on chronological walk-forward folds: in each fold
    the variant with the largest |expansion-rate(high) - rate(low)| on TRAIN
    rows is chosen (train labels must have ended before the test fold), and
    only the TEST rows of that fold are pooled. Because neighbouring rows
    share forward windows, significance is computed on rows thinned to one per
    horizon length. All variants tried are counted. The simple baseline is the
    existing "large closed candle" flag, evaluated on the same test rows.

Whether a labelled regime is "genuine" is not assumed: the verdict only
reports whether the separation persisted out of sample, was stable across
folds, beat the simple baseline and lasted long enough to be usable.
"""

from bisect import bisect_left, bisect_right, insort
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Sequence

from core.market_data import Candle
from research.price_expansion import ExpansionConfig, ExpansionResearch, PredictorSnapshot
from research.study_stats import rate, run_lengths, state_changes, thin, z_two_proportion
from research.walk_forward import WalkForwardConfig, make_folds


FEATURES = ("atr_level", "vol_change", "efficiency", "compression", "break_rate")
STATES = ("low", "mid", "high")
BASELINE_NAME = "baseline_large_candle"


def _check_quantiles(pair: tuple[float, float]) -> None:
    low, high = pair
    if not (0 < low < high < 1):
        raise ValueError("Each quantile pair must satisfy 0 < low < high < 1.")


@dataclass(frozen=True)
class RegimeVariant:
    feature: str
    window: int | None
    low_q: float
    high_q: float

    def __post_init__(self) -> None:
        if self.feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}.")
        _check_quantiles((self.low_q, self.high_q))

    @property
    def name(self) -> str:
        window = "" if self.window is None else f"_w{self.window}"
        return f"{self.feature}{window}_q{round(self.low_q * 100)}_{round(self.high_q * 100)}"


@dataclass(frozen=True)
class RegimeStudyConfig:
    atr_level_windows: tuple[int, ...] = (48, 144)
    efficiency_windows: tuple[int, ...] = (12, 36)
    quantile_pairs: tuple[tuple[float, float], ...] = ((1 / 3, 2 / 3), (0.25, 0.75))
    min_history: int = 500
    momentum_lookback: int = 12
    min_state_rows: int = 30
    min_median_run: int = 6
    min_fold_agreement: float = 2 / 3
    z_threshold: float = 2.0
    baseline_margin: float = 0.02

    def __post_init__(self) -> None:
        for name in ("atr_level_windows", "efficiency_windows"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(type(w) is not int or w < 2 for w in value):
                raise ValueError(f"{name} must be a tuple of integers of at least 2.")
        if not self.quantile_pairs:
            raise ValueError("quantile_pairs must not be empty.")
        for pair in self.quantile_pairs:
            _check_quantiles(pair)
        for name in ("min_history", "momentum_lookback", "min_state_rows", "min_median_run"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.min_fold_agreement <= 1:
            raise ValueError("min_fold_agreement must be in (0, 1].")
        if not self.z_threshold > 0:
            raise ValueError("z_threshold must be positive.")
        if not isinstance(self.baseline_margin, (int, float)) or not self.baseline_margin >= 0:
            raise ValueError("baseline_margin must be a non-negative number.")

    def feature_windows(self) -> tuple[tuple[str, int | None], ...]:
        return (
            *(("atr_level", w) for w in self.atr_level_windows),
            ("vol_change", None),
            *(("efficiency", w) for w in self.efficiency_windows),
            ("compression", None),
            ("break_rate", None),
        )

    def variants(self) -> tuple[RegimeVariant, ...]:
        return tuple(
            RegimeVariant(feature, window, low, high)
            for feature, window in self.feature_windows()
            for low, high in self.quantile_pairs
        )


def contiguous_run_lengths(candles: Sequence[Candle], duration: timedelta) -> list[int]:
    """Number of consecutive exactly-``duration``-spaced candles ending at each candle (itself included)."""

    runs: list[int] = []
    previous: datetime | None = None
    for candle in candles:
        now = candle.timestamp.astimezone(timezone.utc)
        runs.append(runs[-1] + 1 if previous is not None and now - previous == duration else 1)
        previous = now
    return runs


def compute_feature(
    feature: str,
    window: int | None,
    candles: Sequence[Candle],
    snapshots: Sequence[PredictorSnapshot],
    runs: Sequence[int],
    expansion: ExpansionConfig,
) -> list[float | None]:
    """Raw causal feature value per candle (None when unavailable or the window is not contiguous)."""

    n = len(candles)
    out: list[float | None] = [None] * n
    if feature == "efficiency":
        closes = [c.close for c in candles]
        path = [0.0] * n
        for i in range(1, n):
            path[i] = path[i - 1] + abs(closes[i] - closes[i - 1])
        for i in range(window, n):
            if runs[i] < window + 1:
                continue
            length = path[i] - path[i - window]
            if length > 0:
                out[i] = abs(closes[i] - closes[i - window]) / length
    elif feature == "atr_level":
        atrs = [s.atr for s in snapshots]
        total = [0.0] * (n + 1)
        count = [0] * (n + 1)
        for i, atr in enumerate(atrs):
            total[i + 1] = total[i] + (atr or 0.0)
            count[i + 1] = count[i] + (atr is not None)
        for i in range(window, n):
            if runs[i] < window + 1 or atrs[i] is None or count[i] - count[i - window] != window:
                continue
            mean = (total[i] - total[i - window]) / window
            if mean > 0:
                out[i] = atrs[i] / mean
    elif feature == "vol_change":
        need = max(expansion.volatility_periods) + 1
        for i, s in enumerate(snapshots):
            if runs[i] >= need:
                out[i] = s.volatility_ratio
    elif feature == "compression":
        for i, s in enumerate(snapshots):
            if runs[i] >= expansion.compression_long:
                out[i] = s.range_compression_ratio
    else:
        for i, s in enumerate(snapshots):
            if runs[i] >= expansion.event_lookback:
                out[i] = float(s.recent_up_breaks + s.recent_down_breaks)
    return out


def causal_percentile_ranks(values: Sequence[float | None], min_history: int) -> list[float | None]:
    """Mid-rank percentile of each value among STRICTLY EARLIER values; None until ``min_history`` exist."""

    history: list[float] = []
    ranks: list[float | None] = []
    for value in values:
        if value is None:
            ranks.append(None)
            continue
        if len(history) >= min_history:
            below, upto = bisect_left(history, value), bisect_right(history, value)
            ranks.append((below + upto) / 2 / len(history))
        else:
            ranks.append(None)
        insort(history, value)
    return ranks


def classify_rank(rank: float | None, low_q: float, high_q: float) -> str | None:
    if rank is None:
        return None
    if rank < low_q:
        return "low"
    if rank > high_q:
        return "high"
    return "mid"


@dataclass(frozen=True)
class OutcomeRow:
    """Forward outcome of candle ``index`` (FUTURE information: evaluation only)."""

    index: int
    timestamp: datetime
    window_end: datetime
    expanded: bool
    reversal: bool | None
    excursion_atr: float | None
    momentum_continued: bool | None


def build_outcome_rows(
    candles: Sequence[Candle],
    research: ExpansionResearch,
    horizon: timedelta,
    atr_multiple: float,
    runs: Sequence[int],
    momentum_lookback: int,
) -> tuple[tuple[OutcomeRow, ...], Counter]:
    """Labelled rows at one horizon/multiple plus a count of excluded rows by label status."""

    rows: list[OutcomeRow] = []
    excluded: Counter = Counter()
    for label in research.labels:
        if label.horizon != horizon or label.atr_multiple != atr_multiple:
            continue
        if label.expanded is None:
            excluded[label.status] += 1
            continue
        i = label.index
        excursion = None
        if label.max_up is not None and label.max_down is not None and label.atr:
            excursion = max(label.max_up, label.max_down) / label.atr
        continued = None
        if i >= momentum_lookback and runs[i] >= momentum_lookback + 1 and label.end_close_change:
            prior = candles[i].close - candles[i - momentum_lookback].close
            if prior != 0:
                continued = (prior > 0) == (label.end_close_change > 0)
        reversal = None
        if label.expanded and label.outcome is not None:
            reversal = label.outcome == "reversal"
        rows.append(
            OutcomeRow(i, label.timestamp, label.window_end, label.expanded, reversal, excursion, continued)
        )
    rows.sort(key=lambda r: r.index)
    return tuple(rows), excluded


@dataclass
class Tally:
    n: int = 0
    expanded: int = 0
    reversal_n: int = 0
    reversals: int = 0
    cont_n: int = 0
    cont_hits: int = 0
    excursion_n: int = 0
    excursion_sum: float = 0.0

    def add(self, row: OutcomeRow) -> None:
        self.n += 1
        self.expanded += row.expanded
        if row.reversal is not None:
            self.reversal_n += 1
            self.reversals += row.reversal
        if row.momentum_continued is not None:
            self.cont_n += 1
            self.cont_hits += row.momentum_continued
        if row.excursion_atr is not None:
            self.excursion_n += 1
            self.excursion_sum += row.excursion_atr

    def merge(self, other: "Tally") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def expansion_rate(self) -> float | None:
        return rate(self.expanded, self.n)

    @property
    def reversal_rate(self) -> float | None:
        return rate(self.reversals, self.reversal_n)

    @property
    def continuation_rate(self) -> float | None:
        return rate(self.cont_hits, self.cont_n)

    @property
    def mean_excursion_atr(self) -> float | None:
        return self.excursion_sum / self.excursion_n if self.excursion_n else None


def _tally_by_state(rows: Sequence[OutcomeRow], states: Sequence[str | None]) -> dict[str, Tally]:
    out = {state: Tally() for state in STATES}
    for row in rows:
        state = states[row.index]
        if state is not None:
            out[state].add(row)
    return out


def _spread(tallies: dict[str, Tally], minimum: int) -> float | None:
    high, low = tallies["high"], tallies["low"]
    if high.n < minimum or low.n < minimum:
        return None
    return high.expansion_rate - low.expansion_rate


@dataclass(frozen=True)
class VariantDescription:
    """Full-sample description. The classification is causal, but the outcome columns are descriptive only."""

    name: str
    classified_rows: int
    unclassified_rows: int
    frequency: tuple[tuple[str, float], ...]
    median_run: tuple[tuple[str, float | None], ...]
    median_run_all: float | None
    changes_per_100: float | None
    expansion_rate: tuple[tuple[str, float | None, int], ...]
    reversal_rate: tuple[tuple[str, float | None], ...]
    continuation_rate: tuple[tuple[str, float | None], ...]
    mean_excursion_atr: tuple[tuple[str, float | None], ...]
    monthly: tuple[tuple[str, tuple[tuple[str, float | None, int], ...]], ...]


def describe_variant(
    name: str,
    states: Sequence[str | None],
    rows: Sequence[OutcomeRow],
    breaks_before: Sequence[bool],
) -> VariantDescription:
    tallies = _tally_by_state(rows, states)
    classified = sum(t.n for t in tallies.values())
    state_counts = Counter(s for s in states if s is not None)
    total_states = sum(state_counts.values())
    runs = run_lengths(states, breaks_before)
    changes, pairs = state_changes(states, breaks_before)
    monthly: dict[str, dict[str, Tally]] = {}
    for row in rows:
        state = states[row.index]
        if state is None:
            continue
        key = f"{row.timestamp.year}-{row.timestamp.month:02d}"
        monthly.setdefault(key, {s: Tally() for s in STATES})[state].add(row)
    return VariantDescription(
        name=name,
        classified_rows=classified,
        unclassified_rows=len(rows) - classified,
        frequency=tuple((s, state_counts[s] / total_states if total_states else 0.0) for s in STATES),
        median_run=tuple(
            (s, median([length for st, length in runs if st == s]) if any(st == s for st, _ in runs) else None)
            for s in STATES
        ),
        median_run_all=median([length for _, length in runs]) if runs else None,
        changes_per_100=100 * changes / pairs if pairs else None,
        expansion_rate=tuple((s, tallies[s].expansion_rate, tallies[s].n) for s in STATES),
        reversal_rate=tuple((s, tallies[s].reversal_rate) for s in STATES),
        continuation_rate=tuple((s, tallies[s].continuation_rate) for s in STATES),
        mean_excursion_atr=tuple((s, tallies[s].mean_excursion_atr) for s in STATES),
        monthly=tuple(
            (key, tuple((s, t[s].expansion_rate, t[s].n) for s in STATES)) for key, t in sorted(monthly.items())
        ),
    )


@dataclass(frozen=True)
class RegimeFoldResult:
    number: int
    test_from: datetime
    test_until: datetime
    selected: str | None
    train_spread: float | None
    train_rows: int
    test_hot: Tally
    test_cold: Tally
    test_hot_thin: Tally
    test_cold_thin: Tally
    hot_state: str | None

    @property
    def test_spread(self) -> float | None:
        """Hot-state minus cold-state expansion rate on raw test rows (hot = higher TRAIN rate)."""

        if self.test_hot.n == 0 or self.test_cold.n == 0:
            return None
        return self.test_hot.expansion_rate - self.test_cold.expansion_rate


@dataclass(frozen=True)
class PooledOutOfSample:
    hot: Tally
    cold: Tally
    hot_thin: Tally
    cold_thin: Tally

    @property
    def spread(self) -> float | None:
        if self.hot.n == 0 or self.cold.n == 0:
            return None
        return self.hot.expansion_rate - self.cold.expansion_rate

    @property
    def z(self) -> float | None:
        return z_two_proportion(
            self.hot_thin.expanded, self.hot_thin.n, self.cold_thin.expanded, self.cold_thin.n
        )


@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RegimeStudyResult:
    horizon: timedelta
    atr_multiple: float
    total_candles: int
    labelled_rows: int
    excluded_by_status: tuple[tuple[str, int], ...]
    base_rate: float | None
    variants_tried: int
    descriptions: tuple[VariantDescription, ...]
    baseline: VariantDescription
    folds: tuple[RegimeFoldResult, ...]
    pooled: PooledOutOfSample
    baseline_pooled: PooledOutOfSample
    unconditional_oos: Tally
    selection_counts: tuple[tuple[str, int], ...]
    most_selected: str | None
    criteria: tuple[Criterion, ...]
    justifies_further_research: bool


def _sum_tallies(items: Sequence[Tally]) -> Tally:
    total = Tally()
    for item in items:
        total.merge(item)
    return total


def run_regime_study(
    candles: Sequence[Candle],
    research: ExpansionResearch,
    expansion: ExpansionConfig,
    study: RegimeStudyConfig,
    walk_forward: WalkForwardConfig,
    horizon: timedelta,
    atr_multiple: float,
    baseline_large_candle_atr: float = 1.5,
) -> RegimeStudyResult:
    """Run the full causal regime study (see module doc). Deterministic; read-only."""

    if horizon not in expansion.horizons or atr_multiple not in expansion.atr_multiples:
        raise ValueError("horizon and atr_multiple must be configured in the expansion config.")
    duration = expansion.candle_duration
    snapshots = research.snapshots
    runs = contiguous_run_lengths(candles, duration)
    breaks_before = [r == 1 for r in runs]
    rows, excluded = build_outcome_rows(
        candles, research, horizon, atr_multiple, runs, study.momentum_lookback
    )
    horizon_candles = horizon // duration

    rank_cache: dict[tuple[str, int | None], list[float | None]] = {}
    variants = study.variants()
    states: dict[str, list[str | None]] = {}
    for variant in variants:
        key = (variant.feature, variant.window)
        if key not in rank_cache:
            values = compute_feature(variant.feature, variant.window, candles, snapshots, runs, expansion)
            rank_cache[key] = causal_percentile_ranks(values, study.min_history)
        states[variant.name] = [classify_rank(r, variant.low_q, variant.high_q) for r in rank_cache[key]]
    baseline_states: list[str | None] = [
        None if s.range_to_atr is None else ("high" if s.range_to_atr >= baseline_large_candle_atr else "low")
        for s in snapshots
    ]

    descriptions = tuple(describe_variant(v.name, states[v.name], rows, breaks_before) for v in variants)
    baseline_description = describe_variant(BASELINE_NAME, baseline_states, rows, breaks_before)

    fold_results: list[RegimeFoldResult] = []
    baseline_hot, baseline_cold = [], []
    baseline_hot_thin, baseline_cold_thin = [], []
    unconditional = Tally()
    for fold in make_folds(candles, walk_forward, duration):
        train = [
            r for r in rows
            if fold.train_start <= r.index < fold.train_end and r.window_end <= fold.test_from
        ]
        test = [r for r in rows if fold.test_start <= r.index < fold.test_end]
        best: tuple[float, RegimeVariant] | None = None
        for variant in variants:
            spread = _spread(_tally_by_state(train, states[variant.name]), study.min_state_rows)
            if spread is not None and (best is None or abs(spread) > abs(best[0])):
                best = (spread, variant)
        thinned_index = set(thin([r.index for r in test], horizon_candles))
        thinned = [r for r in test if r.index in thinned_index]
        for r in test:
            unconditional.add(r)
        raw_base = _tally_by_state(test, baseline_states)
        thin_base = _tally_by_state(thinned, baseline_states)
        baseline_hot.append(raw_base["high"])
        baseline_cold.append(raw_base["low"])
        baseline_hot_thin.append(thin_base["high"])
        baseline_cold_thin.append(thin_base["low"])
        if best is None:
            empty = Tally()
            fold_results.append(
                RegimeFoldResult(fold.number, fold.test_from, fold.test_until, None, None, len(train),
                                 empty, Tally(), Tally(), Tally(), None)
            )
            continue
        spread, variant = best
        hot_state, cold_state = ("high", "low") if spread > 0 else ("low", "high")
        raw = _tally_by_state(test, states[variant.name])
        thin_tally = _tally_by_state(thinned, states[variant.name])
        fold_results.append(
            RegimeFoldResult(
                fold.number, fold.test_from, fold.test_until, variant.name, spread, len(train),
                raw[hot_state], raw[cold_state], thin_tally[hot_state], thin_tally[cold_state], hot_state,
            )
        )

    pooled = PooledOutOfSample(
        _sum_tallies([f.test_hot for f in fold_results]),
        _sum_tallies([f.test_cold for f in fold_results]),
        _sum_tallies([f.test_hot_thin for f in fold_results]),
        _sum_tallies([f.test_cold_thin for f in fold_results]),
    )
    baseline_pooled = PooledOutOfSample(
        _sum_tallies(baseline_hot), _sum_tallies(baseline_cold),
        _sum_tallies(baseline_hot_thin), _sum_tallies(baseline_cold_thin),
    )
    selection = Counter(f.selected for f in fold_results if f.selected)
    order = {v.name: i for i, v in enumerate(variants)}
    selection_counts = tuple(sorted(selection.items(), key=lambda kv: (-kv[1], order[kv[0]])))
    most_selected = selection_counts[0][0] if selection_counts else None
    described = {d.name: d for d in descriptions}

    evaluated = [f for f in fold_results if f.test_spread is not None]
    agreeing = sum(1 for f in evaluated if f.test_spread > 0)
    z = pooled.z
    baseline_z = baseline_pooled.z
    median_run = described[most_selected].median_run_all if most_selected is not None else None
    criteria = (
        Criterion(
            "out_of_sample_separation",
            z is not None and z >= study.z_threshold,
            f"pooled thinned z={'n/a' if z is None else f'{z:.2f}'} (needs >= {study.z_threshold:g})",
        ),
        Criterion(
            "fold_consistency",
            bool(evaluated) and agreeing / len(evaluated) >= study.min_fold_agreement,
            f"{agreeing}/{len(evaluated)} folds kept the train-chosen direction "
            f"(needs >= {study.min_fold_agreement:.0%})",
        ),
        Criterion(
            "beats_simple_baseline",
            pooled.spread is not None and baseline_pooled.spread is not None
            and pooled.spread > baseline_pooled.spread + study.baseline_margin,
            f"selected OOS spread {_fmt(pooled.spread)} vs baseline flag {_fmt(baseline_pooled.spread)} "
            f"(needs a margin of {_fmt(study.baseline_margin)}; baseline z={'n/a' if baseline_z is None else f'{baseline_z:.2f}'})",
        ),
        Criterion(
            "usable_duration",
            median_run is not None and median_run >= study.min_median_run,
            f"median regime run {'n/a' if median_run is None else f'{median_run:g}'} candles "
            f"(needs >= {study.min_median_run}, so a label outlasts the evaluation horizon)",
        ),
    )
    return RegimeStudyResult(
        horizon=horizon, atr_multiple=atr_multiple, total_candles=len(candles), labelled_rows=len(rows),
        excluded_by_status=tuple(sorted(excluded.items())),
        base_rate=unconditional.expansion_rate if unconditional.n else None,
        variants_tried=len(variants), descriptions=descriptions, baseline=baseline_description,
        folds=tuple(fold_results), pooled=pooled, baseline_pooled=baseline_pooled,
        unconditional_oos=unconditional, selection_counts=selection_counts, most_selected=most_selected,
        criteria=criteria, justifies_further_research=all(c.passed for c in criteria),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}pp"
