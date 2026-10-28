"""Liquidity-sweep and market-structure research: how do the existing definitions behave?

Research only. Nothing here is a strategy, entry, stop, target or model, and
nothing touches a broker. The user's full hypothesis (sweep -> MSS/BMS/BOS/CHOCH
-> IFVG -> new FVG -> key-level retest) is NOT implemented here: sessions,
key levels, MSS/CHOCH and IFVG rules are unresolved. The only combination
studied is the neutral "structural break (existing definition) shortly after /
before a sweep" so that the effect of combining two events can be measured.

Events and point-in-time availability
    Reference levels are the confirmed swings of ``research.market_structure``;
    a swing is only usable by candles whose open is at or after its
    ``confirmed_at`` (right-side confirmation complete). A sweep or break is
    known when its candle closes; that candle is the prediction point.

Distinct crossing categories (counted as candle x live-swing pairs, never merged)
    wick crossing  - the extreme goes strictly beyond the level
    close beyond   - the close is strictly beyond the level
    exact touch    - the extreme equals the level (never a sweep)
    unfiltered sweep - wick crossing with the close back inside (existing definition)
    qualifying sweep - an unfiltered sweep whose penetration > min_penetration_atr x ATR
                       (ATR at the sweep candle close; no ATR -> not qualifying when > 0)

Tested definition axes (all counted as variants)
    swing lookbacks / equal policy, swing LIFETIME (live while
    open - confirmed_at < lifetime; None = unlimited, the existing behaviour),
    repeat policy (every qualifying sweep vs first sweep per swing only) and
    the ATR penetration filter. Unlimited lifetime with repeats can keep the
    sweep feature almost always on; ``active_rate`` and ``mean_live_swings``
    quantify that.

Outcome (FUTURE labels, evaluation only)
    From the existing expansion labels at 5/15/30 min and one ATR multiple.
    ``aligned`` = expansion in the event's implied direction (down after a
    high-side sweep, up after a low-side sweep, the break direction for
    breaks). The expected aligned rate comes from the unconditional rates of
    labelled rows in the same period, so ``excess`` = aligned - expected is a
    measured lift over the market, not a trade result. Significance uses
    events thinned to one per horizon length because windows overlap.

Out-of-sample: per chronological fold the definition with the largest
|train excess| is chosen on TRAIN events only (labels must end before the
test fold) and its TEST events are pooled with the train-chosen sign. The
number of definitions tried is recorded. An association is not a profitable
setup; no executable entry, stop, target, cost or risk rule is defined here.
"""

import gc
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import sqrt
from statistics import median
from typing import Sequence

from core.market_data import Candle
from research.market_structure import (
    EQUAL_POLICIES,
    MarketStructure,
    StructureConfig,
    StructuralBreak,
    SwingPoint,
    analyze_market_structure,
)
from research.price_expansion import ExpansionConfig, ExpansionLabel, ExpansionResearch
from research.study_stats import thin, z_against_expected
from research.walk_forward import WalkForwardConfig, make_folds


PROMINENCE_LOOKBACK = 12
_BIG = 10**18
REFERENCE_STRUCTURE = (2, 2, "strict")


@dataclass(frozen=True)
class SweepDefinition:
    left: int
    right: int
    equal_policy: str
    lifetime: timedelta | None
    first_only: bool
    min_penetration_atr: float

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.equal_policy not in EQUAL_POLICIES:
            raise ValueError(f"equal_policy must be one of {EQUAL_POLICIES}.")
        if self.lifetime is not None and (not isinstance(self.lifetime, timedelta) or self.lifetime <= timedelta(0)):
            raise ValueError("lifetime must be None or a positive timedelta.")
        if type(self.first_only) is not bool:
            raise ValueError("first_only must be a bool.")
        if not isinstance(self.min_penetration_atr, (int, float)) or not self.min_penetration_atr >= 0:
            raise ValueError("min_penetration_atr must be a non-negative number.")

    @property
    def structure_key(self) -> tuple[int, int, str]:
        return (self.left, self.right, self.equal_policy)

    @property
    def name(self) -> str:
        life = "unlimited" if self.lifetime is None else f"{int(self.lifetime.total_seconds() // 3600)}h"
        repeat = "first" if self.first_only else "all"
        return (
            f"sw{self.left}x{self.right}_{self.equal_policy[:3]}_life{life}_{repeat}"
            f"_pen{self.min_penetration_atr:g}"
        )


@dataclass(frozen=True)
class LiquidityStudyConfig:
    structures: tuple[tuple[int, int, str], ...] = (
        REFERENCE_STRUCTURE, (2, 2, "inclusive"), (3, 3, "strict"), (5, 5, "strict"),
    )
    lifetimes: tuple[timedelta | None, ...] = (
        timedelta(hours=1), timedelta(hours=4), timedelta(hours=24), None,
    )
    first_only_options: tuple[bool, ...] = (False, True)
    min_penetration_atrs: tuple[float, ...] = (0.0, 0.25)
    pairing_window: int = 12
    activity_window: int = 12
    min_events: int = 30
    min_pooled_events: int = 100
    min_fold_agreement: float = 2 / 3
    z_threshold: float = 2.0

    def __post_init__(self) -> None:
        for name in ("structures", "lifetimes", "first_only_options", "min_penetration_atrs"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty.")
        for name in ("pairing_window", "activity_window", "min_events", "min_pooled_events"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.min_fold_agreement <= 1:
            raise ValueError("min_fold_agreement must be in (0, 1].")
        if not self.z_threshold > 0:
            raise ValueError("z_threshold must be positive.")
        self.definitions()  # validates every combination

    def definitions(self) -> tuple[SweepDefinition, ...]:
        return tuple(
            SweepDefinition(left, right, policy, life, first, pen)
            for left, right, policy in self.structures
            for life in self.lifetimes
            for first in self.first_only_options
            for pen in self.min_penetration_atrs
        )

    @property
    def reference(self) -> SweepDefinition:
        """The behaviour of the existing detector: reference swings, unlimited lifetime, every sweep, no filter."""

        left, right, policy = REFERENCE_STRUCTURE
        return SweepDefinition(left, right, policy, None, False, 0.0)


@dataclass(frozen=True, slots=True)
class SweepEvent:
    side: str
    index: int
    timestamp: datetime
    available_at: datetime
    level: float
    penetration: float
    penetration_atr: float | None
    age: timedelta
    prior_sweeps: int
    swing_index: int
    prominence_atr: float | None

    @property
    def expected(self) -> str:
        """Implied direction of a reversal after this sweep (hypothesis label, not a prediction)."""

        return "down" if self.side == "high" else "up"


@dataclass(frozen=True)
class PointEvent:
    index: int
    timestamp: datetime
    expected: str


@dataclass(frozen=True)
class CrossingCounts:
    wick_crossings: int
    close_beyond: int
    exact_touches: int
    unfiltered_sweeps: int
    qualifying_sweeps: int


@dataclass(frozen=True)
class DetectionResult:
    events: tuple[SweepEvent, ...]
    counts: CrossingCounts
    mean_live_swings: float
    active_rate: float | None


def _swing_prominence(candles: Sequence[Candle], swing: SwingPoint, atr: float | None) -> float | None:
    if not atr:
        return None
    first = max(0, swing.candle_index - PROMINENCE_LOOKBACK)
    window = candles[first : swing.confirmation_index + 1]
    if swing.kind == "high":
        return (swing.price - min(c.low for c in window)) / atr
    return (max(c.high for c in window) - swing.price) / atr


def detect_sweeps(
    candles: Sequence[Candle],
    structure: MarketStructure,
    atrs: Sequence[float | None],
    definition: SweepDefinition,
    candle_duration: timedelta,
    activity_window: int = 12,
) -> DetectionResult:
    """Efficient lifetime-aware sweep detection over already-confirmed swings.

    With lifetime None, repeats allowed and no penetration filter this yields
    the same events as ``research.liquidity_sweeps.detect_liquidity_sweeps``
    (cross-checked in tests and by the agent on real data).
    """

    n = len(candles)
    opens = [c.timestamp.astimezone(timezone.utc) for c in candles]
    events: list[SweepEvent] = []
    wick = beyond = touch = unfiltered = 0
    live_total = 0
    t_factor = definition.min_penetration_atr
    for side, sign, swings in (
        ("high", 1, structure.swing_highs),
        ("low", -1, structure.swing_lows),
    ):
        prominence: dict[int, float | None] = {}
        keys: list[tuple[float, int]] = []
        queue: deque[int] = deque()
        removed: set[int] = set()
        prior: Counter = Counter()
        nxt = 0

        def drop(order: int) -> None:
            key = (sign * swings[order].price, order)
            position = bisect_left(keys, key)
            if position < len(keys) and keys[position] == key:
                del keys[position]
            removed.add(order)

        for i in range(n):
            open_i = opens[i]
            while nxt < len(swings) and swings[nxt].confirmed_at <= open_i:
                swing = swings[nxt]
                position = bisect_left(keys, (sign * swing.price, nxt))
                keys.insert(position, (sign * swing.price, nxt))
                queue.append(nxt)
                nxt += 1
            if definition.lifetime is not None:
                while queue and (
                    queue[0] in removed or open_i - swings[queue[0]].confirmed_at >= definition.lifetime
                ):
                    order = queue.popleft()
                    if order not in removed:
                        drop(order)
            live_total += len(keys)
            if not keys:
                continue
            candle = candles[i]
            extreme = candle.high if sign == 1 else -candle.low
            close = sign * candle.close
            below_extreme = bisect_left(keys, (extreme, -1))
            above_close = bisect_right(keys, (close, _BIG))
            wick += below_extreme
            beyond += bisect_left(keys, (close, -1))
            touch += bisect_right(keys, (extreme, _BIG)) - below_extreme
            unfiltered += max(0, below_extreme - above_close)
            atr = atrs[i]
            threshold = t_factor * atr if atr is not None else None
            if t_factor > 0 and threshold is None:
                continue
            upper = below_extreme if not t_factor else bisect_left(keys, (extreme - threshold, -1))
            hits = []
            for level, order in keys[above_close:upper]:
                penetration = extreme - level
                if penetration <= (threshold or 0.0):
                    continue
                swing = swings[order]
                if order not in prominence:
                    prominence[order] = _swing_prominence(candles, swing, atrs[swing.confirmation_index])
                events.append(
                    SweepEvent(
                        side=side, index=i, timestamp=candle.timestamp,
                        available_at=open_i + candle_duration, level=swing.price,
                        penetration=penetration,
                        penetration_atr=penetration / atr if atr else None,
                        age=open_i - swing.confirmed_at, prior_sweeps=prior[order],
                        swing_index=swing.candle_index,
                        prominence_atr=prominence[order],
                    )
                )
                prior[order] += 1
                hits.append(order)
            if definition.first_only:
                for order in hits:
                    drop(order)
    events.sort(key=lambda e: (e.index, e.side != "high", e.swing_index))

    active = None
    if n:
        marks = [0] * (n + 1)
        for index in {e.index for e in events}:
            marks[index] += 1
            marks[min(n, index + activity_window)] -= 1
        running = covered = 0
        for i in range(n):
            running += marks[i]
            covered += running > 0
        active = covered / n
    return DetectionResult(
        tuple(events),
        CrossingCounts(wick, beyond, touch, unfiltered, len(events)),
        live_total / n if n else 0.0,
        active,
    )


@dataclass(frozen=True)
class BaseRates:
    n: int
    expanded: float | None
    up: float | None
    down: float | None


def base_rates(labels: Sequence[ExpansionLabel | None], low: int, high: int, cutoff: datetime | None = None) -> BaseRates:
    """Unconditional outcome rates of labelled rows with index in [low, high) (labels ending by ``cutoff``)."""

    n = expanded = up = down = 0
    for label in labels[low:high]:
        if label is None or label.expanded is None or (cutoff is not None and label.window_end > cutoff):
            continue
        n += 1
        if label.expanded:
            expanded += 1
            up += label.direction == "up"
            down += label.direction == "down"
    if not n:
        return BaseRates(0, None, None, None)
    return BaseRates(n, expanded / n, up / n, down / n)


@dataclass
class EventTally:
    n: int = 0
    expanded: int = 0
    aligned: int = 0
    against: int = 0
    both: int = 0
    continuation: int = 0
    retracement: int = 0
    reversal: int = 0
    expected_aligned: float = 0.0
    expected_expanded: float = 0.0
    variance: float = 0.0

    def add(self, label: ExpansionLabel, expected: str, base: BaseRates) -> None:
        probability = base.up if expected == "up" else base.down
        if probability is None or base.expanded is None:
            return
        self.n += 1
        self.expected_aligned += probability
        self.expected_expanded += base.expanded
        self.variance += probability * (1 - probability)
        if not label.expanded:
            return
        self.expanded += 1
        if label.direction == expected:
            self.aligned += 1
        elif label.direction == "both":
            self.both += 1
        else:
            self.against += 1
        if label.outcome == "continuation":
            self.continuation += 1
        elif label.outcome == "retracement":
            self.retracement += 1
        elif label.outcome == "reversal":
            self.reversal += 1

    @property
    def excess_hits(self) -> float:
        return self.aligned - self.expected_aligned

    @property
    def excess_rate(self) -> float | None:
        return self.excess_hits / self.n if self.n else None

    @property
    def z(self) -> float | None:
        return z_against_expected(self.aligned, self.expected_aligned, self.variance)

    def rate(self, hits: int) -> float | None:
        return hits / self.n if self.n else None


def tally_events(
    events: Sequence,
    labels: Sequence[ExpansionLabel | None],
    base: BaseRates,
    horizon_candles: int,
    low: int = 0,
    high: int | None = None,
    cutoff: datetime | None = None,
) -> tuple[EventTally, EventTally]:
    """(raw, thinned) tallies of events with prediction index in [low, high) whose labels are usable."""

    high = len(labels) if high is None else high
    start = bisect_left(events, low, key=lambda e: e.index)
    stop = bisect_left(events, high, key=lambda e: e.index)
    usable = []
    for event in events[start:stop]:
        label = labels[event.index]
        if label is None or label.expanded is None or (cutoff is not None and label.window_end > cutoff):
            continue
        usable.append((event, label))
    raw, thinned = EventTally(), EventTally()
    keep = set()
    last = None
    for position, (event, _) in enumerate(usable):
        if last is None or event.index - last >= horizon_candles:
            keep.add(position)
            last = event.index
    for position, (event, label) in enumerate(usable):
        raw.add(label, event.expected, base)
        if position in keep:
            thinned.add(label, event.expected, base)
    return raw, thinned


@dataclass(frozen=True)
class HorizonStat:
    horizon: timedelta
    labelled: int
    thinned: int
    expansion_rate: float | None
    base_expansion_rate: float | None
    aligned_rate: float | None
    expected_aligned_rate: float | None
    against_rate: float | None
    both_rate: float | None
    excess: float | None
    z: float | None
    continuation: int
    retracement: int
    reversal: int


def horizon_stat(
    events: Sequence, labels: Sequence[ExpansionLabel | None], horizon: timedelta, duration: timedelta
) -> HorizonStat:
    base = base_rates(labels, 0, len(labels))
    raw, thinned = tally_events(events, labels, base, horizon // duration)
    return HorizonStat(
        horizon=horizon, labelled=raw.n, thinned=thinned.n,
        expansion_rate=raw.rate(raw.expanded),
        base_expansion_rate=base.expanded,
        aligned_rate=raw.rate(raw.aligned),
        expected_aligned_rate=raw.expected_aligned / raw.n if raw.n else None,
        against_rate=raw.rate(raw.against), both_rate=raw.rate(raw.both),
        excess=raw.excess_rate, z=thinned.z,
        continuation=raw.continuation, retracement=raw.retracement, reversal=raw.reversal,
    )


def build_families(
    events: Sequence[SweepEvent], breaks: Sequence[StructuralBreak], window: int
) -> dict[str, list[PointEvent]]:
    """Neutral sweep/break combinations, each at the candle where the combination is first knowable.

    sweep                        every sweep (point = sweep candle)
    sweep_with_prior_break       a structural break (any direction) within ``window`` candles up to the sweep
    break                        every structural break (point = break candle, implied direction = break direction)
    break_after_opposite_sweep   a break whose direction opposes a sweep within ``window`` candles up to the break
                                 (high-side sweep then down break, low-side sweep then up break)
    break_without_recent_sweep   the remaining breaks
    These are observation categories, not MSS/CHOCH/BOS definitions.
    """

    break_indices = sorted(b.break_index for b in breaks)
    sweeps_by_side = {side: sorted({e.index for e in events if e.side == side}) for side in ("high", "low")}

    def within(sorted_indices: list[int], index: int) -> bool:
        return bisect_right(sorted_indices, index) - bisect_left(sorted_indices, index - window) > 0

    families: dict[str, list[PointEvent]] = {
        "sweep": [], "sweep_with_prior_break": [], "break": [],
        "break_after_opposite_sweep": [], "break_without_recent_sweep": [],
    }
    for e in events:
        point = PointEvent(e.index, e.timestamp, e.expected)
        families["sweep"].append(point)
        if within(break_indices, e.index):
            families["sweep_with_prior_break"].append(point)
    for b in sorted(breaks, key=lambda x: (x.break_index, x.direction)):
        point = PointEvent(b.break_index, b.break_timestamp, b.direction)
        families["break"].append(point)
        opposing = "high" if b.direction == "down" else "low"
        key = "break_after_opposite_sweep" if within(sweeps_by_side[opposing], b.break_index) else "break_without_recent_sweep"
        families[key].append(point)
    return families


PENETRATION_BINS = (("<=0.1 ATR", 0.0, 0.1), ("0.1-0.25", 0.1, 0.25), ("0.25-0.5", 0.25, 0.5), (">0.5", 0.5, float("inf")))
AGE_BINS = (
    ("<1h", timedelta(0), timedelta(hours=1)), ("1-4h", timedelta(hours=1), timedelta(hours=4)),
    ("4-24h", timedelta(hours=4), timedelta(hours=24)), (">=24h", timedelta(hours=24), timedelta.max),
)
REPEAT_BINS = (("first", 0, 1), ("second", 1, 2), ("third+", 2, 10**9))
PROMINENCE_BINS = (("<1 ATR", 0.0, 1.0), ("1-2", 1.0, 2.0), ("2-4", 2.0, 4.0), (">=4", 4.0, float("inf")))


def bin_events(events: Sequence[SweepEvent]) -> dict[str, dict[str, list[SweepEvent]]]:
    """Fixed, documented (untuned) descriptive bins; events lacking the attribute are ``n/a``."""

    def place(table, value):
        for label, lo, hi in table:
            if lo <= value < hi:
                return label
        return "n/a"

    out: dict[str, dict[str, list[SweepEvent]]] = {
        "penetration": {}, "age": {}, "repeat": {}, "prominence": {},
    }
    for e in events:
        cells = (
            ("penetration", "n/a" if e.penetration_atr is None else place(PENETRATION_BINS, e.penetration_atr)),
            ("age", place(AGE_BINS, e.age)),
            ("repeat", place(REPEAT_BINS, e.prior_sweeps)),
            ("prominence", "n/a" if e.prominence_atr is None else place(PROMINENCE_BINS, e.prominence_atr)),
        )
        for dimension, label in cells:
            out[dimension].setdefault(label, []).append(e)
    return out


@dataclass(frozen=True)
class VariantSummary:
    name: str
    events: int
    counts: CrossingCounts
    mean_live_swings: float
    active_rate: float | None
    focus: HorizonStat


@dataclass(frozen=True)
class FoldSelection:
    number: int
    selected: str | None
    sign: int
    train_excess: float | None
    train_events: int
    test_raw: EventTally
    test_thin: EventTally
    reference_raw: EventTally
    reference_thin: EventTally

    @property
    def oriented_test_excess(self) -> float | None:
        excess = self.test_raw.excess_rate
        return None if excess is None else self.sign * excess


@dataclass(frozen=True)
class PooledEvents:
    """Pooled out-of-sample events using the per-fold TRAIN-chosen sign (sign=+1 always for the reference)."""

    raw_n: int
    thin_n: int
    oriented_excess_rate: float | None
    z: float | None
    aligned_rate_thin: float | None
    expected_rate_thin: float | None


@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DetailTable:
    """Descriptive breakdowns for one definition (full sample; not used for selection)."""

    name: str
    horizons: tuple[HorizonStat, ...]
    families: tuple[tuple[str, HorizonStat], ...]
    bins: tuple[tuple[str, tuple[tuple[str, HorizonStat], ...]], ...]
    by_month: tuple[tuple[str, int, float | None], ...]
    by_regime: tuple[tuple[str, int, float | None], ...]


@dataclass(frozen=True)
class LiquidityStudyResult:
    focus_horizon: timedelta
    atr_multiple: float
    variants_tried: int
    reference_matches_existing: bool | None
    existing_event_count: int | None
    summaries: tuple[VariantSummary, ...]
    reference: DetailTable
    selected_detail: DetailTable | None
    folds: tuple[FoldSelection, ...]
    pooled_selected: PooledEvents
    pooled_reference: PooledEvents
    selection_counts: tuple[tuple[str, int], ...]
    most_selected: str | None
    criteria: tuple[Criterion, ...]
    justifies_further_research: bool


def _pool(items: Sequence[tuple[int, EventTally, EventTally]]) -> PooledEvents:
    raw_n = sum(raw.n for _, raw, _ in items)
    thin_n = sum(t.n for _, _, t in items)
    oriented = sum(sign * raw.excess_hits for sign, raw, _ in items)
    thin_excess = sum(sign * t.excess_hits for sign, _, t in items)
    variance = sum(t.variance for _, _, t in items)
    aligned = sum(t.aligned for _, _, t in items)
    expected = sum(t.expected_aligned for _, _, t in items)
    return PooledEvents(
        raw_n=raw_n, thin_n=thin_n,
        oriented_excess_rate=oriented / raw_n if raw_n else None,
        z=thin_excess / sqrt(variance) if variance > 0 else None,
        aligned_rate_thin=aligned / thin_n if thin_n else None,
        expected_rate_thin=expected / thin_n if thin_n else None,
    )


def _labels_by_index(
    research: ExpansionResearch, horizon: timedelta, atr_multiple: float, n: int
) -> list[ExpansionLabel | None]:
    out: list[ExpansionLabel | None] = [None] * n
    for label in research.labels:
        if label.horizon == horizon and label.atr_multiple == atr_multiple:
            out[label.index] = label
    return out


def _slice_breakdown(
    events: Sequence[SweepEvent], keyfunc, labels, base_rate_all: BaseRates, k: int
) -> tuple[tuple[str, int, float | None], ...]:
    groups: dict[str, list[SweepEvent]] = {}
    for e in events:
        groups.setdefault(keyfunc(e), []).append(e)
    out = []
    for key in sorted(groups):
        raw, _ = tally_events(groups[key], labels, base_rate_all, k)
        out.append((key, raw.n, raw.excess_rate))
    return tuple(out)


def _detail(
    name: str,
    events: Sequence[SweepEvent],
    breaks: Sequence[StructuralBreak],
    labels_by_horizon: dict[timedelta, list[ExpansionLabel | None]],
    focus: timedelta,
    duration: timedelta,
    study: LiquidityStudyConfig,
    regime_by_index: Sequence[str] | None,
) -> DetailTable:
    horizons = tuple(horizon_stat(events, labels_by_horizon[h], h, duration) for h in labels_by_horizon)
    focus_labels = labels_by_horizon[focus]
    families = build_families(events, breaks, study.pairing_window)
    family_stats = tuple(
        (key, horizon_stat(points, focus_labels, focus, duration)) for key, points in families.items()
    )
    bins = []
    for dimension, groups in bin_events(events).items():
        bins.append(
            (dimension, tuple((label, horizon_stat(items, focus_labels, focus, duration)) for label, items in groups.items()))
        )
    all_base = base_rates(focus_labels, 0, len(focus_labels))
    k = focus // duration
    by_month = _slice_breakdown(events, lambda e: f"{e.timestamp.year}-{e.timestamp.month:02d}", focus_labels, all_base, k)
    by_regime = ()
    if regime_by_index is not None:
        by_regime = _slice_breakdown(events, lambda e: regime_by_index[e.index], focus_labels, all_base, k)
    return DetailTable(name, horizons, family_stats, tuple(bins), by_month, by_regime)


def run_liquidity_study(*args, **kwargs) -> "LiquidityStudyResult":
    """Run the full sweep/structure sensitivity and out-of-sample study. Deterministic; read-only.

    The garbage collector is paused while the (very large, acyclic) event lists
    are built; otherwise generational collection dominates the run time.
    """

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _run_liquidity_study(*args, **kwargs)
    finally:
        if was_enabled:
            gc.enable()


def _run_liquidity_study(
    candles: Sequence[Candle],
    research: ExpansionResearch,
    expansion: ExpansionConfig,
    study: LiquidityStudyConfig,
    walk_forward: WalkForwardConfig,
    focus_horizon: timedelta,
    atr_multiple: float,
    regime_by_index: Sequence[str] | None = None,
) -> LiquidityStudyResult:
    """Implementation of :func:`run_liquidity_study` (same arguments)."""

    if focus_horizon not in expansion.horizons or atr_multiple not in expansion.atr_multiples:
        raise ValueError("focus_horizon and atr_multiple must be configured in the expansion config.")
    if regime_by_index is not None and len(regime_by_index) != len(candles):
        raise ValueError("regime_by_index must match the candles in length.")
    duration = expansion.candle_duration
    n = len(candles)
    atrs = [s.atr for s in research.snapshots]
    labels_by_horizon = {h: _labels_by_index(research, h, atr_multiple, n) for h in expansion.horizons}
    focus_labels = labels_by_horizon[focus_horizon]
    k = focus_horizon // duration

    structures: dict[tuple[int, int, str], MarketStructure] = {}
    reference = study.reference
    for key in dict.fromkeys((*study.structures, reference.structure_key)):
        left, right, policy = key
        existing = expansion.structure
        if (existing.left_lookback, existing.right_lookback, existing.equal_policy) == key:
            structures[key] = research.structure
        else:
            structures[key] = analyze_market_structure(
                candles, replace(existing, left_lookback=left, right_lookback=right, equal_policy=policy)
            )

    definitions = study.definitions()
    fold_list = list(make_folds(candles, walk_forward, duration))
    fold_bases = [
        (
            base_rates(focus_labels, fold.train_start, fold.train_end, fold.test_from),
            base_rates(focus_labels, fold.test_start, fold.test_end),
        )
        for fold in fold_list
    ]

    # Definitions are processed one at a time so only compact per-fold tallies (and the
    # reference events) are retained; events of the selected definition are rebuilt afterwards.
    summaries_list: list[VariantSummary] = []
    train_stats: dict[str, list[tuple[float | None, int]]] = {}
    test_tallies: dict[str, list[tuple[EventTally, EventTally]]] = {}
    reference_events: tuple[SweepEvent, ...] = ()
    matches = existing_count = None
    default_sweep = expansion.sweep.min_penetration == 0.0 and not expansion.sweep.first_sweep_only
    check_existing = default_sweep and reference.structure_key == (
        expansion.structure.left_lookback, expansion.structure.right_lookback, expansion.structure.equal_policy
    )

    def process(d: SweepDefinition, keep_summary: bool) -> DetectionResult:
        detection = detect_sweeps(candles, structures[d.structure_key], atrs, d, duration, study.activity_window)
        if keep_summary:
            summaries_list.append(
                VariantSummary(
                    d.name, len(detection.events), detection.counts, detection.mean_live_swings,
                    detection.active_rate, horizon_stat(detection.events, focus_labels, focus_horizon, duration),
                )
            )
            train_stats[d.name] = []
            test_tallies[d.name] = []
            for fold, (train_base, test_base) in zip(fold_list, fold_bases):
                _, train_thin = tally_events(
                    detection.events, focus_labels, train_base, k,
                    fold.train_start, fold.train_end, fold.test_from,
                )
                train_stats[d.name].append((train_thin.excess_rate, train_thin.n))
                test_tallies[d.name].append(
                    tally_events(detection.events, focus_labels, test_base, k, fold.test_start, fold.test_end)
                )
        return detection

    for d in definitions:
        detection = process(d, True)
        if d.name == reference.name:
            reference_events = detection.events
        del detection
    if reference.name not in train_stats:
        reference_events = detect_sweeps(
            candles, structures[reference.structure_key], atrs, reference, duration, study.activity_window
        ).events
    if check_existing:
        mine = {(e.index, e.side, e.swing_index) for e in reference_events}
        theirs = {(s.sweep_index, s.side, s.swing.candle_index) for s in research.sweeps}
        matches, existing_count = mine == theirs, len(theirs)
    summaries = tuple(summaries_list)

    folds: list[FoldSelection] = []
    for position, fold in enumerate(fold_list):
        test_base = fold_bases[position][1]
        best = None
        for d in definitions:
            excess_rate, train_n = train_stats[d.name][position]
            if train_n >= study.min_events and excess_rate is not None:
                if best is None or abs(excess_rate) > abs(best[0]):
                    best = (excess_rate, d, train_n)
        if reference.name in test_tallies:
            ref_raw, ref_thin = test_tallies[reference.name][position]
        else:
            ref_raw, ref_thin = tally_events(
                reference_events, focus_labels, test_base, k, fold.test_start, fold.test_end
            )
        if best is None:
            empty = EventTally()
            folds.append(FoldSelection(fold.number, None, 1, None, 0, empty, EventTally(), ref_raw, ref_thin))
            continue
        excess, d, train_n = best
        raw, thinned = test_tallies[d.name][position]
        folds.append(
            FoldSelection(fold.number, d.name, 1 if excess > 0 else -1, excess, train_n, raw, thinned, ref_raw, ref_thin)
        )

    pooled_selected = _pool([(f.sign, f.test_raw, f.test_thin) for f in folds if f.selected])
    pooled_reference = _pool([(1, f.reference_raw, f.reference_thin) for f in folds])
    counts = Counter(f.selected for f in folds if f.selected)
    order = {d.name: i for i, d in enumerate(definitions)}
    selection_counts = tuple(sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]])))
    most_selected = selection_counts[0][0] if selection_counts else None

    def detail(definition_name: str, definition: SweepDefinition) -> DetailTable:
        if definition_name == reference.name:
            events = reference_events
        else:
            events = detect_sweeps(
                candles, structures[definition.structure_key], atrs, definition, duration, study.activity_window
            ).events
        return _detail(
            definition_name, events, structures[definition.structure_key].breaks,
            labels_by_horizon, focus_horizon, duration, study, regime_by_index,
        )

    by_name = {d.name: d for d in definitions}
    reference_detail = detail(reference.name, reference)
    selected_detail = detail(most_selected, by_name[most_selected]) if most_selected else None

    evaluated = [f for f in folds if f.selected and f.oriented_test_excess is not None]
    agreeing = sum(1 for f in evaluated if f.oriented_test_excess > 0)
    z = pooled_selected.z
    ref_excess = pooled_reference.oriented_excess_rate
    sel_excess = pooled_selected.oriented_excess_rate
    criteria = (
        Criterion(
            "out_of_sample_excess",
            z is not None and z >= study.z_threshold,
            f"pooled thinned z={'n/a' if z is None else f'{z:.2f}'} (needs >= {study.z_threshold:g}); "
            f"{len(definitions)} definitions were tried",
        ),
        Criterion(
            "fold_consistency",
            bool(evaluated) and agreeing / len(evaluated) >= study.min_fold_agreement,
            f"{agreeing}/{len(evaluated)} folds kept the train-chosen direction (needs >= {study.min_fold_agreement:.0%})",
        ),
        Criterion(
            "sample_size",
            pooled_selected.thin_n >= study.min_pooled_events,
            f"{pooled_selected.thin_n} thinned pooled test events (needs >= {study.min_pooled_events})",
        ),
        Criterion(
            "beats_existing_definition",
            sel_excess is not None and ref_excess is not None and sel_excess > ref_excess,
            f"selected OOS excess {_pp(sel_excess)} vs existing unlimited-lifetime definition {_pp(ref_excess)}",
        ),
    )
    return LiquidityStudyResult(
        focus_horizon=focus_horizon, atr_multiple=atr_multiple, variants_tried=len(definitions),
        reference_matches_existing=matches, existing_event_count=existing_count,
        summaries=summaries, reference=reference_detail, selected_detail=selected_detail,
        folds=tuple(folds), pooled_selected=pooled_selected, pooled_reference=pooled_reference,
        selection_counts=selection_counts, most_selected=most_selected, criteria=criteria,
        justifies_further_research=all(c.passed for c in criteria),
    )


def _pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}pp"
