"""Small, dependency-free statistics helpers shared by the research studies.

Research only. Nothing here touches a broker or data files.

Overlapping forward windows make neighbouring observations strongly
dependent, so naive sample sizes overstate the evidence. ``thin`` keeps
observations at least ``min_gap`` candles apart; significance statistics in
the studies are computed on thinned samples (raw counts are reported too).
"""

from math import sqrt
from typing import Sequence


def thin(indices: Sequence[int], min_gap: int) -> list[int]:
    """Greedy chronological thinning: keep an index only if it is >= min_gap after the last kept one."""

    if type(min_gap) is not int or min_gap < 1:
        raise ValueError("min_gap must be an integer of at least 1.")
    kept: list[int] = []
    previous: int | None = None
    for index in indices:
        if previous is not None and index <= previous:
            raise ValueError("indices must be strictly ascending.")
        previous = index
        if not kept or index - kept[-1] >= min_gap:
            kept.append(index)
    return kept


def rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def z_two_proportion(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """Pooled two-proportion z for (k1/n1) - (k2/n2); None when undefined."""

    if n1 <= 0 or n2 <= 0:
        return None
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if variance <= 0:
        return None
    return (k1 / n1 - k2 / n2) / sqrt(variance)


def z_one_proportion(hits: int, total: int, expected: float) -> float | None:
    """z for an observed proportion against a fixed expected proportion; None when undefined."""

    if total <= 0 or not 0 < expected < 1:
        return None
    return (hits / total - expected) / sqrt(expected * (1 - expected) / total)


def z_against_expected(hits: float, expected_hits: float, variance: float) -> float | None:
    """z for a hit count against per-observation expected probabilities (variance = sum p(1-p))."""

    if variance <= 0:
        return None
    return (hits - expected_hits) / sqrt(variance)


def run_lengths(states: Sequence[str | None], breaks_before: Sequence[bool] | None = None) -> list[tuple[str, int]]:
    """Lengths of consecutive equal states. ``None`` or a ``breaks_before`` flag ends a run."""

    if breaks_before is not None and len(breaks_before) != len(states):
        raise ValueError("breaks_before must match states in length.")
    runs: list[tuple[str, int]] = []
    current: str | None = None
    length = 0
    for i, state in enumerate(states):
        broken = breaks_before is not None and breaks_before[i]
        if state is None or broken or state != current:
            if current is not None and length:
                runs.append((current, length))
            current, length = (state, 0) if state is not None else (None, 0)
        if state is not None:
            length += 1
    if current is not None and length:
        runs.append((current, length))
    return runs


def state_changes(states: Sequence[str | None], breaks_before: Sequence[bool] | None = None) -> tuple[int, int]:
    """(changes, adjacent classified pairs) between consecutive, contiguous, classified candles."""

    changes = pairs = 0
    for i in range(1, len(states)):
        if states[i] is None or states[i - 1] is None:
            continue
        if breaks_before is not None and breaks_before[i]:
            continue
        pairs += 1
        changes += states[i] != states[i - 1]
    return changes, pairs
