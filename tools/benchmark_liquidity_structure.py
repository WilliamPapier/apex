"""Benchmark with ``python -m tools.benchmark_liquidity_structure``."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

from data.historical_data_loader import load_candles_csv
from research import liquidity_structure as liquidity
from research.coordinator import build_context
from research.study_agents import LiquidityStructureAgent
from research.walk_forward import make_folds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    r"data\historical\XAUUSDm_M5_20251025T000000Z_20261025T000000Z.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_digest(events) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(repr(event).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fold_data(fold) -> dict:
    return {
        "number": fold.number,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_from": fold.train_from.isoformat(),
        "train_until": fold.train_until.isoformat(),
        "test_from": fold.test_from.isoformat(),
        "test_until": fold.test_until.isoformat(),
    }


def _study_data(study, folds) -> dict:
    return {
        "variants_tried": study.variants_tried,
        "reference_matches_existing": study.reference_matches_existing,
        "existing_event_count": study.existing_event_count,
        "summaries": [
            {
                "name": item.name,
                "events": item.events,
                "counts": asdict(item.counts),
                "mean_live_swings": item.mean_live_swings,
                "active_rate": item.active_rate,
            }
            for item in study.summaries
        ],
        "folds": [
            {
                "number": item.number,
                "selected": item.selected,
                "sign": item.sign,
                "train_excess": item.train_excess,
                "train_events": item.train_events,
                "test_raw": asdict(item.test_raw),
                "test_thin": asdict(item.test_thin),
                "reference_raw": asdict(item.reference_raw),
                "reference_thin": asdict(item.reference_thin),
            }
            for item in study.folds
        ],
        "pooled_selected": asdict(study.pooled_selected),
        "pooled_reference": asdict(study.pooled_reference),
        "selection_counts": study.selection_counts,
        "most_selected": study.most_selected,
        "criteria": [asdict(item) for item in study.criteria],
        "justifies_further_research": study.justifies_further_research,
        "fold_boundaries": [_fold_data(fold) for fold in folds],
    }


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_exclusive(path: Path, data: dict) -> None:
    encoded = json.dumps(data, indent=2, allow_nan=False, default=_json_default)
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(encoded)
        destination.write("\n")


def _benchmark(dataset: Path, label: str) -> Path:
    candles = load_candles_csv(dataset)
    context_started = perf_counter()
    context = build_context(candles, source=str(dataset))
    context_seconds = perf_counter() - context_started
    config = context.config
    folds = make_folds(candles, config.walk_forward, config.expansion.candle_duration)

    calls = []
    timings = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    original_functions = {}

    def instrument(name, function, on_result=None):
        def measured(*args, **kwargs):
            started = perf_counter()
            try:
                result = function(*args, **kwargs)
            finally:
                timings[name]["calls"] += 1
                timings[name]["seconds"] += perf_counter() - started
            if on_result is not None:
                on_result(args, result)
            return result

        return measured

    def record_detection(args, detection):
        definition = args[3]
        calls.append(
            {
                "definition": definition.name,
                "event_count": len(detection.events),
                "crossings": asdict(detection.counts),
                "event_sha256": _event_digest(detection.events),
            }
        )

    for name in (
        "detect_sweeps",
        "analyze_market_structure",
        "tally_events",
        "_detail",
        "build_families",
        "bin_events",
    ):
        original_functions[name] = getattr(liquidity, name)
        setattr(
            liquidity,
            name,
            instrument(
                name,
                original_functions[name],
                record_detection if name == "detect_sweeps" else None,
            ),
        )

    study = None
    agent_started = perf_counter()
    try:
        result = LiquidityStructureAgent().run(context)
        if result.status != "ok":
            raise RuntimeError(result.error or result.headline)
        study = result.data["study"]
    finally:
        agent_seconds = perf_counter() - agent_started
        for name, function in original_functions.items():
            setattr(liquidity, name, function)

    parameters = {
        "expansion": {
            "candle_duration_seconds": config.expansion.candle_duration.total_seconds(),
            "horizons_seconds": [value.total_seconds() for value in config.expansion.horizons],
            "atr_multiples": config.expansion.atr_multiples,
            "structure": asdict(config.expansion.structure),
            "sweep": asdict(config.expansion.sweep),
        },
        "walk_forward": {
            "train_span_seconds": config.walk_forward.train_span.total_seconds(),
            "test_span_seconds": config.walk_forward.test_span.total_seconds(),
            "step_seconds": config.walk_forward.step.total_seconds(),
            "embargo_seconds": config.walk_forward.embargo.total_seconds(),
            "min_train_trades": config.walk_forward.min_train_trades,
        },
        "liquidity_study": {
            "structures": config.liquidity_study.structures,
            "lifetimes_seconds": [
                None if value is None else value.total_seconds()
                for value in config.liquidity_study.lifetimes
            ],
            "first_only_options": config.liquidity_study.first_only_options,
            "min_penetration_atrs": config.liquidity_study.min_penetration_atrs,
            "pairing_window": config.liquidity_study.pairing_window,
            "activity_window": config.liquidity_study.activity_window,
            "min_events": config.liquidity_study.min_events,
            "min_pooled_events": config.liquidity_study.min_pooled_events,
            "min_fold_agreement": config.liquidity_study.min_fold_agreement,
            "z_threshold": config.liquidity_study.z_threshold,
        },
        "focus_horizon_seconds": study.focus_horizon.total_seconds(),
        "atr_multiple": study.atr_multiple,
    }
    data = {
        "label": label,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "candle_count": len(candles),
        "first_timestamp": candles[0].timestamp.isoformat(),
        "last_timestamp": candles[-1].timestamp.isoformat(),
        "parameters": parameters,
        "input_preparation_seconds": context_seconds,
        "liquidity_agent_seconds": agent_seconds,
        "function_timings": dict(timings),
        "study": _study_data(study, folds),
        "study_repr_sha256": hashlib.sha256(repr(study).encode("utf-8")).hexdigest(),
        "detections": calls,
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "logs" / f"liquidity-benchmark-{label}-{run_id}.json"
    _write_json_exclusive(output, data)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--label", default="manual")
    args = parser.parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    if not dataset.is_file():
        parser.error(f"dataset does not exist: {dataset}")
    output = _benchmark(dataset, args.label)
    print(f"Benchmark report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
