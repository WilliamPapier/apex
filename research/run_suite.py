"""Command-line entry point for the offline research suite (no broker).

    python -m research.run_suite <csv path> [--limit-candles N]

The input CSV is read-only. The latest rendered report is saved under ``logs``.
"""

import argparse
from datetime import datetime, timedelta, timezone
from time import perf_counter
import sys
from typing import Sequence

from research.agents import CandidateStrategyAgent, PriceExpansionAgent
from research.coordinator import load_context, run_agents
from research.report_store import save_latest_report, save_run_report
from research.study_agents import LiquidityStructureAgent, MarketRegimeAgent
from data.historical_data_loader import HistoricalDataError


def default_agents():
    """Deterministic agent order: descriptive expansion study, regimes, liquidity/structure, then candidates."""

    return [PriceExpansionAgent(), MarketRegimeAgent(), LiquidityStructureAgent(), CandidateStrategyAgent()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline APEX research suite (no broker connection).")
    parser.add_argument("path", help="Existing APEX M5 candle CSV.")
    parser.add_argument("--limit-candles", type=int, default=None, help="Use only the first N candles (quick check).")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        context = load_context(args.path, limit_candles=args.limit_candles)
    except (HistoricalDataError, ValueError, OSError) as exc:
        print(f"Cannot run research: {exc}", file=sys.stderr)
        return 2
    report = run_agents(context, default_agents())
    generated_at = datetime.now(timezone.utc)
    run_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_label = (
        "full-year"
        if args.limit_candles is None
        and report.last_timestamp - report.first_timestamp >= timedelta(days=350)
        else "research"
    )
    run_metadata = (
        f"generated_at: {generated_at.isoformat()}\n"
        f"run_id: {run_id}\n"
        f"run_runtime_seconds: {perf_counter() - started:.3f}\n"
        f"run_archive: logs\\research-{run_label}-{run_id}.txt\n\n"
    )
    report_text = run_metadata + report.to_text()
    try:
        archive_path = save_run_report(report_text, run_label, run_id)
        save_latest_report(report_text)
    except (OSError, ValueError) as exc:
        print(f"Cannot save research report: {exc}", file=sys.stderr)
        return 2
    print(f"Archived report: {archive_path}")
    print(report_text)
    return 0 if report.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
