"""Persistence for the latest generated offline research report."""

import os
from pathlib import Path
import re
import tempfile


LATEST_REPORT_PATH = Path(__file__).resolve().parents[1] / "logs" / "research-latest.txt"


def save_run_report(
    report: str,
    run_label: str,
    run_id: str,
    directory: str | Path | None = None,
) -> Path:
    """Create an exclusive, timestamped report in logs without replacing prior runs."""

    if not isinstance(report, str) or not report.strip():
        raise ValueError("report must be a non-empty string.")
    if not re.fullmatch(r"[a-z0-9-]+", run_label) or not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
        raise ValueError("run_label or run_id has an invalid format.")
    report_directory = LATEST_REPORT_PATH.parent if directory is None else Path(directory)
    destination = report_directory.resolve() / f"research-{run_label}-{run_id}.txt"
    if any(part.lower() == "historical" for part in destination.resolve().parts):
        raise ValueError("Research reports must not be written into a historical-data directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as report_file:
        report_file.write(report)
    return destination


def save_latest_report(report: str, path: str | Path = LATEST_REPORT_PATH) -> Path:
    """Atomically replace the latest report outside historical data."""

    if not isinstance(report, str) or not report.strip():
        raise ValueError("report must be a non-empty string.")
    destination = Path(path).resolve()
    if any(part.lower() == "historical" for part in destination.parts):
        raise ValueError("Research reports must not be written into a historical-data directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as report_file:
            temporary_path = Path(report_file.name)
            report_file.write(report)
            report_file.flush()
            os.fsync(report_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
