"""Local, read-only APEX status and research dashboard (standard library only)."""

import argparse
import json
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

from config.settings import SETTINGS
from research.report_store import LATEST_REPORT_PATH


HOST = "127.0.0.1"
PROJECT_ROOT = Path(__file__).resolve().parent
_AGENT_HEADER = re.compile(r"^== ([a-zA-Z0-9_-]+) \[(ok|error)\] ==$", re.MULTILINE)
_SOURCE_LINE = re.compile(r"^source: (.+)$", re.MULTILINE)
_RANGE_LINE = re.compile(r"^candles: (\d+)  range: (.+) -> (.+)$", re.MULTILINE)


def _section(report: str, agent: str) -> str:
    match = next((m for m in _AGENT_HEADER.finditer(report) if m.group(1) == agent), None)
    if match is None:
        return ""
    end = _AGENT_HEADER.search(report, match.end())
    return report[match.start():end.start() if end else len(report)].strip()


def dashboard_snapshot(root: str | Path = PROJECT_ROOT) -> dict:
    """Read only local settings and the latest saved research report, if present."""

    root_path = Path(root)
    report_path = root_path / LATEST_REPORT_PATH.relative_to(PROJECT_ROOT)
    report_available = report_path.is_file()
    report = report_path.read_text(encoding="utf-8") if report_available else ""
    generated_at = (
        datetime.fromtimestamp(report_path.stat().st_mtime, timezone.utc).isoformat()
        if report_available else None
    )
    agents = [
        {"name": match.group(1), "status": match.group(2)}
        for match in _AGENT_HEADER.finditer(report)
    ]
    errors = []
    for match in _AGENT_HEADER.finditer(report):
        if match.group(2) != "error":
            continue
        following_agent = _AGENT_HEADER.search(report, match.end())
        section_end = following_agent.start() if following_agent else len(report)
        error = next(
            (line[len("ERROR: "):].strip() for line in report[match.end():section_end].splitlines()
             if line.startswith("ERROR: ")),
            "Agent failed without a reported error message.",
        )
        errors.append(f"{match.group(1)}: {error}")
    source_match = _SOURCE_LINE.search(report)
    range_match = _RANGE_LINE.search(report)
    monthly = next(
        (line.strip() for line in report.splitlines() if "by month:" in line),
        None,
    )
    candidate_report = _section(report, "candidate_strategy")
    candidate_validation = next(
        (line.strip() for line in candidate_report.splitlines() if "Promotion gate:" in line),
        "Not available" if not candidate_report else "See candidate research details",
    )
    drawdown = next(
        (
            line.strip()
            for line in candidate_report.splitlines()
            if "Pooled out-of-sample:" in line and "max_drawdown=" in line
        ),
        "Not available",
    )
    return {
        "system": {
            "name": SETTINGS.app_name,
            "mode": "Offline research and simulation only",
            "live_trading_enabled": SETTINGS.live_trading_enabled,
            "broker": "Not connected",
            "risk_per_trade": SETTINGS.risk_per_trade,
            "max_open_positions": SETTINGS.max_open_positions,
        },
        "research": {
            "available": report_available,
            "generated_at": generated_at,
            "source": source_match.group(1) if source_match else None,
            "candle_count": int(range_match.group(1)) if range_match else None,
            "data_range": f"{range_match.group(2)} -> {range_match.group(3)}" if range_match else None,
            "agents": agents,
            "errors": errors,
            "report": report,
        },
        "candidates": {
            "validation": candidate_validation,
            "details": candidate_report or "Not available",
        },
        "simulation": {
            "positions_and_trade_history": "Not available (no persistent simulated-position or per-trade ledger).",
            "daily_performance": "Not available",
            "weekly_performance": "Not available",
            "monthly_performance": monthly or "Not available",
            "performance_source": (
                "Latest saved offline walk-forward report; costs are configured placeholders."
                if monthly else "Not available"
            ),
            "drawdown": drawdown,
        },
        "health": {
            "status": (
                "Latest research report contains agent errors."
                if errors else "Latest offline research report available."
                if report_available else "No saved research report."
            ),
            "data_freshness": (
                "Historical data range from the saved research report; no live feed."
                if range_match else "Not available; no saved research report."
            ),
        },
    }


def render_dashboard(snapshot: dict) -> str:
    system = snapshot["system"]
    research = snapshot["research"]
    simulation = snapshot["simulation"]
    agents = research["agents"]
    agent_table = (
        "<table><thead><tr><th>Agent</th><th>Status</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td>{escape(agent['name'])}</td><td>{escape(agent['status'])}</td></tr>"
            for agent in agents
        )
        + "</tbody></table>"
        if agents else "<p>Not available; run the offline research suite.</p>"
    )
    errors = (
        "<ul>" + "".join(f"<li>{escape(error)}</li>" for error in research["errors"]) + "</ul>"
        if research["errors"] else "<p>None reported.</p>" if research["available"] else "<p>Not available.</p>"
    )
    report = escape(research["report"]) if research["report"] else "Not available"
    monthly = escape(simulation["monthly_performance"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>APEX local dashboard</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#101820;color:#e8eef2}}
section{{background:#1b2a34;padding:1rem 1.25rem;margin:1rem 0;border-radius:8px}}
h1,h2{{color:#8ed1c6}}table{{border-collapse:collapse}}td,th{{padding:.4rem 1rem;border-bottom:1px solid #40515b;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:34rem;overflow:auto}}.muted{{color:#b4c3cb}}
</style></head><body>
<h1>APEX status</h1>
<p class="muted">Local read-only dashboard. It does not connect to a broker or control trading.</p>
<section><h2>System</h2>
<p>Mode: {escape(system['mode'])} · Live trading: {'ENABLED' if system['live_trading_enabled'] else 'DISABLED'} ·
Broker: {escape(system['broker'])}</p>
<p>Configured risk: {system['risk_per_trade']:.2%} per trade · Maximum open positions: {system['max_open_positions']}</p></section>
<section><h2>Research agents</h2><p>{escape(snapshot['health']['status'])}</p>
<p>Latest report timestamp: {escape(research['generated_at'] or 'Not available')}</p>
<p>Input source: {escape(research['source'] or 'Not available')} · Candles: {research['candle_count'] if research['candle_count'] is not None else 'Not available'}</p>
<p>Historical data range: {escape(research['data_range'] or 'Not available')}</p>{agent_table}</section>
<section><h2>Strategy candidates</h2><p>Validation: {escape(snapshot['candidates']['validation'])}</p>
<p>Research agents cannot approve candidates for live trading.</p>
<details><summary>Candidate research details</summary><pre>{escape(snapshot['candidates']['details'])}</pre></details></section>
<section><h2>Simulation and performance</h2>
<p>Positions and per-trade history: {escape(simulation['positions_and_trade_history'])}</p>
<p>Daily: {escape(simulation['daily_performance'])} · Weekly: {escape(simulation['weekly_performance'])}</p>
<p>Monthly performance: {monthly}</p>
<p>Performance data source: {escape(simulation['performance_source'])}</p>
<p>Drawdown: {escape(simulation['drawdown'])}</p></section>
<section><h2>System health and errors</h2><p>{escape(snapshot['health']['data_freshness'])}</p>{errors}</section>
<section><h2>Latest research results</h2><pre>{report}</pre></section>
</body></html>"""


def create_server(port: int = 8765, root: str | Path = PROJECT_ROOT) -> ThreadingHTTPServer:
    """Create an HTTP server bound only to loopback with GET-only dashboard endpoints."""

    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 to 65535.")

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/status":
                body = json.dumps(dashboard_snapshot(root), ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif self.path == "/":
                body = render_dashboard(dashboard_snapshot(root)).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self.send_error(405, "Dashboard is read-only.")

    return ThreadingHTTPServer((HOST, port), DashboardHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local, read-only APEX dashboard.")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765).")
    args = parser.parse_args(argv)
    try:
        server = create_server(args.port)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    port = server.server_address[1]
    print(f"APEX dashboard: http://{HOST}:{port}/ (read-only; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
