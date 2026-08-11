"""Read-only local web dashboard for TradingBot SQLite research data."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from trading_bot.dashboard.repository import DashboardRepository, DashboardScanRow
from trading_bot.persistence.sqlite_store import DEFAULT_DATABASE_PATH


class DashboardApplication:
    def __init__(
        self,
        *,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        current_cache_dir: str | Path = "data/current",
    ) -> None:
        self.repository = DashboardRepository(database_path, current_cache_dir=current_cache_dir)

    def handle_path(self, path: str) -> tuple[int, str, bytes]:
        parsed = urlparse(path)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items() if values}
        try:
            if parsed.path == "/api/overview":
                return _json_response(asdict(self.repository.overview()))
            if parsed.path == "/api/scanner/latest":
                return _json_response([_scan_row_dict(row) for row in self.repository.latest_scan_rows()])
            if parsed.path in {"", "/"}:
                return _html_response(self._overview_page())
            if parsed.path == "/scanner":
                return _html_response(self._scanner_page())
            if parsed.path == "/candidate":
                return _html_response(self._candidate_page(query))
            if parsed.path == "/forward":
                return _html_response(self._forward_page())
            if parsed.path == "/history":
                return _html_response(self._history_page(query))
            if parsed.path == "/system":
                return _html_response(self._system_page())
            if parsed.path == "/static/dashboard.css":
                return HTTPStatus.OK, "text/css; charset=utf-8", CSS.encode("utf-8")
        except Exception as exc:
            return _html_response(_layout("Dashboard Error", f"<div class='card error'>{_escape(str(exc))}</div>"), status=500)
        return _html_response(_layout("Not Found", "<div class='card'>Page not found.</div>"), status=404)

    def _overview_page(self) -> str:
        overview = self.repository.overview()
        latest_rows = self.repository.latest_scan_rows()
        forward = self.repository.forward_stats()[0]
        performance = self.repository.performance_summary()
        equity_points = self.repository.equity_curve()
        open_positions = self.repository.open_positions()
        activities = self.repository.recent_activity()
        broker = self.repository.broker_status()
        risk = _risk_profile_summary(overview.risk_profile)
        body = (
            "<section class='hero compact'><div><p class='eyebrow'>Lokal forskningsdashboard</p>"
            "<h1>TradingBot översikt</h1>"
            "<p>Simulerade/papper-resultat från sparade AI-beslut. Livehandel är avstängd.</p></div>"
            f"{_status_badge('ONLINE')}</section>"
            "<section class='priority-grid'>"
            + _metric_card("Portföljvärde", _fmt_sek(performance.portfolio_value), "Simulerat/papper")
            + _metric_card(
                "Dagens P&L",
                f"{_fmt_sek(performance.today_pnl_sek)} / {_fmt_percent(performance.today_pnl_pct)}",
                "Realiserat simulerat resultat",
            )
            + _metric_card(
                "30 dagar P&L",
                f"{_fmt_sek(performance.last_30_days_pnl_sek)} / {_fmt_percent(performance.last_30_days_pnl_pct)}",
                "Realiserat simulerat resultat",
            )
            + _metric_card(
                "All-time P&L",
                f"{_fmt_sek(performance.all_time_pnl_sek)} / {_fmt_percent(performance.all_time_pnl_pct)}",
                f"Realiserat {_fmt_sek(performance.realized_pnl_sek)} · Orealiserat {_fmt_sek(performance.unrealized_pnl_sek)}",
            )
            + _metric_card("Exponering", _fmt_percent(performance.current_exposure_pct, signed=False), "Nuvarande sparad portföljexponering")
            + _metric_card("Riskläge", risk["mode"], risk["details"])
            + _metric_card(
                "Bot / marknadsdata",
                "Aktuell" if overview.latest_market_data_candle else "Ingen cache",
                f"Senaste candle: {_fmt_timestamp(overview.latest_market_data_candle)}",
            )
            + "</section>"
            "<section class='card'><div class='section-heading'><div><h2>Equitykurva</h2>"
            "<p>Simulerat/papper. Filter är visuella tills mer historik finns.</p></div>"
            "<div class='range-tabs'><button>1D</button><button>7D</button><button>30D</button><button>3M</button><button>ALL</button></div></div>"
            + _equity_curve(equity_points)
            + "</section>"
            "<section class='overview-layout'>"
            "<div class='main-panel card'><div class='section-heading'><div><h2>Dagens AI-beslut</h2>"
            f"<p>Senaste scan: {_fmt_timestamp(overview.latest_scan_timestamp)}</p></div>"
            f"<a class='button' href='/scanner'>Öppna scanner</a></div>"
            + f"<div id='today-decisions'>{_decision_cards(latest_rows[:3])}</div>"
            + "</div>"
            "<aside class='side-panel card'><h2>Forwardtest</h2>"
            + _forward_summary(forward)
            + "<a class='button secondary' href='/forward'>Visa detaljer</a></aside>"
            "</section>"
            "<section class='overview-layout lower'>"
            "<div class='card'><h2>Öppna positioner</h2>"
            + _open_positions(open_positions)
            + "</div>"
            "<div class='card'><h2>Broker</h2>"
            + _broker_section(broker)
            + "</div></section>"
            "<section class='card'><h2>Senaste aktivitet</h2>"
            + _recent_activity(activities)
            + "</section>"
            + AUTO_REFRESH_SCRIPT
        )
        return _layout("Översikt", body, active="/")

    def _scanner_page(self) -> str:
        rows = self.repository.latest_scan_rows()
        body = (
            "<section class='page-title'><h1>Dagens AI-scanner</h1>"
            "<p>Sparade aktuella scanresultat. Sidan anropar inte OpenAI.</p></section>"
            "<div id='scanner-table'>"
            + _scanner_table(rows)
            + "</div>"
            + SCANNER_REFRESH_SCRIPT
        )
        return _layout("Dagens AI-scanner", body, active="/scanner")

    def _candidate_page(self, query: dict[str, str]) -> str:
        scan_timestamp = query.get("scan", "")
        symbol = query.get("symbol", "")
        row = self.repository.candidate(scan_timestamp=scan_timestamp, symbol=symbol)
        if row is None:
            return _layout("Candidate", "<div class='card'>Candidate not found.</div>", active="/scanner")
        values = row.normalized_snapshot.get("values", {})
        units = row.normalized_snapshot.get("units", {})
        feature_lines = "".join(
            f"<tr><td>{_escape(key)}</td><td>{_escape(str(value))}</td><td>{_escape(str(units.get(key, '')))}</td></tr>"
            for key, value in values.items()
        )
        body = (
            f"<section class='page-title'><h1>{_escape(_company_name(row.symbol))}</h1>"
            f"<p>{_escape(row.symbol)} · Beslutstid: {_fmt_timestamp(row.decision_timestamp)}</p></section>"
            "<section class='grid'>"
            + _metric_card("XGBoost-rank", str(row.xgboost_rank))
            + _metric_card("XGBoost-sannolikhet", _fmt_percent(row.xgboost_probability_pct, signed=False))
            + _metric_card("OpenAI-beslut", _badge(row.openai_decision))
            + _metric_card("OpenAI-konfidens", _fmt_optional_pct(row.openai_confidence_pct))
            + "</section>"
            f"<section class='card'><h2>OpenAI-sammanfattning</h2><p>{_escape(row.summary or 'Ingen OpenAI-sammanfattning sparad.')}</p></section>"
            "<section class='split'>"
            + _list_card("Positiva faktorer", row.positive_factors)
            + _list_card("Negativa faktorer", row.negative_factors)
            + _list_card("Riskflaggor", row.risk_flags)
            + "</section>"
            "<section class='card'><h2>Teknisk diagnostik</h2>"
            f"<table><thead><tr><th>Feature</th><th>Värde</th><th>Enhet</th></tr></thead><tbody>{feature_lines}</tbody></table>"
            "</section>"
            + _sources_card(row.sources)
        )
        return _layout(row.symbol, body, active="/scanner")

    def _forward_page(self) -> str:
        rows = self.repository.forward_stats()
        invalid = self.repository.system_status().invalid_stale_records
        body = (
            "<section class='page-title'><h1>Forwardtest</h1>"
            "<p>Simulerade utfall för sparade beslut. Inaktuella beslut är exkluderade.</p></section>"
            + _forward_table(rows, invalid)
            + _forward_chart(rows)
        )
        return _layout("Forwardtest", body, active="/forward")

    def _history_page(self, query: dict[str, str]) -> str:
        rows = self.repository.history(
            symbol=query.get("symbol") or None,
            decision=query.get("decision") or None,
            status=query.get("status") or None,
            date_text=query.get("date") or None,
        )
        body = (
            "<section class='page-title'><h1>Historik</h1>"
            "<p>Skrivskyddad historik över giltiga sparade AI-beslut.</p></section>"
            + _history_filters(query)
            + _history_table(rows)
        )
        return _layout("Historik", body, active="/history")

    def _system_page(self) -> str:
        status = self.repository.system_status()
        broker = self.repository.broker_status()
        body = (
            "<section class='page-title'><h1>System</h1>"
            "<p>Driftstatus för lokal forskningsdata och cache.</p></section>"
            "<section class='grid'>"
            + _metric_card("Databasstatus", "OK" if status.database_exists else "Saknas")
            + _metric_card("Databasväg", str(status.database_path))
            + _metric_card("Marknadscache", f"{status.current_market_cache_files} CSV-filer")
            + _metric_card("Senaste candle", _fmt_timestamp(status.latest_market_data_candle))
            + _metric_card("OpenAI-modell", status.openai_model)
            + _metric_card("Sparade scans", str(status.stored_scans))
            + _metric_card("Väntande forwardtester", str(status.pending_forward_tests))
            + _metric_card("Inaktuella/ogiltiga poster", str(status.invalid_stale_records))
            + _metric_card("Livehandel", status.live_trading_status)
            + _metric_card("Broker", broker.name)
            + _metric_card("Brokerstatus", broker.connection_status)
            + _metric_card("Brokerkonto", broker.account_id_masked)
            + _metric_card("Tradingrättigheter", broker.trading_permissions)
            + "</section>"
        )
        return _layout("System", body, active="/system")


def run_dashboard(
    *,
    host: str = "localhost",
    port: int = 8000,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    current_cache_dir: str | Path = "data/current",
) -> None:
    application = DashboardApplication(database_path=database_path, current_cache_dir=current_cache_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, content_type, body = application.handle_path(self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"TradingBot dashboard running at http://{host}:{port}")
    print("Live trading: LOCKED / UNAVAILABLE")
    server.serve_forever()


def _layout(title: str, body: str, *, active: str = "") -> str:
    links = [
        ("/", "Översikt"),
        ("/scanner", "AI-scanner"),
        ("/forward", "Forwardtest"),
        ("/history", "Historik"),
        ("/system", "System"),
    ]
    navigation = "".join(
        f"<a class='{'active' if href == active else ''}' href='{href}'>{label}</a>" for href, label in links
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>TradingBot - {_escape(title)}</title>"
        "<link rel='stylesheet' href='/static/dashboard.css'></head><body>"
        "<aside><div class='brand'>TradingBot</div><nav>"
        f"{navigation}</nav><div class='lock'>Livehandel: LOCKED</div></aside>"
        f"<main>{body}</main></body></html>"
    )


def _scanner_table(rows: list[DashboardScanRow]) -> str:
    if not rows:
        return "<div class='card empty'>Inga sparade scanresultat ännu. Kör run_ai_scan.bat först.</div>"
    body = ""
    for row in rows:
        detail_url = "/candidate?" + urlencode({"scan": row.scan_timestamp, "symbol": row.symbol})
        values = row.normalized_snapshot.get("values", {})
        body += (
            "<tr>"
            f"<td><a href='{detail_url}'>{_escape(_company_name(row.symbol))}</a><span class='ticker'>{_escape(row.symbol)}</span></td>"
            f"<td>{_fmt_percent(row.xgboost_probability_pct, signed=False)}</td>"
            f"<td>{_fmt_optional_pct(row.openai_confidence_pct)}</td>"
            f"<td>{_escape(row.sentiment or 'ej analyserad')}</td>"
            f"<td>{_escape(row.market_regime or 'ej analyserad')}</td>"
            f"<td>{_badge(row.openai_decision)}</td>"
            "</tr>"
        )
    return (
        "<section class='card'><table><thead><tr><th>Bolag / ticker</th><th>XGBoost-sannolikhet</th>"
        "<th>OpenAI-konfidens</th><th>Sentiment</th><th>Trend</th><th>Beslut</th></tr></thead><tbody>"
        f"{body}</tbody></table></section>"
    )


def _forward_table(rows: list[Any], invalid_stale_records: int) -> str:
    body = ""
    for row in rows:
        body += (
            "<tr>"
            f"<td>{_escape(_group_label(row.label))}</td><td>{row.pending_decisions}</td><td>{row.completed_trades}</td>"
            f"<td>{row.wins}</td><td>{row.losses}</td><td>{_fmt_percent(row.win_rate * Decimal('100'), signed=False)}</td>"
            f"<td>{_fmt_percent(row.average_return_pct)}</td><td>{_fmt_percent(row.median_return_pct)}</td>"
            f"<td>{_fmt_sek(row.total_simulated_pnl_sek)}</td>"
            "</tr>"
        )
    return (
        f"<section class='card'><div class='muted'>Exkluderade inaktuella beslut: {invalid_stale_records}</div>"
        "<table><thead><tr><th>Grupp</th><th>Väntande</th><th>Klara</th><th>Vinster</th><th>Förluster</th>"
        "<th>Vinstgrad</th><th>Snittavkastning</th><th>Medianavkastning</th><th>Simulerad P&L</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def _forward_chart(rows: list[Any]) -> str:
    if not any(row.completed_trades for row in rows):
        return "<section class='card empty'><h2>Resultatgraf</h2><p>Inga klara forwardtest-affärer ännu.</p></section>"
    bars = ""
    for index, row in enumerate(rows):
        width = max(4, min(100, abs(float(row.average_return_pct))))
        color = "#22c55e" if row.average_return_pct >= 0 else "#ef4444"
        bars += (
            f"<text x='10' y='{30 + index * 36}' fill='#cbd5e1'>{_escape(row.label)}</text>"
            f"<rect x='180' y='{14 + index * 36}' width='{width * 4}' height='18' fill='{color}' rx='5'></rect>"
        )
    return f"<section class='card'><h2>Snittavkastning</h2><svg viewBox='0 0 640 170'>{bars}</svg></section>"


def _history_filters(query: dict[str, str]) -> str:
    return (
        "<form class='filters' method='get'>"
        f"<input name='symbol' placeholder='Ticker' value='{_escape(query.get('symbol', ''))}'>"
        f"<select name='decision'>{_option('', 'Alla beslut', query.get('decision', ''))}"
        f"{_option('APPROVE', 'APPROVE', query.get('decision', ''))}"
        f"{_option('WATCH', 'WATCH', query.get('decision', ''))}"
        f"{_option('REJECT', 'REJECT', query.get('decision', ''))}</select>"
        f"<select name='status'>{_option('', 'Alla statusar', query.get('status', ''))}"
        f"{_option('PENDING', 'PENDING', query.get('status', ''))}"
        f"{_option('COMPLETED', 'COMPLETED', query.get('status', ''))}</select>"
        f"<input name='date' placeholder='YYYY-MM-DD' value='{_escape(query.get('date', ''))}'>"
        "<button type='submit'>Filtrera</button></form>"
    )


def _history_table(rows: list[Any]) -> str:
    if not rows:
        return "<section class='card empty'>Inga giltiga sparade beslut matchar filtren.</section>"
    body = ""
    for row in rows:
        body += (
            "<tr>"
            f"<td>{_fmt_timestamp(str(row['scan_timestamp']))}</td><td>{_escape(_company_name(str(row['symbol'])))}<span class='ticker'>{_escape(str(row['symbol']))}</span></td>"
            f"<td>{_escape(str(row['xgboost_rank']))}</td><td>{_fmt_percent(Decimal(str(row['xgboost_probability_pct'])), signed=False)}</td>"
            f"<td>{_badge(str(row['openai_decision']))}</td><td>{_escape(str(row['forward_status']))}</td>"
            f"<td>{_fmt_percent(Decimal(str(row['net_return_pct']))) if row['net_return_pct'] is not None else 'pending'}</td>"
            "</tr>"
        )
    return (
        "<section class='card'><table><thead><tr><th>Scan</th><th>Bolag / ticker</th><th>Rank</th>"
        "<th>XGBoost</th><th>OpenAI</th><th>Status</th><th>Nettoavkastning</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def _list_card(title: str, values: list[str]) -> str:
    items = "".join(f"<li>{_escape(value)}</li>" for value in values) or "<li>Inget sparat.</li>"
    return f"<section class='card'><h2>{_escape(title)}</h2><ul>{items}</ul></section>"


def _sources_card(sources: list[dict[str, str]]) -> str:
    if not sources:
        return "<section class='card'><h2>Källor</h2><p>Inga källor sparade.</p></section>"
    items = "".join(
        f"<li><a href='{_escape(source.get('url', '#'))}' target='_blank'>{_escape(source.get('title') or source.get('url', 'source'))}</a></li>"
        for source in sources
    )
    return f"<section class='card'><h2>Källor</h2><ul>{items}</ul></section>"


def _decision_cards(rows: list[DashboardScanRow]) -> str:
    if not rows:
        return "<div class='empty'>Inga sparade AI-beslut ännu. Kör run_ai_scan.bat först.</div>"
    cards = ""
    for row in rows:
        detail_url = "/candidate?" + urlencode({"scan": row.scan_timestamp, "symbol": row.symbol})
        cards += (
            "<article class='decision-card'>"
            "<div class='decision-top'>"
            f"<div><h3>{_escape(_company_name(row.symbol))}</h3><span class='muted'>{_escape(row.symbol)} · Rank #{row.xgboost_rank}</span></div>"
            f"{_badge(row.openai_decision)}"
            "</div>"
            f"<p>{_escape(_short_summary(row.summary))}</p>"
            "<div class='decision-metrics'>"
            f"<span>XGB <b>{_fmt_percent(row.xgboost_probability_pct, signed=False)}</b></span>"
            f"<span>AI-konfidens <b>{_fmt_optional_pct(row.openai_confidence_pct)}</b></span>"
            f"<span>{_escape(row.sentiment or 'ej analyserad')} / {_escape(row.market_regime or 'ej analyserad')}</span>"
            "</div>"
            f"<a class='button' href='{detail_url}'>Full analys</a>"
            "</article>"
        )
    return f"<div class='decision-grid'>{cards}</div>"


def _equity_curve(points: list[Any]) -> str:
    if len(points) <= 1:
        return "<div class='empty chart-empty'>Ingen equitykurva ännu. Klara simulerade forwardtest-affärer krävs.</div>"
    max_value = max(point.portfolio_value for point in points)
    min_value = min(point.portfolio_value for point in points)
    span = max(max_value - min_value, Decimal("1"))
    chart_width = Decimal("720")
    chart_height = Decimal("180")
    coordinates = []
    for index, point in enumerate(points):
        x_position = (Decimal(index) / Decimal(max(len(points) - 1, 1))) * chart_width
        y_position = chart_height - ((point.portfolio_value - min_value) / span) * Decimal("150") - Decimal("15")
        coordinates.append(f"{x_position:.2f},{y_position:.2f}")
    return (
        "<svg class='equity-chart' viewBox='0 0 720 190' role='img' aria-label='Simulerad equitykurva'>"
        "<line x1='0' y1='175' x2='720' y2='175' stroke='#263447'/>"
        f"<polyline points='{' '.join(coordinates)}' fill='none' stroke='#38bdf8' stroke-width='3'/>"
        "</svg>"
    )


def _open_positions(rows: list[Any]) -> str:
    if not rows:
        return (
            "<div class='empty'>Inga öppna simulerade positioner finns sparade. "
            "Dashboarden skapar inte positioner och visar inte fejkdatan.</div>"
        )
    body = "".join(
        "<tr>"
        f"<td>{_escape(str(row['company_name']))}</td>"
        f"<td>{_escape(str(row['symbol']))}</td>"
        f"<td>{_escape(str(row['entry_price']))}</td>"
        f"<td>{_escape(str(row['current_price']))}</td>"
        f"<td>{_escape(str(row['unrealized_pnl']))}</td>"
        f"<td>{_escape(str(row['stop']))}</td>"
        f"<td>{_escape(str(row['holding_day']))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Bolag</th><th>Ticker</th><th>Entry</th><th>Nuvarande pris</th>"
        "<th>Orealiserad P&L</th><th>Stop</th><th>Innehavsdag / max</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _recent_activity(rows: list[Any]) -> str:
    if not rows:
        return "<div class='empty'>Ingen sparad aktivitet ännu.</div>"
    return "<div class='activity-list'>" + "".join(
        "<div class='activity-item'>"
        f"<span>{_fmt_timestamp(row.timestamp)}</span>"
        f"<strong>{_escape(row.title)}</strong>"
        f"<p>{_escape(row.detail)}</p>"
        "</div>"
        for row in rows
    ) + "</div>"


def _broker_section(broker: Any) -> str:
    status_class = "current" if broker.connection_status == "CONNECTED" else "locked"
    return (
        "<div class='summary-list'>"
        f"<div><span>Miljö</span><strong>{_escape(broker.name)}</strong></div>"
        f"<div><span>Status</span><strong><span class='status {status_class}'>{_escape(broker.connection_status)}</span></strong></div>"
        f"<div><span>Konto</span><strong>{_escape(broker.account_id_masked)}</strong></div>"
        f"<div><span>Kontovärde</span><strong>{_fmt_optional_sek(broker.account_value)}</strong></div>"
        f"<div><span>Cash</span><strong>{_fmt_optional_sek(broker.cash)}</strong></div>"
        f"<div><span>Positioner</span><strong>{broker.positions_count}</strong></div>"
        f"<div><span>Synk</span><strong>{_escape(broker.broker_sync_status)}</strong></div>"
        f"<div><span>Tradingrättigheter</span><strong>{_escape(broker.trading_permissions)}</strong></div>"
        "</div>"
    )


def _forward_summary(row: Any) -> str:
    return (
        "<div class='summary-list'>"
        f"<div><span>Väntande</span><strong>{row.pending_decisions}</strong></div>"
        f"<div><span>Klara</span><strong>{row.completed_trades}</strong></div>"
        f"<div><span>Vinstgrad</span><strong>{_fmt_percent(row.win_rate * Decimal('100'), signed=False)}</strong></div>"
        f"<div><span>P&L</span><strong>{_fmt_sek(row.total_simulated_pnl_sek)}</strong></div>"
        "</div>"
    )


def _metric_card(label: str, value: str, hint: str = "") -> str:
    hint_html = f"<small>{_escape(hint)}</small>" if hint else ""
    return f"<article class='metric'><span>{_escape(label)}</span><strong>{value}</strong>{hint_html}</article>"


def _status_badge(value: str) -> str:
    return f"<span class='status {value.lower()}'>{_escape(value)}</span>"


def _badge(value: str | None) -> str:
    label = value or "ej analyserad"
    css = label.lower().replace(" ", "-")
    return f"<span class='badge {css}'>{_escape(label)}</span>"


def _option(value: str, label: str, selected: str) -> str:
    return f"<option value='{value}' {'selected' if value == selected else ''}>{label}</option>"


def _scan_row_dict(row: DashboardScanRow) -> dict[str, Any]:
    payload = asdict(row)
    payload["company_name"] = _company_name(row.symbol)
    return json.loads(json.dumps(payload, default=str))


def _json_response(payload: Any) -> tuple[int, str, bytes]:
    return HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, default=str).encode("utf-8")


def _html_response(body: str, *, status: int = 200) -> tuple[int, str, bytes]:
    return status, "text/html; charset=utf-8", body.encode("utf-8")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_decimal(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def _fmt_optional_pct(value: Decimal | None) -> str:
    return _fmt_percent(value, signed=False) if value is not None else "ej analyserad"


def _fmt_percent(value: Decimal, *, signed: bool = True) -> str:
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{_fmt_decimal(value)}%"


def _fmt_sek(value: Decimal) -> str:
    return f"{_fmt_decimal(value)} SEK"


def _fmt_optional_sek(value: Decimal | None) -> str:
    return _fmt_sek(value) if value is not None else "n/a"


def _fmt_timestamp(value: str | None) -> str:
    if not value:
        return "Not available"
    try:
        timestamp = value if hasattr(value, "strftime") else datetime.fromisoformat(str(value))
        return timestamp.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return _escape(value)


def _risk_profile_summary(raw: str) -> dict[str, str]:
    mode = str(raw).split(";", maxsplit=1)[0].strip() or "MEDIUM"
    values = _risk_profile_values(mode)
    return {
        "mode": mode,
        "details": (
            f"Risk/trade: {values['risk']} · Max exponering: {values['exposure']} · "
            f"Max positioner: {values['positions']} · Max drawdown: {values['drawdown']} · Leverage: Av"
        ),
    }


def _risk_profile_values(mode: str) -> dict[str, str]:
    profiles = {
        "LOW": {"risk": "0.5%", "exposure": "30%", "positions": "2", "drawdown": "8%"},
        "MEDIUM": {"risk": "1%", "exposure": "60%", "positions": "4", "drawdown": "15%"},
        "HIGH": {"risk": "2%", "exposure": "100%", "positions": "6", "drawdown": "25%"},
    }
    return profiles.get(mode.upper(), profiles["MEDIUM"])


def _short_summary(value: str | None) -> str:
    if not value:
        return "Ingen AI-sammanfattning sparad för kandidaten."
    text = " ".join(value.split())
    return text if len(text) <= 180 else f"{text[:177]}..."


def _company_name(symbol: str) -> str:
    return COMPANY_NAMES.get(symbol, symbol)


def _group_label(label: str) -> str:
    labels = {
        "ALL XGBoost top-3": "Alla XGBoost topp 3",
        "OpenAI APPROVE": "OpenAI APPROVE",
        "OpenAI WATCH": "OpenAI WATCH",
        "OpenAI REJECT": "OpenAI REJECT",
    }
    return labels.get(label, label)


COMPANY_NAMES = {
    "SAND.ST": "Sandvik",
    "ERIC-B.ST": "Ericsson B",
    "SHB-A.ST": "Handelsbanken A",
    "VOLV-B.ST": "Volvo B",
    "SEB-A.ST": "SEB A",
    "SWED-A.ST": "Swedbank A",
    "INVE-B.ST": "Investor B",
    "HM-B.ST": "H&M B",
    "ATCO-A.ST": "Atlas Copco A",
    "ASSA-B.ST": "ASSA ABLOY B",
}


AUTO_REFRESH_SCRIPT = """
<script>
function esc(value){
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function pct(value){
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)}%` : 'ej analyserad';
}
function decisionBadge(value){
  const label = value || 'ej analyserad';
  return `<span class="badge ${label.toLowerCase().replaceAll(' ', '-')}">${esc(label)}</span>`;
}
function shortSummary(value){
  const text = String(value || 'Ingen AI-sammanfattning sparad för kandidaten.').replace(/\\s+/g, ' ').trim();
  return text.length > 180 ? `${text.slice(0,177)}...` : text;
}
async function refreshOverview(){
  const [overview, scanner] = await Promise.all([fetch('/api/overview'), fetch('/api/scanner/latest')]);
  if(!overview.ok || !scanner.ok) return;
  const rows = await scanner.json();
  const target = document.querySelector('#today-decisions');
  if(!target) return;
  target.innerHTML = rows.length
    ? `<div class="decision-grid">${rows.slice(0,3).map(r => `<article class="decision-card"><div class="decision-top"><div><h3>${esc(r.company_name || r.symbol)}</h3><span class="muted">${esc(r.symbol)} · Rank #${esc(r.xgboost_rank)}</span></div>${decisionBadge(r.openai_decision)}</div><p>${esc(shortSummary(r.summary))}</p><div class="decision-metrics"><span>XGB <b>${pct(r.xgboost_probability_pct)}</b></span><span>AI-konfidens <b>${pct(r.openai_confidence_pct)}</b></span><span>${esc(r.sentiment || 'ej analyserad')} / ${esc(r.market_regime || 'ej analyserad')}</span></div><a class="button" href="/candidate?scan=${encodeURIComponent(r.scan_timestamp)}&symbol=${encodeURIComponent(r.symbol)}">Full analys</a></article>`).join('')}</div>`
    : '<div class="empty">Inga sparade AI-beslut ännu. Kör run_ai_scan.bat först.</div>';
}
refreshOverview();
setInterval(refreshOverview, 15000);
</script>
"""


SCANNER_REFRESH_SCRIPT = """
<script>
async function refreshScannerTimestamp(){
  const response = await fetch('/api/overview');
  if(!response.ok) return;
  const data = await response.json();
  document.title = `TradingBot - Scanner - ${data.latest_scan_timestamp || 'no scan'}`;
}
refreshScannerTimestamp();
setInterval(refreshScannerTimestamp, 15000);
</script>
"""


CSS = """
:root{color-scheme:dark;--bg:#070b14;--panel:#101826;--panel2:#141f31;--text:#e5e7eb;--muted:#94a3b8;--line:#263447;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#38bdf8}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#0b1120,#070b14 45%,#0f172a);color:var(--text);font:14px/1.5 Segoe UI,Inter,Arial,sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:220px;background:rgba(9,14,25,.96);border-right:1px solid var(--line);padding:22px}
.brand{font-weight:800;font-size:22px;letter-spacing:.03em;margin-bottom:24px}nav{display:grid;gap:6px}nav a{color:var(--muted);text-decoration:none;padding:10px 11px;border-radius:10px}nav a.active,nav a:hover{background:#1e293b;color:#fff}
.lock{position:absolute;bottom:18px;left:22px;right:22px;color:#fecaca;background:#3f1111;border:1px solid #7f1d1d;border-radius:999px;padding:8px 10px;text-align:center;font-weight:800;font-size:12px}
main{margin-left:220px;padding:28px;max-width:none}.hero,.page-title{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:18px}.hero.compact h1{font-size:34px}.eyebrow{color:var(--blue);text-transform:uppercase;letter-spacing:.14em;font-size:11px}h1{font-size:36px;margin:.05em 0}h2{margin:0 0 8px}h3{margin:0;font-size:20px}
.priority-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:18px}.overview-layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:18px;align-items:start;margin-top:18px}.overview-layout.lower{grid-template-columns:minmax(0,1fr) minmax(340px,.7fr)}.main-panel{min-height:360px}.side-panel{position:sticky;top:28px}
.metric,.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 16px 42px rgba(0,0,0,.2)}.metric span,.muted{color:var(--muted);display:block}.metric strong{font-size:24px;display:block;margin-top:5px;word-break:break-word}.metric small{color:var(--muted);display:block;margin-top:5px}.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:20px 0}
.section-heading{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:14px}.section-heading p{color:var(--muted);margin:0}.decision-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.decision-card{background:#0f172a;border:1px solid var(--line);border-radius:16px;padding:16px}.decision-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.decision-card p{color:#cbd5e1;min-height:64px}.decision-metrics{display:grid;gap:6px;color:var(--muted);margin:12px 0}.decision-metrics b{color:var(--text)}.ticker{display:block;color:var(--muted);font-size:12px;margin-top:2px}
table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid var(--line);padding:12px 10px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}a{color:#7dd3fc}.button{display:inline-block;text-decoration:none;background:#2563eb;color:white;border-radius:999px;padding:8px 13px;font-weight:800}.button.secondary{background:#334155;color:#e2e8f0;margin-top:14px}
.badge,.status{border-radius:999px;padding:4px 10px;background:#334155;color:#e2e8f0;font-weight:800;font-size:12px;white-space:nowrap}.approve{background:#064e3b;color:#bbf7d0}.watch{background:#713f12;color:#fde68a}.reject{background:#7f1d1d;color:#fecaca}.online,.current{background:#064e3b;color:#bbf7d0}.locked{background:#3f1111;color:#fecaca}.empty{color:var(--muted)}.error{border-color:#ef4444}.summary-list{display:grid;gap:10px}.summary-list div{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px}.summary-list span{color:var(--muted)}.summary-list strong{font-size:18px}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}.filters input,.filters select,.filters button{background:#0f172a;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px}.filters button{background:#2563eb;border-color:#1d4ed8;font-weight:700}.range-tabs{display:flex;gap:8px;flex-wrap:wrap}.range-tabs button{background:#0f172a;color:#cbd5e1;border:1px solid var(--line);border-radius:999px;padding:7px 11px;font-weight:800}.range-tabs button:last-child{background:#1e3a8a;color:#dbeafe}.equity-chart{height:220px}.chart-empty{padding:26px}.activity-list{display:grid;gap:12px}.activity-item{border-bottom:1px solid var(--line);padding-bottom:10px}.activity-item span{color:var(--muted);font-size:12px}.activity-item strong{display:block;margin-top:3px}.activity-item p{margin:4px 0 0;color:#cbd5e1}
svg{width:100%;height:auto}@media(max-width:1150px){.priority-grid,.overview-layout{grid-template-columns:1fr}.side-panel{position:static}aside{position:static;width:auto}.lock{position:static;margin-top:18px}main{margin-left:0}}
"""
