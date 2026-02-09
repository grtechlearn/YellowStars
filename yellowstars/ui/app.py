"""
Flask-based web dashboard for monitoring the trading system.

Features:
- Real-time portfolio overview
- Equity curve visualization
- Trade log
- Strategy signals
- System health status
- Backtest runner
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from loguru import logger

from yellowstars.config.settings import Settings


def create_app(settings: Optional[Settings] = None) -> Flask:
    """Create the Flask monitoring dashboard."""

    app = Flask(__name__)
    CORS(app)

    # Store settings
    app.config["YS_SETTINGS"] = settings

    # ============================================================
    # Dashboard HTML (single-page app embedded)
    # ============================================================

    DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YellowStars - Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Segoe UI', Roboto, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            min-height: 100vh;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #ffd700;
        }
        .header h1 { color: #ffd700; font-size: 24px; }
        .header .status { display: flex; gap: 15px; align-items: center; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .status-dot.green { background: #00ff88; box-shadow: 0 0 8px #00ff88; }
        .status-dot.red { background: #ff4444; box-shadow: 0 0 8px #ff4444; }
        .status-dot.yellow { background: #ffd700; box-shadow: 0 0 8px #ffd700; }

        /* Navigation */
        .nav { display: flex; gap: 5px; padding: 10px 30px; background: #111127; }
        .nav button {
            padding: 8px 20px;
            border: 1px solid #333;
            background: transparent;
            color: #888;
            cursor: pointer;
            border-radius: 6px;
            font-size: 13px;
        }
        .nav button.active { background: #ffd700; color: #000; border-color: #ffd700; font-weight: bold; }
        .nav button:hover { border-color: #ffd700; color: #ffd700; }

        /* Main Content */
        .main { padding: 20px 30px; }

        /* Cards */
        .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .card {
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
        }
        .card .label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
        .card .value { font-size: 28px; font-weight: bold; margin-top: 5px; }
        .card .sub { color: #666; font-size: 12px; margin-top: 3px; }
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .neutral { color: #ffd700; }

        /* Table */
        .table-container { background: #1a1a2e; border-radius: 10px; border: 1px solid #333; overflow: hidden; margin-top: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #16213e; padding: 12px 15px; text-align: left; color: #00d4ff; font-size: 12px; text-transform: uppercase; }
        td { padding: 10px 15px; border-bottom: 1px solid #1e1e3e; font-size: 13px; }
        tr:hover { background: #16213e44; }

        /* Chart placeholder */
        .chart-container {
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .chart-container canvas { width: 100% !important; }

        /* Sections */
        .section-title { color: #00d4ff; margin: 20px 0 10px; font-size: 16px; }

        /* Actions */
        .actions { display: flex; gap: 10px; margin: 20px 0; }
        .btn {
            padding: 10px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 13px;
        }
        .btn-primary { background: #ffd700; color: #000; }
        .btn-danger { background: #ff4444; color: #fff; }
        .btn-secondary { background: #333; color: #fff; }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); }

        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #444;
            font-size: 11px;
            border-top: 1px solid #222;
            margin-top: 30px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .card-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⭐ YellowStars</h1>
        <div class="status">
            <span class="status-dot green" id="systemStatus"></span>
            <span id="systemStatusText">System Online</span>
            <span style="color: #666">|</span>
            <span id="modeDisplay">PAPER</span>
            <span style="color: #666">|</span>
            <span id="timeDisplay"></span>
        </div>
    </div>

    <div class="nav">
        <button class="active" onclick="showTab('overview')">Overview</button>
        <button onclick="showTab('positions')">Positions</button>
        <button onclick="showTab('trades')">Trade Log</button>
        <button onclick="showTab('backtest')">Backtest</button>
        <button onclick="showTab('settings')">Settings</button>
    </div>

    <div class="main">
        <!-- Overview Tab -->
        <div id="tab-overview">
            <div class="card-grid">
                <div class="card">
                    <div class="label">Total Equity</div>
                    <div class="value" id="totalEquity">$0.00</div>
                    <div class="sub" id="totalReturn">Loading...</div>
                </div>
                <div class="card">
                    <div class="label">Today's P&L</div>
                    <div class="value" id="dailyPnl">$0.00</div>
                    <div class="sub" id="dailyReturn">0.00%</div>
                </div>
                <div class="card">
                    <div class="label">Current Signal</div>
                    <div class="value neutral" id="currentSignal">HOLD</div>
                    <div class="sub" id="signalInstrument">--</div>
                </div>
                <div class="card">
                    <div class="label">Position Size</div>
                    <div class="value" id="positionSize">0%</div>
                    <div class="sub" id="positionValue">$0.00 allocated</div>
                </div>
                <div class="card">
                    <div class="label">Max Drawdown</div>
                    <div class="value negative" id="maxDrawdown">0.0%</div>
                    <div class="sub">Lifetime peak-to-trough</div>
                </div>
                <div class="card">
                    <div class="label">Total Trades</div>
                    <div class="value" id="totalTrades">0</div>
                    <div class="sub" id="winRate">Win rate: --</div>
                </div>
            </div>

            <div class="chart-container" id="equityChart">
                <span style="color: #555">Equity curve will render here (connect data provider)</span>
            </div>

            <h3 class="section-title">Recent Activity</h3>
            <div class="table-container">
                <table id="recentActivity">
                    <thead>
                        <tr><th>Time</th><th>Action</th><th>Symbol</th><th>Quantity</th><th>Price</th><th>P&L</th></tr>
                    </thead>
                    <tbody id="activityBody">
                        <tr><td colspan="6" style="text-align:center; color:#555">No recent activity</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Backtest Tab -->
        <div id="tab-backtest" style="display:none">
            <h3 class="section-title">Run Backtest</h3>
            <div class="actions">
                <button class="btn btn-primary" onclick="runBacktest()">Run Backtest</button>
                <button class="btn btn-secondary" onclick="refreshData()">Refresh Data</button>
            </div>
            <div id="backtestResults" class="chart-container">
                <span style="color: #555">Configure and run a backtest to see results</span>
            </div>
        </div>

        <!-- Other tabs -->
        <div id="tab-positions" style="display:none">
            <h3 class="section-title">Open Positions</h3>
            <div class="table-container">
                <table>
                    <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th><th>P&L</th><th>%</th></tr></thead>
                    <tbody id="positionsBody">
                        <tr><td colspan="7" style="text-align:center; color:#555">No open positions</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div id="tab-trades" style="display:none">
            <h3 class="section-title">Trade History</h3>
            <div class="table-container">
                <table>
                    <thead><tr><th>Date</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Duration</th></tr></thead>
                    <tbody id="tradesBody">
                        <tr><td colspan="8" style="text-align:center; color:#555">No trades yet</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div id="tab-settings" style="display:none">
            <h3 class="section-title">System Configuration</h3>
            <div class="card-grid">
                <div class="card">
                    <div class="label">Trading Mode</div>
                    <div class="value neutral" id="settingsMode">PAPER</div>
                </div>
                <div class="card">
                    <div class="label">Data Provider</div>
                    <div class="value" id="settingsProvider">--</div>
                </div>
                <div class="card">
                    <div class="label">Broker</div>
                    <div class="value" id="settingsBroker">--</div>
                </div>
                <div class="card">
                    <div class="label">Strategy</div>
                    <div class="value" id="settingsStrategy">--</div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        YellowStars Trading Platform v0.1.0 | Built for autonomous systematic trading
    </div>

    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('[id^="tab-"]').forEach(t => t.style.display = 'none');
            document.getElementById('tab-' + tabName).style.display = 'block';
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
        }

        // Update clock
        function updateTime() {
            document.getElementById('timeDisplay').textContent = new Date().toLocaleTimeString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // Fetch status
        async function fetchStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                updateDashboard(data);
            } catch(e) {
                console.log('Status fetch error:', e);
            }
        }

        function updateDashboard(data) {
            if (data.equity) document.getElementById('totalEquity').textContent = '$' + data.equity.toLocaleString(undefined, {minimumFractionDigits: 2});
            if (data.daily_pnl !== undefined) {
                const pnl = data.daily_pnl;
                const el = document.getElementById('dailyPnl');
                el.textContent = '$' + Math.abs(pnl).toLocaleString(undefined, {minimumFractionDigits: 2});
                el.className = 'value ' + (pnl >= 0 ? 'positive' : 'negative');
                el.textContent = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toLocaleString(undefined, {minimumFractionDigits: 2});
            }
            if (data.mode) document.getElementById('modeDisplay').textContent = data.mode.toUpperCase();
        }

        // Auto-refresh every 30 seconds
        setInterval(fetchStatus, 30000);
        fetchStatus();

        async function runBacktest() {
            document.getElementById('backtestResults').innerHTML = '<span style="color:#ffd700">Running backtest...</span>';
            try {
                const resp = await fetch('/api/backtest', { method: 'POST' });
                const data = await resp.json();
                document.getElementById('backtestResults').innerHTML =
                    '<pre style="color:#00ff88; font-size:12px; text-align:left; width:100%; overflow:auto;">' +
                    JSON.stringify(data, null, 2) + '</pre>';
            } catch(e) {
                document.getElementById('backtestResults').innerHTML = '<span style="color:#ff4444">Error: ' + e + '</span>';
            }
        }

        function refreshData() { fetchStatus(); }
    </script>
</body>
</html>"""

    # ============================================================
    # Routes
    # ============================================================

    @app.route("/")
    def dashboard():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/status")
    def api_status():
        """System status endpoint."""
        settings = app.config.get("YS_SETTINGS")
        return jsonify({
            "status": "online",
            "mode": settings.trading_mode.value if settings else "unknown",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "equity": 0,
            "daily_pnl": 0,
            "data_provider": settings.data_provider.name if settings else "",
            "broker": settings.broker.name if settings else "",
            "strategies": len(settings.strategies) if settings else 0,
        })

    @app.route("/api/portfolio")
    def api_portfolio():
        """Portfolio endpoint."""
        return jsonify({
            "cash": 0,
            "positions": [],
            "total_equity": 0,
        })

    @app.route("/api/trades")
    def api_trades():
        """Trade history endpoint."""
        return jsonify({"trades": []})

    @app.route("/api/backtest", methods=["POST"])
    def api_backtest():
        """Run a backtest via API."""
        settings = app.config.get("YS_SETTINGS")
        if not settings:
            return jsonify({"error": "No settings configured"}), 500

        try:
            from yellowstars.data.manager import DataManager
            from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
            from yellowstars.backtest.engine import BacktestEngine
            from yellowstars.backtest.metrics import PerformanceMetrics

            dm = DataManager(settings.data_provider)
            dm.initialize()

            strat_settings = settings.strategies[0] if settings.strategies else None
            strategy = MalikWhiteLightStrategy(strat_settings)

            underlying = strat_settings.underlying_index if strat_settings else "NDX"
            data = dm.get_data(underlying, date(1985, 1, 1), date.today())

            engine = BacktestEngine(settings.backtest)
            result = engine.run(strategy, data)

            pm = PerformanceMetrics(result.equity_curve, result.benchmark_curve, result.trades)
            metrics = pm.compute_all()

            return jsonify({
                "status": "success",
                "strategy": strategy.name,
                "metrics": {
                    "return_metrics": metrics.get("return_metrics", {}),
                    "risk_metrics": metrics.get("risk_metrics", {}),
                    "drawdown_metrics": metrics.get("drawdown_metrics", {}),
                    "trade_metrics": metrics.get("trade_metrics", {}),
                },
            })

        except Exception as e:
            logger.error(f"Backtest API error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health")
    def api_health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

    return app
