#!/usr/bin/env python3
"""
Start the YellowStars web monitoring dashboard.

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8080
"""

import sys
import argparse

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings
from yellowstars.ui.app import create_app


def main():
    parser = argparse.ArgumentParser(description="YellowStars Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "INFO")

    settings = load_settings(args.config)
    app = create_app(settings)

    print(f"\n{'='*60}")
    print(f"  YellowStars Trading Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
