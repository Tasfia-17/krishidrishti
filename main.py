"""
KrishiDrishti — AI Crop Doctor
Entry point: starts the FastAPI server.

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 7878
    python main.py --mode accurate
"""
import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path

import uvicorn

from config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishidrishti")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║   🌾  KrishiDrishti — AI Crop Doctor                 ║
║   कृषिदृष्टि — Powered by Gemma 4                   ║
║                                                      ║
║   100% Local · 100% Private · Zero Cloud            ║
╚══════════════════════════════════════════════════════╝
""")


def check_gemma():
    """Warn if llama-server is not reachable at startup."""
    from gemma_engine import llm_client
    if llm_client.is_available():
        logger.info(f"✓ Gemma 4 reachable at {settings.llama_server_host}")
    else:
        logger.warning(
            f"⚠  Cannot reach llama-server at {settings.llama_server_host}\n"
            f"   Start it with:\n"
            f"   llama-server -m gemma-4-it-Q4_K_M.gguf --port 8080\n"
            f"   KrishiDrishti will work once llama-server is running."
        )


def main():
    parser = argparse.ArgumentParser(
        description="KrishiDrishti — AI Crop Doctor powered by Gemma 4"
    )
    parser.add_argument("--host",   default=settings.host,   help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=settings.port,   type=int, help="Server port (default: 7878)")
    parser.add_argument("--mode",   default=None,            help="Analysis mode: fast/balanced/accurate")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser on start")
    args = parser.parse_args()

    # Override mode from CLI if provided
    if args.mode:
        settings.analysis_mode = args.mode

    print_banner()
    check_gemma()

    url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
    logger.info(f"Starting KrishiDrishti at {url}")
    logger.info(f"API docs: {url}/docs")
    logger.info(f"Analysis mode: {settings.analysis_mode}")
    logger.info(f"Database: {settings.db_path}")

    if not args.no_browser:
        import threading
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
