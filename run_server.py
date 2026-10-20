"""CLI launcher for NeuroScan AI server.

Usage:
    python run_server.py [--host 0.0.0.0] [--port 8000] [--reload]
"""
from __future__ import annotations

import argparse
import os
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI Server Runner")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 1)), help="Worker processes")

    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 65)
    print("   [+] NEUROSCAN AI - Brain Tumor MRI Classification Server")
    print(f"   [+] Serving on: http://{args.host}:{args.port}")
    print(f"   [+] API Documentation: http://{args.host}:{args.port}/docs")
    print("=" * 65)

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        access_log=True,
    )


if __name__ == "__main__":
    main()
