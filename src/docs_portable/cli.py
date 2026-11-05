from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docs-portable",
        description="docs-mcp-server with a local BGE-M3 OpenAI-compatible embedding endpoint.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="install anything missing, then start everything (default)")
    sub.add_parser("setup", help="only download/install dependencies and the model")
    sub.add_parser("embedding-server", help="run just the OpenAI-compatible embedding server")
    sub.add_parser("rerank-server", help="run just the /v1/rerank server")
    sub.add_parser("jev-server", help="run just the JEV-9B /v1/systemone server (needs DOCS_PORTABLE_JEV_TOKEN)")
    cli = sub.add_parser("cli", help="run the docs-mcp-server CLI (scrape, search, list, ...) with this setup")
    cli.add_argument("cli_args", nargs=argparse.REMAINDER, help="arguments passed to docs-mcp-server")
    args = parser.parse_args(argv)
    command = args.command or "run"

    try:
        cfg = load_config()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if command == "setup":
        from .installer import ensure_all

        ensure_all(cfg)
        return 0
    if command == "embedding-server":
        from .embedding_server import main as serve

        serve()
        return 0
    if command == "rerank-server":
        from .rerank_server import main as serve_rerank

        serve_rerank()
        return 0
    if command == "jev-server":
        from .jev_server import main as serve_jev

        serve_jev()
        return 0

    from .launcher import run, run_cli

    if command == "cli":
        return run_cli(cfg, args.cli_args)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
