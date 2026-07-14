"""Verify and migrate a copied router from format v1 to sharded format v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ardevo.routing import migrate_router_library


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path, help="source library containing router format v1")
    parser.add_argument("--output", required=True, type=Path, help="new library copy; must not exist")
    args = parser.parse_args(argv)
    print(json.dumps(migrate_router_library(args.library, args.output), indent=2))


if __name__ == "__main__":
    main()
