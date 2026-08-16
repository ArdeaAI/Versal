"""CLI for deterministic reports over durable orchestrated-run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from versal.reporting import write_run_report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--library", type=Path, help="optional library root; only index.json and motifs.json are read")
    args = parser.parse_args(argv)
    report = write_run_report(args.run_dir, args.library)
    print(json.dumps({"schema_version": report["schema_version"], "run_dir": str(args.run_dir), "artifacts": ["rung_summary.csv", "run_report.json", "run_report.md"]}))


if __name__ == "__main__":
    main()
