"""Generate or verify Versal's deterministic canonical-runtime inventory."""

import argparse
import difflib
import hashlib
import importlib
import json
import tomllib
from pathlib import Path
from typing import Any

from versal.evolution.registry import Registry
from versal.utils.config import Config

PROJECT_ROOT = Config.PROJECT_ROOT
DEFAULT_MANIFEST = PROJECT_ROOT / "runtime_inventory.json"
CANONICAL_CONFIG_PATH = Config.DEFAULT_CONFIG.relative_to(PROJECT_ROOT).as_posix()

REGISTRY_MODULES = (
    "versal.decompose",
    "versal.evolution.composition",
    "versal.evolution.crossover",
    "versal.evolution.evaluate",
    "versal.evolution.fitness",
    "versal.evolution.init",
    "versal.evolution.loop",
    "versal.evolution.mutation",
    "versal.evolution.schedule",
    "versal.evolution.selection",
    "versal.evolution.speciation",
    "versal.evolution.train",
    "versal.library",
    "versal.strategy",
)

# Paths intentionally describe the ordinary app entry point, not opt-in maintenance tools. Keep
# conditions explicit so the manifest does not imply every path is touched by every task.
RUN_PATHS: tuple[dict[str, str], ...] = (
    {"path": "pyproject.toml", "access": "read", "condition": "startup project metadata"},
    {"path": CANONICAL_CONFIG_PATH, "access": "read", "condition": "default startup"},
    {"path": "<Hugging Face Hub cache>", "access": "read-write", "condition": "selected Icarus Parquet shards; content-addressed and outside the repository"},
    {"path": "<library_dir>/index.json", "access": "read-write", "condition": "library load, admission, statistics, retirement, and optional run-end GC"},
    {"path": "<library_dir>/entries/<key>.json", "access": "read-write", "condition": "library entry load, admission, statistics, and optional run-end GC"},
    {"path": "<library_dir>/router/router_meta.json", "access": "read-write", "condition": "routed persistence enabled"},
    {"path": "<library_dir>/router/router_state.pt", "access": "read-write", "condition": "routed persistence enabled"},
    {"path": "<library_dir>/router_stale_<timestamp>/", "access": "write", "condition": "persisted router is incompatible and persist_strict is false"},
    {"path": "<library_dir>/grammar/grammar.json", "access": "read-write", "condition": "grammar strategy rebuilds after the live library key set changes"},
    {"path": "<library_dir>/images/<key>.png", "access": "delete", "condition": "optional run-end GC removes an unreferenced retired entry"},
    {"path": "<library_dir>/images/overmind.png", "access": "write", "condition": "routed expert set changes"},
    {"path": "<library_dir>/images/overmind_pruned.png", "access": "write", "condition": "full overmind portrait is written; retired experts are compacted out"},
    {"path": "results/<timestamp>_orchestrated/", "access": "write", "condition": "new run creates <run_dir>; --resume uses the supplied directory instead"},
    {"path": "results/compute_policy.json", "access": "read", "condition": "canonical scheduled trainer uses a matching explicit calibration profile when present"},
    {"path": "<run_dir>/config.toml", "access": "read-write", "condition": "source config snapshot written on a new run and loaded by implicit --resume"},
    {"path": "<run_dir>/config.toml.sha256", "access": "write", "condition": "source config snapshot digest"},
    {"path": "<run_dir>/config.effective.json", "access": "read-write", "condition": "CLI-adjusted effective config written on a new run and loaded by implicit --resume"},
    {"path": "<run_dir>/config.effective.json.sha256", "access": "write", "condition": "effective config snapshot digest"},
    {"path": "<run_dir>/frozen_library/", "access": "write", "condition": "[library] fresh_per_task freezes the starting state for the no-memory control"},
    {"path": "<run_dir>/run_summary.json", "access": "read-write", "condition": "prior rows load on resume; refreshed at startup, task boundaries, crash, and completion"},
    {"path": "<run_dir>/task_pool.json", "access": "read-write", "condition": "pinned streamed task references written after discovery and reused on resume"},
    {"path": "<run_dir>/checkpoint.json", "access": "read-write", "condition": "loaded on resume and overwritten with rolling resumable state"},
    {"path": "<run_dir>/task_<NNNN>/stats.json", "access": "write", "condition": "task admits a library entry"},
    {"path": "<run_dir>/task_<NNNN>/checkpoint.json", "access": "write", "condition": "task admits a library entry"},
    {"path": "<run_dir>/task_<NNNN>/net.png", "access": "write", "condition": "task admits a library entry"},
    {"path": "<run_dir>/task_<NNNN>/speciation.png", "access": "write", "condition": "task admits a library entry"},
)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _config_surface(table: dict[str, Any]) -> tuple[list[str], list[str]]:
    tables: list[str] = []
    keys: list[str] = []

    def visit(value: dict[str, Any], prefix: str = "") -> None:
        for name in sorted(value):
            path = f"{prefix}.{name}" if prefix else name
            child = value[name]
            if isinstance(child, dict):
                tables.append(path)
                visit(child, path)
            else:
                keys.append(path)

    visit(table)
    return sorted(tables), sorted(keys)


def _registry_surface() -> dict[str, list[str]]:
    registries: dict[str, list[str]] = {}
    seen: set[int] = set()
    for module_name in REGISTRY_MODULES:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if not isinstance(value, Registry) or id(value) in seen:
                continue
            seen.add(id(value))
            if value.kind in registries:
                raise RuntimeError(f"duplicate registry kind {value.kind!r}")
            # Tests register local stub strategies into the process-global registries. Inventory only
            # repository implementations so collection order cannot contaminate the checked-in file.
            registries[value.kind] = sorted(name for name, item in value._items.items() if getattr(item, "__module__", "").startswith("versal."))
    return dict(sorted(registries.items()))


def build_inventory() -> dict[str, Any]:
    config_path = Config.DEFAULT_CONFIG
    config_bytes = config_path.read_bytes()
    effective_config, _sources = Config._load_config_tree(config_path)
    with open(Config.TOML_FILE, "rb") as handle:
        project = tomllib.load(handle)
    tables, keys = _config_surface(effective_config)
    scripts = {str(name): str(target) for name, target in project.get("project", {}).get("scripts", {}).items()}
    runtime_config = Config(conf_path=config_path).current
    return {
        "schema_version": 1,
        "canonical_config": _relative(config_path),
        "canonical_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "available_configs": sorted(_relative(path) for path in (PROJECT_ROOT / "configs").glob("*.toml")),
        "config_tables": tables,
        "config_keys": keys,
        "runtime_config_keys": sorted(runtime_config),
        "runtime_hyperparameter_keys": sorted(runtime_config["hyperparameters"]),
        "registries": _registry_surface(),
        "console_scripts": dict(sorted(scripts.items())),
        "run_paths": list(RUN_PATHS),
    }


def render_inventory(inventory: dict[str, Any] | None = None) -> str:
    return json.dumps(inventory if inventory is not None else build_inventory(), indent=2, sort_keys=True) + "\n"


def check_manifest(path: Path = DEFAULT_MANIFEST) -> bool:
    expected = render_inventory()
    try:
        actual = path.read_text()
    except FileNotFoundError:
        actual = ""
    if actual == expected:
        return True
    diff = difflib.unified_diff(actual.splitlines(), expected.splitlines(), fromfile=str(path), tofile="generated runtime inventory", lineterm="")
    print("\n".join(diff))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify runtime_inventory.json against the canonical Versal runtime surface.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Exit nonzero when the checked-in inventory differs from the live runtime surface.")
    mode.add_argument("--write", action="store_true", help="Refresh the checked-in inventory from the live runtime surface.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Inventory path (default: repository runtime_inventory.json).")
    args = parser.parse_args()

    if args.check:
        if not check_manifest(args.manifest):
            raise SystemExit(1)
        print(f"runtime inventory is current: {_relative(args.manifest)}")
    elif args.write:
        args.manifest.write_text(render_inventory())
        print(f"wrote runtime inventory: {_relative(args.manifest)}")
    else:
        print(render_inventory(), end="")


if __name__ == "__main__":
    main()
