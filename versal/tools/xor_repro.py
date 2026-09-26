"""
Offline multi-seed XOR reproducibility experiment with isolated persistent state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, cast

import torch

from versal.checkpoint import deserialize_rng, serialize_rng, write_checkpoint
from versal.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from versal.evolution.loop import state_from_dict, state_to_dict
from versal.evolution.registry import build_loop
from versal.library import ModuleLibrary, expanded_payload_complexity, macro_resolver, payload_refs
from versal.orchestrator import Orchestrator, comp_task_spec
from versal.orchestrator_types import attempts_from_dicts, attempts_to_dicts
from versal.strategy_sessions import capture_torch_rng, restore_torch_rng
from versal.substrate import decode_module, set_macro_resolver
from versal.utils.config import Config


def xor_task() -> Task:
    """
    Build the complete four-row Boolean domain without fetching a dataset.
    """

    def field(values: list[int]) -> Field:
        return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)

    pairs = [(field([a, b]), field([a ^ b])) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    return Task(TaskMeta(1, TaskKind.MAP, "xor", fixed_split=True), pairs, pairs)


def _checkpoint(orchestrator: Orchestrator) -> dict[str, Any]:
    return {
        "rng": serialize_rng(orchestrator.state.rng),
        "torch_rng": capture_torch_rng(),
        "loop": state_to_dict(orchestrator.state),
        "speciation": cast(Any, orchestrator.loop.evolver.speciate).state_dict(),
        "search": orchestrator.search.state_dict() if orchestrator.search is not None else None,
        "attempts": attempts_to_dicts(orchestrator.attempts),
        "counters": orchestrator.counters,
    }


def run_seed(
    config: dict[str, Any], directory: Path, *, seed: int, encounters: int = 100, policy: str = "interleaved", resume: bool = False, stop_at_minimum: bool = False
) -> dict[str, Any]:
    """
    Run or resume one cold-library experiment and retain executable final evidence.
    """
    if encounters < 1:
        raise ValueError("encounters must be positive")
    if resume and not (directory / "checkpoint.json").exists():
        raise FileNotFoundError(f"no completed task boundary to resume: {directory}")
    config = copy.deepcopy(config)
    config["seed"] = seed
    config["orchestrator"]["search_policy"] = policy
    config["orchestrator"]["library_dir"] = str(directory / "library")
    checkpoint_path = directory / "checkpoint.json"
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f"refusing to mix cold-seed evidence with existing state: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted((Config.PROJECT_ROOT / "versal").rglob("*.py")):
        digest.update(path.relative_to(Config.PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    manifest = directory / "invocations.json"
    invocations = json.loads(manifest.read_text()) if manifest.exists() else []
    invocations.append(
        {
            "code_sha256": digest.hexdigest(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "resume": resume,
            "requested_encounters": encounters,
            "stop_at_minimum": stop_at_minimum,
        }
    )
    manifest.write_text(json.dumps(invocations, indent=2))
    (directory / "config.json").write_text(json.dumps(config, indent=2, default=str))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    loop = build_loop(config)
    library = ModuleLibrary(directory / "library")
    loop.attach_library(library)
    set_macro_resolver(macro_resolver(library))
    restored = json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else None
    state = state_from_dict(restored["loop"], deserialize_rng(restored["rng"])) if restored else loop.fresh_state(random.Random(seed))
    if restored:
        loop.evolver.speciate.load_state_dict(restored["speciation"])
        restore_torch_rng(restored["torch_rng"])
    orchestrator = Orchestrator(config, loop, library, state)
    if restored:
        orchestrator.attempts = attempts_from_dicts(restored["attempts"])
        orchestrator.counters.update(restored["counters"])
        if orchestrator.search is not None and restored["search"] is not None:
            orchestrator.search.load_state_dict(restored["search"])
    task = xor_task()
    rows = json.loads((directory / "trajectory.json").read_text()) if restored else []
    started = time.perf_counter()
    for encounter in range(len(rows), encounters):
        solution = orchestrator.solve(task)
        attempt = orchestrator.attempts[-1]
        orchestrator.finish_root_task(attempt)
        key = solution.key if solution is not None else None
        entry = library.load(key) if key else None
        row = {
            "encounter": encounter + 1,
            "key": key,
            "support": attempt.support_accuracy,
            "query": attempt.query_accuracy,
            "complexity": expanded_payload_complexity(entry.entry_type, entry.payload, library) if entry else None,
            "outcome": attempt.outcome,
            "strategy": attempt.strategy,
            "validation": attempt.validation_status,
            "generations": attempt.generations,
            "refinement_generations": attempt.refine_generations,
            "work": attempt.strategy_work,
        }
        rows.append(row)
        (directory / "trajectory.json").write_text(json.dumps(rows, indent=2))
        write_checkpoint(directory, _checkpoint(orchestrator))
        print(json.dumps({"seed": seed, "policy": policy, **{k: v for k, v in row.items() if k != "work"}}), flush=True)
        if stop_at_minimum and row["support"] == 1.0 and row["complexity"] is not None and row["complexity"] <= 5:
            break
    final = rows[-1]
    closure: dict[str, Any] = {}
    pending = [final["key"]] if final["key"] else []
    while pending:
        key = pending.pop()
        if key in closure:
            continue
        entry = library.load(key)
        closure[key] = entry.to_dict()
        pending.extend(payload_refs(entry.entry_type, entry.payload))
    (directory / "final_payloads.json").write_text(json.dumps(closure, indent=2))
    predictions = None
    if final["key"]:
        from versal.evolution.composition import AssemblyContext, assemble, comp_from_dict
        from versal.evolution.genome import genome_from_dict
        from versal.library import MODULE

        spec = comp_task_spec(task)
        entry = library.load(final["key"])
        module = (
            decode_module(genome_from_dict(entry.payload), spec.n_inputs, spec.output_width, macro_resolver=macro_resolver(library), max_inline_depth=loop.max_inline_depth)
            if entry.entry_type == MODULE
            else assemble(
                comp_from_dict(entry.payload), AssemblyContext(bank_columns=dict(spec.bank_columns), library=library, max_inline_depth=loop.max_inline_depth), spec.n_inputs
            )
        )
        with torch.no_grad():
            logits = module(spec.encoded.support_input[0])
            labels = spec.encoder.decode(logits, spec.encoded.support_target[2])
        predictions = {
            "inputs": spec.encoded.support_input[0].tolist(),
            "logits": logits.reshape(-1).tolist(),
            "predictions": labels.reshape(-1).tolist(),
            "expected": [0, 1, 1, 0],
        }
        (directory / "predictions.json").write_text(json.dumps(predictions, indent=2))
    accepted = [row for row in rows if row["support"] == 1.0 and row["key"]]
    nonregressing = all(row["support"] == 1.0 and row["query"] == 1.0 for row in rows[rows.index(accepted[0]) :]) if accepted else False
    nonincreasing = all(right["complexity"] <= left["complexity"] for left, right in zip(accepted, accepted[1:]))
    summary = {
        "seed": seed,
        "policy": policy,
        "encounters": len(rows),
        "final": final,
        "perfect_nonregressing": nonregressing,
        "complexity_nonincreasing": nonincreasing,
        "passed": (
            final["support"] == final["query"] == 1.0
            and final["complexity"] is not None
            and final["complexity"] <= 5
            and nonregressing
            and nonincreasing
            and predictions is not None
            and predictions["predictions"] == predictions["expected"]
        ),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "activations": sorted({node["activation"] for entry in closure.values() for node in entry["payload"].get("nodes", []) if "activation" in node}),
        "predictions": predictions,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/xor_repro.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--encounters", type=int, default=100)
    parser.add_argument("--policy", choices=["interleaved", "ladder"], default="interleaved")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at-minimum", action="store_true", help="End a seed once a perfect support solution reaches complexity <= 5; query remains report-only.")
    args = parser.parse_args()
    config = Config(args.config).current
    summaries = [
        run_seed(
            config,
            args.output / f"{args.policy}-seed{seed}",
            seed=int(seed),
            encounters=args.encounters,
            policy=args.policy,
            resume=args.resume,
            stop_at_minimum=args.stop_at_minimum,
        )
        for seed in args.seeds.split(",")
    ]
    print(json.dumps({"passed": all(summary["passed"] for summary in summaries), "seeds": len(summaries)}), flush=True)
    if not all(summary["passed"] for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
