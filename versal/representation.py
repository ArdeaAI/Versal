"""
Registered module representations at serialization, execution, and reuse boundaries.

The explicit genome remains the fallback, including legacy field payloads whose
task adapter still supplies their historical feature encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from versal.evolution.registry import Registry


@dataclass(frozen=True)
class RepresentationCodec:
    restore: Callable[[dict[str, Any]], Any]
    serialize: Callable[[Any], dict[str, Any]]
    decode: Callable[..., Any]
    ports: Callable[[Any], tuple[int, int]]
    expand: Callable[..., Any]
    topology: Callable[[dict[str, Any]], dict[str, Any]]


REPRESENTATION: Registry[RepresentationCodec] = Registry("module_representation")


def codec_for(value: Any) -> RepresentationCodec | None:
    name = value.get("representation") if isinstance(value, dict) else getattr(value, "representation", None)
    if name == "spatial" and name not in REPRESENTATION.names():
        from versal import spatial  # noqa: F401
    return REPRESENTATION.get(name) if name is not None else None


def module_ports(genome: Any) -> tuple[int, int]:
    codec = codec_for(genome)
    return codec.ports(genome) if codec is not None else (len(genome.input_ids), len(genome.output_ids))


def explicit_genome(genome: Any, *, limit: int = 16384) -> Any:
    codec = codec_for(genome)
    return codec.expand(genome, limit=limit) if codec is not None else genome


def topology_extension(payload: dict[str, Any]) -> dict[str, Any]:
    codec = codec_for(payload)
    return codec.topology(payload) if codec is not None else {}
