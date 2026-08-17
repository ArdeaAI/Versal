"""NEAT-style genome: a directed acyclic graph of typed nodes and weighted connections.

A genome describes a topology, not weights-in-isolation: connection weights ride along on
the genes so structure and weights co-evolve. The substrate decoder turns a genome into an
executable torch module; the mutation operators grow the graph from a minimal seed.
"""

import math
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class NodeKind(Enum):
    INPUT = "input"
    BIAS = "bias"
    HIDDEN = "hidden"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class NodeGene:
    id: int
    kind: NodeKind
    activation: str
    # Axis-normalized position used ONLY by the geometry-biased mutation operators (the substrate
    # decoder ignores it). None means "no geometry" - the single-task path and the bias node leave it
    # unset, so locality operators simply skip those nodes.
    coordinate: tuple[float, ...] | None = None
    # How incoming edges combine before the activation: "sum" (the classic neuron) or "product"
    # (multiplicative unit). Product nodes make gating and second-order interactions evolvable.
    aggregation: str = "sum"


@dataclass(frozen=True, slots=True)
class ConnectionGene:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innovation: int
    # Recurrent edges are TIME-DELAYED: they read the previous step's value, so they are exempt from
    # the acyclicity rules (the forward graph stays a DAG) and are inert under the plain GraphNet.
    recurrent: bool = False
    # HARD WEIGHT SHARING (the WeightGroup lever): edges carrying the same tie_group share ONE
    # trainable parameter at decode (gradient accumulates across every stamped copy), which is how
    # convolution-like structure becomes trainable at wide widths without being hand-inserted:
    # add_shared_motif(tied=true) assigns groups at stamp time, untie_motif_weights dissolves them.
    # Group ids come from InnovationTracker.new_marker (run-unique, crossover-stable). None (the
    # default) is the ordinary independent weight: untied genomes are byte-identical everywhere,
    # on disk and in the substrate. Forward edges only; recurrent genes ignore the field.
    tie_group: int | None = None


@dataclass(frozen=True, slots=True)
class MacroGene:
    """A whole library network as a single opaque unit inside a flat genome (the LSTM-cell idea).

    The inner network is resolved at decode time from `ref` and runs FROZEN: its k ordered inputs
    read `input_node_ids` and its m outputs land on `output_node_ids`, which are real HIDDEN
    identity NodeGenes so all downstream wiring/serialization/rendering works unchanged. One
    `innovation` marker makes the placement an atomic unit for crossover and speciation."""

    ref: str  # "library:<key>" only; live refs would dangle the moment the run ends
    input_node_ids: tuple[int, ...]  # ordered, length k = inner input count
    output_node_ids: tuple[int, ...]  # length m = inner output count; fresh HIDDEN ids owned by this macro
    innovation: int
    trainable: bool = False  # reserved; v1 macros are always frozen


@dataclass
class Genome:
    """A topology candidate. Node genes are keyed by id; connection genes are an ordered list."""

    nodes: dict[int, NodeGene] = field(default_factory=dict)
    connections: list[ConnectionGene] = field(default_factory=list)
    macros: list[MacroGene] = field(default_factory=list)
    # How many times the refine substrate re-applies this network to a STATIC input, carrying node
    # state across passes (the TRM idea: recursion = effective depth without parameters). 1 = a plain
    # feedforward pass (the default decode path is byte-identical). Evolved by `tweak_refine_steps`;
    # only meaningful once the genome also carries recurrent edges to thread state between passes.
    refine_steps: int = 1
    # Self-adaptive mutation rates (lever F): per-operator probabilities carried as strategy genes and
    # perturbed by `AdaptiveMutationPipeline` (ES perturb-and-inherit). Empty = the fixed-rate default,
    # so a genome that never met the adaptive pipeline is byte-identical (the key is absent on disk too).
    operator_rates: dict[str, float] = field(default_factory=dict)
    # Gradient-proposed growth signals (NeST/GradMax scores from versal/evolution/growth.py), read
    # by the hinted mutation operators. IN-MEMORY ONLY: never serialized (genome_to_dict ignores it),
    # so library payloads, fingerprints, and checkpoints are byte-identical whether scoring ran or
    # not. Regenerated every trained generation; crossover children start without hints.
    growth_hints: dict[str, dict[int, float]] | None = None

    def clone(self) -> "Genome":
        # Gene dataclasses are frozen, so shallow copies of the containers are deep copies.
        return Genome(
            nodes=dict(self.nodes),
            connections=list(self.connections),
            macros=list(self.macros),
            refine_steps=self.refine_steps,
            operator_rates=dict(self.operator_rates),
            growth_hints=self.growth_hints,
        )

    def ids_of(self, kind: NodeKind) -> list[int]:
        return sorted(node.id for node in self.nodes.values() if node.kind is kind)

    @property
    def input_ids(self) -> list[int]:
        return self.ids_of(NodeKind.INPUT)

    @property
    def bias_ids(self) -> list[int]:
        return self.ids_of(NodeKind.BIAS)

    @property
    def output_ids(self) -> list[int]:
        return self.ids_of(NodeKind.OUTPUT)

    @property
    def hidden_ids(self) -> list[int]:
        return self.ids_of(NodeKind.HIDDEN)

    def enabled_connections(self) -> list[ConnectionGene]:
        return [conn for conn in self.connections if conn.enabled]

    def forward_connections(self) -> list[ConnectionGene]:
        """Enabled non-recurrent edges: the DAG the substrate levels and cycle checks operate on."""
        return [conn for conn in self.connections if conn.enabled and not conn.recurrent]

    def recurrent_connections(self) -> list[ConnectionGene]:
        return [conn for conn in self.connections if conn.enabled and conn.recurrent]

    def has_connection(self, in_id: int, out_id: int, recurrent: bool = False) -> bool:
        return any(conn.in_id == in_id and conn.out_id == out_id and conn.recurrent == recurrent for conn in self.connections)

    @property
    def macro_output_ids(self) -> frozenset[int]:
        return frozenset(node_id for macro in self.macros for node_id in macro.output_node_ids)

    def complexity(self) -> int:
        """Structural cost: enabled edges plus hidden nodes plus macro placements. The m macro
        output stubs already count as hidden nodes; the +1 per macro prices the placement itself."""
        return sum(connection.enabled for connection in self.connections) + sum(node.kind is NodeKind.HIDDEN for node in self.nodes.values()) + len(self.macros)

    def max_node_id(self) -> int:
        return max(self.nodes) if self.nodes else -1


class ForwardReachability:
    """Cycle checks for a BATCH of candidate edges against one genome snapshot.

    `would_create_cycle` rebuilds the forward adjacency and BFSes per call, which is fine for a
    single check but quadratic death inside the geometry mutators' source x target enumeration
    (measured: ONE `add_local_connection` at width 3072 took 4.9s and pegged the run's main thread
    for hours on image rungs). This builds the adjacency ONCE and memoizes the full reach-set per
    queried target, so a whole pair sweep pays at most |distinct targets| BFS walks. Answers are
    exactly `would_create_cycle`'s: a forward edge in -> out closes a cycle iff out already
    reaches in, or they are equal. The snapshot goes stale if the genome gains forward edges;
    build a fresh instance after structural appends."""

    def __init__(self, genome: Genome) -> None:
        self._adjacency: dict[int, list[int]] = {}
        for conn in genome.forward_connections():
            self._adjacency.setdefault(conn.in_id, []).append(conn.out_id)
        for source, target in macro_implied_edges(genome):
            self._adjacency.setdefault(source, []).append(target)
        self._reach: dict[int, set[int]] = {}

    def creates_cycle(self, in_id: int, out_id: int) -> bool:
        if in_id == out_id:
            return True
        reach = self._reach.get(out_id)
        if reach is None:
            reach = self._reach_from(out_id)
            self._reach[out_id] = reach
        return in_id in reach

    def _reach_from(self, start: int) -> set[int]:
        queue = deque([start])
        seen = {start}
        while queue:
            current = queue.popleft()
            for nxt in self._adjacency.get(current, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen


def would_create_cycle(genome: Genome, in_id: int, out_id: int) -> bool:
    """True if adding a FORWARD edge `in_id -> out_id` would make the forward graph cyclic.

    A cycle appears iff `out_id` can already reach `in_id` along enabled forward edges (or they are
    equal). Recurrent edges are time-delayed and never participate. Single-shot form with an
    early-exit BFS. Batch callers should reuse `ForwardReachability` to avoid rebuilding the graph."""
    if in_id == out_id:
        return True
    adjacency: dict[int, list[int]] = {}
    for conn in genome.forward_connections():
        adjacency.setdefault(conn.in_id, []).append(conn.out_id)
    for source, target in macro_implied_edges(genome):
        adjacency.setdefault(source, []).append(target)

    queue = deque([out_id])
    seen = {out_id}
    while queue:
        current = queue.popleft()
        if current == in_id:
            return True
        for nxt in adjacency.get(current, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def macro_implied_edges(genome: Genome) -> list[tuple[int, int]]:
    """The dataflow edges a macro placement implies (every input feeds every output), so the DAG
    helpers see macros as connectivity. Implied-only cycles cannot exist (output ids are freshly
    allocated at mutation time); only regular connections can close a cycle through a macro."""
    return [(source, target) for macro in genome.macros for source in macro.input_node_ids for target in macro.output_node_ids]


def coordinate_distance(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    """Euclidean distance between two node coordinates, or inf when they are incomparable.

    Coordinates from different banks/axis-signatures have different lengths (or are None); those
    pairs are incomparable, so geometry operators treat them as infinitely far and never wire them
    together. This is what keeps a binary bit and a continuous coordinate out of the same receptive
    field even though they share one growing topology.
    """
    if a is None or b is None or len(a) != len(b):
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def topological_order(genome: Genome) -> list[int]:
    """Kahn's algorithm over enabled FORWARD edges (recurrent edges are time-delayed, not graph
    order). Raises ValueError on a cycle."""
    incoming: dict[int, int] = {node_id: 0 for node_id in genome.nodes}
    adjacency: dict[int, list[int]] = {}
    for conn in genome.forward_connections():
        adjacency.setdefault(conn.in_id, []).append(conn.out_id)
        incoming[conn.out_id] += 1
    for source, target in macro_implied_edges(genome):
        adjacency.setdefault(source, []).append(target)
        incoming[target] += 1

    ready = deque(sorted(node_id for node_id, count in incoming.items() if count == 0))
    order: list[int] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for nxt in adjacency.get(current, []):
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                ready.append(nxt)

    if len(order) != len(genome.nodes):
        raise ValueError("genome graph contains a cycle; cannot topologically order")
    return order


@dataclass
class InnovationTracker:
    """Allocates fresh node ids and stable per-edge innovation numbers for one evolver run."""

    _next_node_id: int
    _next_innovation: int = 0
    _edge_innovations: dict[tuple[int, int, bool], int] = field(default_factory=dict)

    @classmethod
    def from_genomes(cls, genomes: list[Genome]) -> "InnovationTracker":
        next_id = max((genome.max_node_id() for genome in genomes), default=-1) + 1
        max_innov = max(
            (conn.innovation for genome in genomes for conn in genome.connections),
            default=-1,
        )
        return cls(_next_node_id=next_id, _next_innovation=max_innov + 1)

    def new_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def new_marker(self) -> int:
        """A fresh one-off innovation number for unit genes (macro placements): allocated once at
        the mutation event, never looked up again, so the serialized edge map stays untouched."""
        marker = self._next_innovation
        self._next_innovation += 1
        return marker

    def innovation(self, in_id: int, out_id: int, recurrent: bool = False) -> int:
        """Same edge -> same innovation number, so crossover can align genes across genomes. A forward
        and a recurrent edge between the same pair are DIFFERENT edges and get distinct numbers."""
        key = (in_id, out_id, recurrent)
        if key not in self._edge_innovations:
            self._edge_innovations[key] = self._next_innovation
            self._next_innovation += 1
        return self._edge_innovations[key]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the id/innovation counters and the edge->innovation map (for checkpoint/resume)."""
        return {
            "next_node_id": self._next_node_id,
            "next_innovation": self._next_innovation,
            "edge_innovations": [[in_id, out_id, recurrent, innovation] for (in_id, out_id, recurrent), innovation in self._edge_innovations.items()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InnovationTracker":
        tracker = cls(_next_node_id=int(data["next_node_id"]), _next_innovation=int(data["next_innovation"]))
        edges: dict[tuple[int, int, bool], int] = {}
        for item in data["edge_innovations"]:
            if len(item) == 3:  # legacy checkpoints predate recurrence; their edges are all forward
                in_id, out_id, innovation = item
                recurrent = False
            else:
                in_id, out_id, recurrent, innovation = item
            edges[(int(in_id), int(out_id), bool(recurrent))] = int(innovation)
        tracker._edge_innovations = edges
        return tracker


def set_connection(genome: Genome, target: ConnectionGene) -> None:
    """Replace the gene for (in_id, out_id, recurrent) in place, or append it if new.

    The recurrent flag is part of the edge identity (matching `InnovationTracker.innovation` and
    `export_weights` keys): a forward and a recurrent gene on the same node pair are DIFFERENT
    genes and must never clobber each other."""
    for index, conn in enumerate(genome.connections):
        if conn.in_id == target.in_id and conn.out_id == target.out_id and conn.recurrent == target.recurrent:
            genome.connections[index] = target
            return
    genome.connections.append(target)


def make_acyclic(genome: Genome) -> Genome:
    """Return a copy whose enabled graph is a DAG, disabling any edge that would close a cycle.

    Recombination (innovation-aligned crossover) and re-enabling edges can introduce cycles into an
    otherwise feedforward genome; this repair keeps the substrate decodable. Edges are considered in
    order, so earlier (typically older) genes are preferred when a conflict arises.

    Two performance shapes, identical decisions (the module pool runs this per child per
    generation, so it must stay O(V+E) in the common case):
    - FAST PATH: already-acyclic genomes (the overwhelming majority) are detected with one
      `topological_order` pass over the exact same edge set the repair checks (enabled forward +
      macro-implied); the repair would disable nothing, so an unmodified copy is returned.
    - REPAIR: one incrementally-grown adjacency + an early-exit BFS per candidate edge, instead of
      rebuilding the adjacency per edge (the old per-edge `would_create_cycle` calls).
    """
    try:
        topological_order(genome)
    except ValueError:
        pass  # a cycle exists somewhere: run the ordered repair below
    else:
        return Genome(
            nodes=dict(genome.nodes), connections=list(genome.connections), macros=list(genome.macros), refine_steps=genome.refine_steps, operator_rates=dict(genome.operator_rates)
        )

    adjacency: dict[int, list[int]] = {}
    for source, target in macro_implied_edges(genome):
        adjacency.setdefault(source, []).append(target)

    def reaches(start: int, goal: int) -> bool:
        queue = deque([start])
        seen = {start}
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            for nxt in adjacency.get(current, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    repaired: list[ConnectionGene] = []
    for conn in genome.connections:
        if not conn.enabled or conn.recurrent:
            # Recurrent edges are time-delayed: cycles through them are legal, so they pass through.
            repaired.append(conn)
        elif conn.in_id == conn.out_id or reaches(conn.out_id, conn.in_id):
            repaired.append(replace(conn, enabled=False))
        else:
            adjacency.setdefault(conn.in_id, []).append(conn.out_id)
            repaired.append(conn)
    return Genome(nodes=dict(genome.nodes), connections=repaired, macros=list(genome.macros), refine_steps=genome.refine_steps, operator_rates=dict(genome.operator_rates))


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    """Serialize a genome to a plain dict (topology + weights), reloadable by `genome_from_dict`."""
    payload: dict[str, Any] = {
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind.value,
                "activation": node.activation,
                "coordinate": list(node.coordinate) if node.coordinate is not None else None,
                "aggregation": node.aggregation,
            }
            for node in genome.nodes.values()
        ],
        "connections": [
            {"in": conn.in_id, "out": conn.out_id, "weight": conn.weight, "enabled": conn.enabled, "innovation": conn.innovation, "recurrent": conn.recurrent}
            | ({"tie": conn.tie_group} if conn.tie_group is not None else {})  # absent when untied: old payloads stay byte-identical
            for conn in genome.connections
        ],
        "macros": [
            {"ref": macro.ref, "inputs": list(macro.input_node_ids), "outputs": list(macro.output_node_ids), "innovation": macro.innovation, "trainable": macro.trainable}
            for macro in genome.macros
        ],
        "refine_steps": genome.refine_steps,
    }
    if genome.operator_rates:  # absent when off, so fixed-rate genomes serialize byte-identically
        payload["operator_rates"] = dict(genome.operator_rates)
    return payload


def genome_from_dict(data: dict[str, Any]) -> Genome:
    """Rebuild a genome from the dict produced by `genome_to_dict`."""
    nodes: dict[int, NodeGene] = {}
    for node in data["nodes"]:
        node_id = int(node["id"])
        coordinate = tuple(node["coordinate"]) if node.get("coordinate") is not None else None
        nodes[node_id] = NodeGene(node_id, NodeKind(node["kind"]), node["activation"], coordinate, node.get("aggregation", "sum"))
    connections = [
        ConnectionGene(
            int(conn["in"]),
            int(conn["out"]),
            float(conn["weight"]),
            bool(conn["enabled"]),
            int(conn["innovation"]),
            bool(conn.get("recurrent", False)),
            int(conn["tie"]) if conn.get("tie") is not None else None,
        )
        for conn in data["connections"]
    ]
    macros = [
        MacroGene(
            ref=item["ref"],
            input_node_ids=tuple(int(node_id) for node_id in item["inputs"]),
            output_node_ids=tuple(int(node_id) for node_id in item["outputs"]),
            innovation=int(item["innovation"]),
            trainable=bool(item.get("trainable", False)),
        )
        for item in data.get("macros", [])
    ]
    operator_rates = {str(name): float(rate) for name, rate in data.get("operator_rates", {}).items()}
    return Genome(nodes=nodes, connections=connections, macros=macros, refine_steps=int(data.get("refine_steps", 1)), operator_rates=operator_rates)
