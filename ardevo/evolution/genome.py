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
    # Weight-tying tag: edges sharing a `share_group` decode to ONE shared parameter (a convolution
    # kernel reused across tiled placements). None = an independent per-edge weight (the default; the
    # substrate is byte-identical until a weight-tied motif is actually grown).
    share_group: int | None = None


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
    # Per-genome mutation-operator probabilities (the ACI "mutation as a meta-parameter"): when
    # self-adaptive rates are on, each genome carries its own {operator_name: prob} which is perturbed
    # and inherited (Evolution-Strategy self-adaptation), so the search tunes its own operator mix and
    # selection on the genome implicitly selects good rates. None = use the config base rates (the
    # default; the flat path is byte-identical until self-adaptation is enabled).
    operator_rates: dict[str, float] | None = None

    def clone(self) -> "Genome":
        # Gene dataclasses are frozen, so shallow copies of the containers are deep copies.
        return Genome(
            nodes=dict(self.nodes),
            connections=list(self.connections),
            macros=list(self.macros),
            refine_steps=self.refine_steps,
            operator_rates=dict(self.operator_rates) if self.operator_rates else None,
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
        return len(self.enabled_connections()) + len(self.hidden_ids) + len(self.macros)

    def max_node_id(self) -> int:
        return max(self.nodes) if self.nodes else -1


def would_create_cycle(genome: Genome, in_id: int, out_id: int) -> bool:
    """True if adding a FORWARD edge `in_id -> out_id` would make the forward graph cyclic.

    A cycle appears iff `out_id` can already reach `in_id` along enabled forward edges (or they are
    equal). Recurrent edges are time-delayed and never participate.
    """
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


def with_connection_weight(conn: ConnectionGene, weight: float) -> ConnectionGene:
    return replace(conn, weight=weight)


def make_acyclic(genome: Genome) -> Genome:
    """Return a copy whose enabled graph is a DAG, disabling any edge that would close a cycle.

    Recombination (innovation-aligned crossover) and re-enabling edges can introduce cycles into an
    otherwise feedforward genome; this repair keeps the substrate decodable. Edges are considered in
    order, so earlier (typically older) genes are preferred when a conflict arises.
    """
    kept = Genome(nodes=dict(genome.nodes), connections=[], macros=list(genome.macros))
    repaired: list[ConnectionGene] = []
    for conn in genome.connections:
        if not conn.enabled or conn.recurrent:
            # Recurrent edges are time-delayed: cycles through them are legal, so they pass through.
            repaired.append(conn)
        elif conn.in_id == conn.out_id or would_create_cycle(kept, conn.in_id, conn.out_id):
            repaired.append(replace(conn, enabled=False))
        else:
            kept.connections.append(conn)
            repaired.append(conn)
    return Genome(
        nodes=dict(genome.nodes),
        connections=repaired,
        macros=list(genome.macros),
        refine_steps=genome.refine_steps,
        operator_rates=dict(genome.operator_rates) if genome.operator_rates else None,
    )


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    """Serialize a genome to a plain dict (topology + weights), reloadable by `genome_from_dict`."""
    return {
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
            {
                "in": conn.in_id,
                "out": conn.out_id,
                "weight": conn.weight,
                "enabled": conn.enabled,
                "innovation": conn.innovation,
                "recurrent": conn.recurrent,
                "share_group": conn.share_group,
            }
            for conn in genome.connections
        ],
        "macros": [
            {"ref": macro.ref, "inputs": list(macro.input_node_ids), "outputs": list(macro.output_node_ids), "innovation": macro.innovation, "trainable": macro.trainable}
            for macro in genome.macros
        ],
        "refine_steps": genome.refine_steps,
        "operator_rates": genome.operator_rates,
    }


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
            None if conn.get("share_group") is None else int(conn["share_group"]),
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
    operator_rates = data.get("operator_rates")
    rates = {str(name): float(rate) for name, rate in operator_rates.items()} if operator_rates else None
    return Genome(nodes=nodes, connections=connections, macros=macros, refine_steps=int(data.get("refine_steps", 1)), operator_rates=rates)
