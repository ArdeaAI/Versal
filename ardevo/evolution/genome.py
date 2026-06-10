"""NEAT-style genome: a directed acyclic graph of typed nodes and weighted connections.

A genome describes a topology, not weights-in-isolation: connection weights ride along on
the genes so structure and weights co-evolve. The substrate decoder turns a genome into an
executable torch module; the mutation operators grow the graph from a minimal seed.
"""

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


@dataclass(frozen=True, slots=True)
class ConnectionGene:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innovation: int


@dataclass
class Genome:
    """A topology candidate. Node genes are keyed by id; connection genes are an ordered list."""

    nodes: dict[int, NodeGene] = field(default_factory=dict)
    connections: list[ConnectionGene] = field(default_factory=list)

    def clone(self) -> "Genome":
        # NodeGene/ConnectionGene are frozen, so a shallow copy of the containers is a deep copy.
        return Genome(nodes=dict(self.nodes), connections=list(self.connections))

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

    def has_connection(self, in_id: int, out_id: int) -> bool:
        return any(conn.in_id == in_id and conn.out_id == out_id for conn in self.connections)

    def complexity(self) -> int:
        """Structural cost: enabled edges plus hidden nodes. Drives the complexity penalty."""
        return len(self.enabled_connections()) + len(self.hidden_ids)

    def max_node_id(self) -> int:
        return max(self.nodes) if self.nodes else -1


def would_create_cycle(genome: Genome, in_id: int, out_id: int) -> bool:
    """True if adding `in_id -> out_id` would make the enabled graph cyclic.

    A cycle appears iff `out_id` can already reach `in_id` along enabled edges (or they are equal).
    """
    if in_id == out_id:
        return True
    adjacency: dict[int, list[int]] = {}
    for conn in genome.enabled_connections():
        adjacency.setdefault(conn.in_id, []).append(conn.out_id)

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


def topological_order(genome: Genome) -> list[int]:
    """Kahn's algorithm over enabled edges. Raises ValueError on a cycle."""
    incoming: dict[int, int] = {node_id: 0 for node_id in genome.nodes}
    adjacency: dict[int, list[int]] = {}
    for conn in genome.enabled_connections():
        adjacency.setdefault(conn.in_id, []).append(conn.out_id)
        incoming[conn.out_id] += 1

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
    _edge_innovations: dict[tuple[int, int], int] = field(default_factory=dict)

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

    def innovation(self, in_id: int, out_id: int) -> int:
        """Same edge -> same innovation number, so crossover can align genes across genomes."""
        key = (in_id, out_id)
        if key not in self._edge_innovations:
            self._edge_innovations[key] = self._next_innovation
            self._next_innovation += 1
        return self._edge_innovations[key]


def set_connection(genome: Genome, target: ConnectionGene) -> None:
    """Replace the gene for (in_id, out_id) in place, or append it if new."""
    for index, conn in enumerate(genome.connections):
        if conn.in_id == target.in_id and conn.out_id == target.out_id:
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
    kept = Genome(nodes=dict(genome.nodes), connections=[])
    repaired: list[ConnectionGene] = []
    for conn in genome.connections:
        if not conn.enabled:
            repaired.append(conn)
        elif conn.in_id == conn.out_id or would_create_cycle(kept, conn.in_id, conn.out_id):
            repaired.append(replace(conn, enabled=False))
        else:
            kept.connections.append(conn)
            repaired.append(conn)
    return Genome(nodes=dict(genome.nodes), connections=repaired)


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    """Serialize a genome to a plain dict (topology + weights), reloadable by `genome_from_dict`."""
    return {
        "nodes": [{"id": node.id, "kind": node.kind.value, "activation": node.activation} for node in genome.nodes.values()],
        "connections": [{"in": conn.in_id, "out": conn.out_id, "weight": conn.weight, "enabled": conn.enabled, "innovation": conn.innovation} for conn in genome.connections],
    }


def genome_from_dict(data: dict[str, Any]) -> Genome:
    """Rebuild a genome from the dict produced by `genome_to_dict`."""
    nodes = {int(node["id"]): NodeGene(int(node["id"]), NodeKind(node["kind"]), node["activation"]) for node in data["nodes"]}
    connections = [ConnectionGene(int(conn["in"]), int(conn["out"]), float(conn["weight"]), bool(conn["enabled"]), int(conn["innovation"])) for conn in data["connections"]]
    return Genome(nodes=nodes, connections=connections)
