"""Crossover operators: an independent stage that combines two parents into one child.

`none` is asexual (clone the first parent). `neat` aligns genes by innovation number, inheriting
matching genes from either parent at random and disjoint/excess genes from the first parent.
"""

import random
from typing import Callable

from ardevo.evolution.genome import Genome, NodeGene
from ardevo.evolution.registry import Registry

CrossoverOp = Callable[..., Genome]

CROSSOVER: Registry[CrossoverOp] = Registry("crossover")


@CROSSOVER.register("none")
def asexual(parent_a: Genome, parent_b: Genome, *, rng: random.Random) -> Genome:
    return parent_a.clone()


@CROSSOVER.register("neat")
def neat(parent_a: Genome, parent_b: Genome, *, rng: random.Random) -> Genome:
    """Innovation-aligned crossover. `parent_a` is treated as the more-fit structural base."""
    by_innovation_b = {conn.innovation: conn for conn in parent_b.connections}

    child_connections = []
    for conn_a in parent_a.connections:
        conn_b = by_innovation_b.get(conn_a.innovation)
        child_connections.append(conn_a if (conn_b is None or rng.random() < 0.5) else conn_b)

    nodes: dict[int, NodeGene] = dict(parent_a.nodes)
    for conn in child_connections:
        for node_id in (conn.in_id, conn.out_id):
            if node_id not in nodes:
                nodes[node_id] = parent_b.nodes[node_id]
    # Macro placements are atomic unit genes: a matching marker in parent_b IS the same immutable
    # gene, so parent_a's list carries everything inheritable; disjoint macros from b drop exactly
    # like disjoint connections do. Self-adaptive rates (lever F) inherit from the fitter base too, so
    # the adaptive pipeline perturbs an inherited schedule rather than reseeding from the config each
    # generation; empty when off, keeping the scalar path byte-identical.
    return Genome(nodes=nodes, connections=child_connections, macros=list(parent_a.macros), operator_rates=dict(parent_a.operator_rates))
