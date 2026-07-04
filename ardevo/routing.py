"""The routed substrate: a learned sparse mixture-of-experts over frozen library entries.

The library already holds the vertices (immutable module/composition entries); this module adds
the learned edges. Every vertex talks to a shared d_model residual bus through trainable per-vertex
adapters (the composition-glue idea applied once per vertex instead of once per edge, so growth is
an append, never a resize), and a task-conditioned gate selects a sparse top-k of experts per
routing step. Routing is iterated for a bounded number of steps, so module-to-module pathways,
including cycles, emerge as gate selections across steps, unrolled exactly like RefineGraphNet
unrolls refinement passes: termination is structural, never a convergence hope.

Integration is a single EVOLVE_STRATEGY named "routed" (registered in `ardevo/strategy.py`): the
orchestrator ladder, lookup, decompose, and the other strategies are untouched, so baseline vs
routed is a pure config A/B. Routed winners are NOT shelved in the library (a RoutedSolution is a
record, not a payload; the executable state lives in the router itself).
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ardevo.dataset.icarus import Task
from ardevo.evolution.composition import BIAS_REF, AssemblyContext, CompositionAssemblyError, assemble, comp_from_dict
from ardevo.evolution.genome import genome_from_dict
from ardevo.evolution.loop import AssessedComposition, CompositionGenome, CompTaskSpec
from ardevo.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary, macro_resolver, task_io
from ardevo.substrate import SubstrateModule, decode_module
from ardevo.utils.logging import Logger

logger = Logger.get_logger()

ROUTER_FORMAT_VERSION = 1
_TRAIN_HISTORY_CAP = 200


def sanitize_key(raw: str) -> str:
    """nn.ModuleDict / nn.ParameterDict keys must be attribute-safe; keep a reverse map in meta."""
    return re.sub(r"[^0-9A-Za-z_]", "_", raw)


def _is_temporal_signature(signature: str) -> bool:
    return "|" in signature and "T" in signature.split("|", 1)[1].split(",")


def _entry_widths(entry: LibraryEntry) -> tuple[int, int]:
    in_width = sum(int(item["width"]) for item in entry.io["inputs"])
    out_width = int(entry.io["output"]["width"])
    return in_width, out_width


def _bottleneck_linear(in_features: int, out_features: int, rank: int) -> nn.Module:
    """A rank-limited adapter (the anti-bypass lever): rank > 0 forces the direct path through a
    narrow bottleneck so the adapters alone cannot absorb the task and routing must earn its keep.
    rank 0 (or a rank that would not actually shrink the map) stays a plain dense Linear."""
    if 0 < rank < min(in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features))
    return nn.Linear(in_features, out_features)


@dataclass
class RouterVertex:
    """One expert: a frozen library entry decoded once and wired to the bus by its adapters."""

    original_key: str
    sanitized_key: str
    in_width: int
    out_width: int
    module: SubstrateModule


def build_vertex(entry: LibraryEntry, library: ModuleLibrary, *, max_inline_depth: int = 4) -> RouterVertex | None:
    """Decode/assemble an entry into a frozen expert, or None when it cannot serve as one here
    (temporal, undecodable): the same tolerance `_quick_metric` extends to library candidates."""
    if _is_temporal_signature(entry.io["inputs"][0].get("signature", "")):
        return None
    in_width, out_width = _entry_widths(entry)
    try:
        if entry.entry_type == MODULE:
            module: SubstrateModule = decode_module(genome_from_dict(entry.payload), in_width, out_width, macro_resolver=macro_resolver(library))
        elif entry.entry_type == COMPOSITION:
            comp = comp_from_dict(entry.payload)
            # Positional bank columns: consecutive ranges per non-bias INPUT node in id order, the
            # exact `_nested_context` convention a referenced composition reads its input by.
            columns: dict[str, list[int]] = {}
            cursor = 0
            for node_id in comp.input_ids:
                node = comp.nodes[node_id]
                if node.ref == BIAS_REF:
                    continue
                columns[node.ref] = list(range(cursor, cursor + node.out_width))
                cursor += node.out_width
            ctx = AssemblyContext(bank_columns=columns, library=library, max_inline_depth=max_inline_depth)
            module = assemble(comp, ctx, in_width)
        else:
            return None
    except (ValueError, CompositionAssemblyError, KeyError) as error:
        logger.debug("library entry %s not routable: %s", entry.key, error)
        return None
    for parameter in module.parameters():
        parameter.requires_grad_(False)  # experts are frozen; only adapters/gate/heads learn
    return RouterVertex(original_key=entry.key, sanitized_key=sanitize_key(entry.key), in_width=in_width, out_width=out_width, module=module)


class RoutedNet(nn.Module):
    """The shared-bus MoE. Persistent state is everything trainable (vertex adapters + embeddings,
    signature input adapters, output heads, gate, norm); experts are rebuilt from the library.

    Growth is append-only by construction: the gate scores vertices by an embedding dot product
    (never a Linear onto V outputs), and every dict is keyed by sanitized library key or signature,
    so a grown state_dict is a superset of the old one and old rows stay byte-identical.
    """

    def __init__(
        self,
        *,
        d_model: int,
        top_k: int,
        max_steps: int,
        adapter_rank: int = 0,
        expert_ablation: str = "none",
        halting: bool = False,
        ponder_epsilon: float = 0.01,
        ponder_cost: float = 0.001,
        edge_bias: bool = False,
        edge_dim: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.max_steps = max_steps
        self.adapter_rank = adapter_rank
        self.expert_ablation = expert_ablation
        self.halting = halting
        self.ponder_epsilon = ponder_epsilon
        self.ponder_cost = ponder_cost
        self.edge_bias = edge_bias
        self.edge_dim = edge_dim
        self.vertex_in_adapters = nn.ModuleDict()
        self.vertex_out_adapters = nn.ModuleDict()
        self.vertex_embeddings = nn.ParameterDict()
        # Explicit pathway structure (Stage C): edge_bias[i, j] = edge_out[i] . edge_in[j], a
        # factorized form of the V x V bias matrix that keeps growth append-only (a raw [V, V]
        # Parameter would need a resize on every new vertex, breaking the persistence invariant).
        self.vertex_edge_out = nn.ParameterDict()
        self.vertex_edge_in = nn.ParameterDict()
        self.input_adapters = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        self.output_signature_embeddings = nn.ParameterDict()
        self.task_condition = nn.Linear(2 * d_model, d_model)
        self.gate_projection = nn.Linear(2 * d_model, d_model)
        self.bus_norm = nn.LayerNorm(d_model)
        self.halt_head = nn.Linear(d_model, 1) if halting else None
        self._vertices: dict[str, RouterVertex] = {}  # sanitized key -> expert (rebuilt, not persisted)
        self._vertex_order: list[str] = []  # sanitized keys, insertion order = persisted row order
        self._retired: set[str] = set()
        self.input_adapter_widths: dict[str, int] = {}  # sanitized key -> raw width (for meta/rebuild)
        self.output_head_widths: dict[str, int] = {}
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        self.last_gate_stats: dict[str, float] = {}
        self.last_selections: list[torch.Tensor] = []  # per step: [batch, k] selected vertex positions
        self.last_probs: list[torch.Tensor] = []  # per step: [batch, k] gate probs (each row sums to 1)
        self.last_trace: torch.Tensor | None = None  # [batch, steps_run, out_width] when collected
        self.last_expected_steps: float = 0.0  # halting: mean expected steps (the ponder diagnostic)

    # --- vertex table -------------------------------------------------------------------------------

    def register_vertex(self, vertex: RouterVertex) -> None:
        key = vertex.sanitized_key
        if key in self.vertex_embeddings:
            self._vertices[key] = vertex  # adapters/embedding already exist (e.g. rebuilt after load)
            return
        if len(self.vertex_embeddings) == 0:
            embedding = torch.randn(self.d_model) * 0.02
        else:
            # A newcomer starts "roughly average" until evidence arrives: mean of peers plus noise.
            mean = torch.stack([self.vertex_embeddings[name].detach() for name in self._vertex_order]).mean(0)
            embedding = mean + torch.randn(self.d_model) * 0.02
        self.vertex_embeddings[key] = nn.Parameter(embedding)
        self.vertex_in_adapters[key] = _bottleneck_linear(self.d_model, vertex.in_width, self.adapter_rank)
        self.vertex_out_adapters[key] = _bottleneck_linear(vertex.out_width, self.d_model, self.adapter_rank)
        if self.edge_bias:
            self.vertex_edge_out[key] = nn.Parameter(torch.randn(self.edge_dim) * 0.02)
            self.vertex_edge_in[key] = nn.Parameter(torch.randn(self.edge_dim) * 0.02)
        self._vertices[key] = vertex
        self._vertex_order.append(key)

    def sync_with_library(self, library: ModuleLibrary, *, include_compositions: bool = True, exclude_temporal: bool = True) -> int:
        """Append vertices for library keys not yet in the table; refresh the retired mask. Returns
        the number of new vertices. The vertex set only grows (entries tombstone, never delete)."""
        added = 0
        known = {vertex.original_key for vertex in self._vertices.values()}
        for summary in library.summaries(include_retired=True):
            key = summary["key"]
            sanitized = sanitize_key(key)
            if summary.get("retired", False):
                if sanitized in self.vertex_embeddings:
                    self._retired.add(sanitized)
                continue
            if key in known:
                self._retired.discard(sanitized)
                continue
            entry = library.load(key)
            if entry.entry_type == COMPOSITION and not include_compositions:
                continue
            if exclude_temporal and _is_temporal_signature(entry.io["inputs"][0].get("signature", "")):
                continue
            vertex = build_vertex(entry, library)
            if vertex is None:
                continue
            self.register_vertex(vertex)
            added += 1
        return added

    # --- lazy per-signature surfaces ------------------------------------------------------------------

    def ensure_input_adapter(self, signature: str, width: int) -> str:
        key = sanitize_key(f"{signature}:{width}")
        if key not in self.input_adapters:
            self.input_adapters[key] = _bottleneck_linear(width, self.d_model, self.adapter_rank)
            self.input_adapter_widths[key] = width
        return key

    def ensure_output_head(self, signature: str, width: int) -> str:
        key = sanitize_key(f"{signature}:{width}")
        if key not in self.output_heads:
            self.output_heads[key] = nn.Linear(self.d_model, width)  # heads stay dense: they must express the output
            self.output_head_widths[key] = width
            self.output_signature_embeddings[key] = nn.Parameter(torch.randn(self.d_model) * 0.02)
        return key

    def task_embedding(self, support_input: torch.Tensor, input_key: str, head_key: str) -> torch.Tensor:
        pooled = self.input_adapters[input_key](support_input).mean(dim=0)
        return torch.tanh(self.task_condition(torch.cat([pooled, self.output_signature_embeddings[head_key]])))

    # --- routing --------------------------------------------------------------------------------------

    def _live_mask(self) -> torch.Tensor:
        return torch.tensor([name not in self._retired for name in self._vertex_order], dtype=torch.bool)

    def route(self, x: torch.Tensor, *, input_key: str, head_key: str, task_embed: torch.Tensor, collect_trace: bool = False) -> torch.Tensor:
        """Bounded bus updates, then the head readout. Cycles in the module graph are legal (A at
        step 1 can feed B at step 2 which feeds A at step 3); every pass executes at most
        `max_steps` finite updates, so it cannot loop forever. With `halting` on, a learned
        geometric halting head weights per-step readouts (stop mass p_t times the survival product)
        and execution EXITS EARLY once every sample has spent its halting mass: halting only ever
        shortens the unroll under the same hard cap."""
        input_inject = self.input_adapters[input_key](x)
        h = input_inject
        aux_terms: list[torch.Tensor] = []
        usage: dict[str, float] = {}
        self.last_selections = []
        self.last_probs = []
        step_readouts: list[torch.Tensor] = []
        halt_weights: list[torch.Tensor] = []
        survival = torch.ones(x.shape[0], 1, dtype=x.dtype)
        live_names = [name for name in self._vertex_order if name not in self._retired]
        previous_selection: torch.Tensor | None = None
        for step in range(self.max_steps):
            if live_names:
                moe, aux, step_usage, previous_selection = self._moe_step(h, task_embed, live_names, previous_selection)
                aux_terms.append(aux)
                for name, value in step_usage.items():
                    usage[name] = usage.get(name, 0.0) + value
            else:
                moe = torch.zeros_like(h)
            h = self.bus_norm(h + input_inject + moe)
            if collect_trace or self.halting:
                step_readouts.append(self.output_heads[head_key](h))
            if self.halting and self.halt_head is not None:
                last = step == self.max_steps - 1
                stop_mass = survival if last else torch.sigmoid(self.halt_head(h)) * survival  # the cap absorbs leftover mass
                halt_weights.append(stop_mass)
                survival = survival - stop_mass
                if bool((survival <= self.ponder_epsilon).all()):
                    break  # every sample has halted: the unroll genuinely shortens
        self.last_aux_loss = torch.stack(aux_terms).mean() if aux_terms else torch.zeros(())
        if self.halting and halt_weights:
            weights = torch.stack(halt_weights, dim=1)  # [B, steps_run, 1]
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            expected_steps = (weights.squeeze(-1) * torch.arange(1, weights.shape[1] + 1, dtype=x.dtype)).sum(dim=1).mean()
            self.last_aux_loss = self.last_aux_loss + self.ponder_cost * expected_steps
            self.last_expected_steps = float(expected_steps.detach())
            output = (torch.stack(step_readouts, dim=1) * weights).sum(dim=1)
        else:
            output = self.output_heads[head_key](h)
        self.last_trace = torch.stack(step_readouts, dim=1) if step_readouts else None
        total = sum(usage.values()) or 1.0
        self.last_gate_stats = {name: value / total for name, value in usage.items()}
        return output

    def routing_trace(self, x: torch.Tensor, *, input_key: str, head_key: str, task_embed: torch.Tensor) -> torch.Tensor:
        """Per-step head readouts, [batch, steps_run, output_width]: the `refine_trace` analogue for
        deep supervision (a `support_loss_deep`-shaped loss over routing steps)."""
        self.route(x, input_key=input_key, head_key=head_key, task_embed=task_embed, collect_trace=True)
        assert self.last_trace is not None
        return self.last_trace

    def _moe_step(
        self, h: torch.Tensor, task_embed: torch.Tensor, live_names: list[str], previous_selection: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.Tensor]:
        batch = h.shape[0]
        embeddings = torch.stack([self.vertex_embeddings[name] for name in live_names])  # [V, d]
        query = self.gate_projection(torch.cat([h, task_embed.expand(batch, -1)], dim=1))  # [B, d]
        logits = query @ embeddings.T  # [B, V]
        if self.edge_bias and previous_selection is not None:
            # Pathway prior: experts that fired last step bias which experts fire next.
            edge_out = torch.stack([self.vertex_edge_out[name] for name in live_names])  # [V, e]
            edge_in = torch.stack([self.vertex_edge_in[name] for name in live_names])  # [V, e]
            fired = edge_out[previous_selection].mean(dim=1)  # [B, e]
            logits = logits + fired @ edge_in.T
        k = min(self.top_k, len(live_names))
        top_values, top_indices = logits.topk(k, dim=1)
        top_probs = torch.softmax(top_values, dim=1)  # [B, k]
        self.last_selections.append(top_indices.detach())
        self.last_probs.append(top_probs.detach())
        # Switch-style load balance: importance (mean softmax mass) x load (fraction of selections).
        full_probs = torch.softmax(logits, dim=1)
        selection_counts = torch.zeros(len(live_names), dtype=h.dtype).scatter_add_(0, top_indices.reshape(-1), torch.ones(batch * k, dtype=h.dtype))
        load = selection_counts / (batch * k)
        aux = len(live_names) * (full_probs.mean(dim=0) * load).sum()

        moe = torch.zeros_like(h)
        usage: dict[str, float] = {}
        for vertex_position in torch.unique(top_indices):
            index = int(vertex_position)
            name = live_names[index]
            selected = top_indices == index  # [B, k]
            weight = (top_probs * selected).sum(dim=1, keepdim=True)  # [B, 1], zero for non-selectors
            vertex = self._vertices[name]
            expert_in = self.vertex_in_adapters[name](h)
            expert_out = torch.zeros(batch, vertex.out_width, dtype=h.dtype) if self.expert_ablation == "zero" else vertex.module(expert_in)
            moe = moe + weight * self.vertex_out_adapters[name](expert_out)
            usage[name] = float(weight.detach().sum())
        return moe, aux, usage, top_indices


class RoutedTaskView(nn.Module):
    """Binds the router to ONE task's keys and support tensor, exposing a plain `forward(x)`, so the
    unchanged `evaluate` / `support_loss` machinery scores the router like any substrate."""

    def __init__(self, net: RoutedNet, *, input_key: str, head_key: str, support_input: torch.Tensor) -> None:
        super().__init__()
        self.net = net
        self.input_key = input_key
        self.head_key = head_key
        self.support_input = support_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Task conditioning recomputes from the support set each pass, so it tracks adapter training.
        task_embed = self.net.task_embedding(self.support_input, self.input_key, self.head_key)
        return self.net.route(x, input_key=self.input_key, head_key=self.head_key, task_embed=task_embed)


@dataclass(frozen=True)
class RoutedSolution:
    """A RECORD of a routed win, not an executable payload: the state lives in the persisted router."""

    router_version: int
    input_key: str
    head_key: str
    zero_shot: bool
    zero_shot_metric: float
    trained_metric: float
    steps_used: int
    expert_usage: dict[str, float] = field(default_factory=dict)


class RouterService:
    """Owns the persistent RoutedNet, its growth against the library, and the disk round-trip."""

    def __init__(
        self,
        library: ModuleLibrary,
        *,
        d_model: int,
        top_k: int,
        max_steps: int,
        adapter_rank: int = 0,
        expert_ablation: str = "none",
        halting: bool = False,
        ponder_epsilon: float = 0.01,
        ponder_cost: float = 0.001,
        edge_bias: bool = False,
        persist_dir: Path | None = None,
        persist_strict: bool = False,
        image_dir: Path | None = None,
    ) -> None:
        self.library = library
        self.persist_dir = persist_dir
        self.persist_strict = persist_strict
        self.image_dir = image_dir  # <library_dir>/images; where overmind.png lands
        self.adapter_rank = adapter_rank
        self.version = 0
        self.train_history: list[dict[str, Any]] = []
        self._rendered_vertex_count = -1  # forces the first overmind render once vertices exist
        self.net = RoutedNet(
            d_model=d_model,
            top_k=top_k,
            max_steps=max_steps,
            adapter_rank=adapter_rank,
            expert_ablation=expert_ablation,
            halting=halting,
            ponder_epsilon=ponder_epsilon,
            ponder_cost=ponder_cost,
            edge_bias=edge_bias,
        )
        if persist_dir is not None and (persist_dir / "router_meta.json").exists():
            self._load(persist_dir)

    def sync(self, *, include_compositions: bool = True, exclude_temporal: bool = True) -> int:
        added = self.net.sync_with_library(self.library, include_compositions=include_compositions, exclude_temporal=exclude_temporal)
        if added:
            self.render_overmind()  # a new expert is the "significant addition" that refreshes the portrait
        return added

    def render_overmind(self) -> None:
        """Draw the WHOLE routed model (every expert embedded, wired to the shared bus) to
        <library_dir>/images/overmind.png. Refreshed on structural growth; never raises, so a bad
        render can no more kill a run than a library-entry render can."""
        if self.image_dir is None:
            return
        live = len(self.net._vertex_order)
        if live == self._rendered_vertex_count:
            return
        try:
            from ardevo.rendering import OvermindVertex, OvermindView, render_overmind

            vertices = [
                OvermindVertex(
                    key=vertex.original_key,
                    label=f"{vertex.original_key}  ({self.net.last_gate_stats.get(name, 0.0):.0%})" + ("  retired" if name in self.net._retired else ""),
                    retired=name in self.net._retired,
                )
                for name, vertex in ((name, self.net._vertices[name]) for name in self.net._vertex_order)
            ]
            view = OvermindView(
                vertices=vertices,
                input_signatures=list(self.net.input_adapter_widths),
                output_signatures=list(self.net.output_head_widths),
                d_model=self.net.d_model,
                top_k=self.net.top_k,
                max_steps=self.net.max_steps,
            )
            render_overmind(self.image_dir / "overmind.png", view, library=self.library)
            self._rendered_vertex_count = live
        except Exception as error:  # a render must never break a run
            logger.debug("overmind render skipped: %s", error)

    def record_task(self, row: dict[str, Any]) -> None:
        self.train_history.append(row)
        del self.train_history[:-_TRAIN_HISTORY_CAP]

    # --- persistence ----------------------------------------------------------------------------------

    def _meta(self) -> dict[str, Any]:
        vertex_keys = [self.net._vertices[name].original_key for name in self.net._vertex_order]
        return {
            "format_version": ROUTER_FORMAT_VERSION,
            "d_model": self.net.d_model,
            "top_k": self.net.top_k,
            "max_steps": self.net.max_steps,
            "adapter_rank": self.adapter_rank,
            "halting": self.net.halting,
            "edge_bias": self.net.edge_bias,
            "version": self.version,
            "vertex_keys": vertex_keys,
            "sanitized_key_map": {sanitize_key(key): key for key in vertex_keys},
            "input_adapter_keys": [{"key": key, "width": width} for key, width in self.net.input_adapter_widths.items()],
            "output_head_keys": [{"key": key, "width": width} for key, width in self.net.output_head_widths.items()],
            "train_history": self.train_history,
        }

    def save(self) -> None:
        if self.persist_dir is None:
            return
        self.version += 1
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        state_tmp = self.persist_dir / "router_state.pt.tmp"
        torch.save(self.net.state_dict(), state_tmp)
        os.replace(state_tmp, self.persist_dir / "router_state.pt")
        meta_tmp = self.persist_dir / "router_meta.json.tmp"
        meta_tmp.write_text(json.dumps(self._meta(), indent=2))
        os.replace(meta_tmp, self.persist_dir / "router_meta.json")

    def _load(self, directory: Path) -> None:
        meta = json.loads((directory / "router_meta.json").read_text())
        expected = {
            "d_model": self.net.d_model,
            "top_k": self.net.top_k,
            "max_steps": self.net.max_steps,
            "adapter_rank": self.adapter_rank,
            "halting": self.net.halting,
            "edge_bias": self.net.edge_bias,
        }
        mismatched = [name for name, value in expected.items() if int(meta.get(name, -1)) != int(value)]
        if int(meta.get("format_version", 0)) != ROUTER_FORMAT_VERSION or mismatched:
            message = f"router state at {directory} does not match config (format/{','.join(mismatched) or 'version'})"
            if self.persist_strict:
                raise ValueError(message)
            stale = directory.parent / f"router_stale_{int(time.time())}"
            logger.warning("%s; starting fresh (old state moved to %s)", message, stale)
            os.replace(directory, stale)
            return
        # Rebuild the skeleton exactly as saved (rows in order, lazy surfaces recreated), THEN load.
        for key in meta["vertex_keys"]:
            entry = self.library.load(key)  # tombstoned entries still load: rows never dangle
            vertex = build_vertex(entry, self.library)
            if vertex is None:
                raise ValueError(f"persisted router vertex {key!r} no longer decodes; state is unusable")
            self.net.register_vertex(vertex)
        for item in meta["input_adapter_keys"]:
            self.net.input_adapters[item["key"]] = _bottleneck_linear(int(item["width"]), self.net.d_model, self.net.adapter_rank)
            self.net.input_adapter_widths[item["key"]] = int(item["width"])
        for item in meta["output_head_keys"]:
            self.net.output_heads[item["key"]] = nn.Linear(self.net.d_model, int(item["width"]))
            self.net.output_head_widths[item["key"]] = int(item["width"])
            self.net.output_signature_embeddings[item["key"]] = nn.Parameter(torch.zeros(self.net.d_model))
        self.net.load_state_dict(torch.load(directory / "router_state.pt", weights_only=True), strict=True)
        self.version = int(meta.get("version", 0))
        self.train_history = list(meta.get("train_history", []))
        self.net.sync_with_library(self.library)  # append anything admitted since the last save


@dataclass
class RoutedStrategy:
    """The "routed" EVOLVE_STRATEGY: zero-shot first (the router's library-hit analogue), then a
    bounded gradient fit of adapters/gate/heads on the support set. Experts never train."""

    library_dir: str
    d_model: int = 64
    top_k: int = 2
    max_steps: int = 4
    train_steps: int = 200
    lr: float = 0.003
    weight_decay: float = 0.0001
    adapter_rank: int = 0
    load_balance_weight: float = 0.01
    zero_shot_accept: bool = True
    generation_cost: int = 10
    replay_tasks: int = 8
    replay_every: int = 4
    include_compositions: bool = True
    exclude_temporal: bool = True
    persist: bool = True
    persist_strict: bool = False
    expert_ablation: str = "none"
    halting: bool = False
    ponder_epsilon: float = 0.01
    ponder_cost: float = 0.001
    edge_bias: bool = False
    name: str = "routed"
    service: RouterService | None = None
    _replay: list[tuple[Any, str, str, torch.Tensor]] = field(default_factory=list)

    def _service(self, library: ModuleLibrary) -> RouterService:
        if self.service is None:
            self.service = RouterService(
                library,
                d_model=self.d_model,
                top_k=self.top_k,
                max_steps=self.max_steps,
                adapter_rank=self.adapter_rank,
                expert_ablation=self.expert_ablation,
                halting=self.halting,
                ponder_epsilon=self.ponder_epsilon,
                ponder_cost=self.ponder_cost,
                edge_bias=self.edge_bias,
                persist_dir=(Path(self.library_dir) / "router") if self.persist else None,
                persist_strict=self.persist_strict,
                image_dir=Path(self.library_dir) / "images",  # overmind.png lands beside the entry renders
            )
        return self.service

    @staticmethod
    def _metrics_view(metrics: dict[str, float]) -> AssessedComposition:
        # The established stand-in shape (`_quick_metric` precedent) so runtime.metric_of just works.
        return AssessedComposition(comp=CompositionGenome(), metrics=metrics, fitness=metrics.get("support_accuracy", 0.0), net=None)

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: Any,
        *,
        budget: int,
        seed_comps: list | None = None,
        seed_entries: list | None = None,
    ) -> Any:
        from ardevo.evaluation import evaluate

        service = self._service(runtime.library)
        service.sync(include_compositions=self.include_compositions, exclude_temporal=self.exclude_temporal)
        io = task_io(task)
        net = service.net
        input_key = net.ensure_input_adapter(io["inputs"][0]["signature"], io["inputs"][0]["width"])
        head_key = net.ensure_output_head(io["output"]["signature"], io["output"]["width"])
        support_input, _descriptor = spec.encoded.support_input
        view = RoutedTaskView(net, input_key=input_key, head_key=head_key, support_input=support_input)

        zero_shot_metrics = evaluate(view, spec.encoded, spec.encoder)
        zero_shot_metric = runtime.metric_of(self._metrics_view(zero_shot_metrics))
        if self.zero_shot_accept and zero_shot_metric >= runtime.accept_threshold:
            return self._result(task, service, view, zero_shot_metrics, zero_shot_metric, zero_shot=True, steps_used=0, generations_used=0, runtime=runtime)

        steps_used = self._train(view, spec, runtime)
        metrics = evaluate(view, spec.encoded, spec.encoder)
        metric = runtime.metric_of(self._metrics_view(metrics))
        metrics["routed_zero_shot_metric"] = zero_shot_metric
        self._remember_for_replay(spec, input_key, head_key, support_input)
        return self._result(task, service, view, metrics, metric, zero_shot=False, steps_used=steps_used, generations_used=self.generation_cost, runtime=runtime)

    def _train(self, view: RoutedTaskView, spec: CompTaskSpec, runtime: Any) -> int:
        from ardevo.evaluation import evaluate, support_loss

        trainable = [parameter for parameter in view.net.parameters() if parameter.requires_grad]
        if not trainable:
            return 0
        optimizer = torch.optim.Adam(trainable, lr=self.lr, weight_decay=self.weight_decay)
        milestone = max(1, self.train_steps // 4)
        steps_run = 0
        for step in range(self.train_steps):
            optimizer.zero_grad()
            loss = support_loss(view, spec.encoded) + self.load_balance_weight * view.net.last_aux_loss
            if not torch.isfinite(loss):
                break  # the `gradient` op's non-finite bail: keep the last finite state
            loss.backward()
            optimizer.step()
            steps_run = step + 1
            if self._replay and step % self.replay_every == self.replay_every - 1:
                self._replay_step(optimizer)
            if runtime.on_generation is not None and steps_run % milestone == 0:
                metrics = evaluate(view, spec.encoded, spec.encoder)
                holder = self._metrics_view(metrics)
                runtime.on_generation(self.name, steps_run // milestone, holder, float(-loss.detach()))
        return steps_run

    def _replay_step(self, optimizer: torch.optim.Optimizer) -> None:
        from ardevo.evaluation import support_loss

        encoded, input_key, head_key, support_input = self._replay[int(torch.randint(len(self._replay), (1,)))]
        net = self.service.net if self.service is not None else None
        if net is None:
            return
        replay_view = RoutedTaskView(net, input_key=input_key, head_key=head_key, support_input=support_input)
        optimizer.zero_grad()
        loss = support_loss(replay_view, encoded)
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()

    def _remember_for_replay(self, spec: CompTaskSpec, input_key: str, head_key: str, support_input: torch.Tensor) -> None:
        if self.replay_tasks <= 0:
            return
        self._replay.append((spec.encoded, input_key, head_key, support_input))
        del self._replay[: -self.replay_tasks]

    def _result(
        self,
        task: Task,
        service: RouterService,
        view: RoutedTaskView,
        metrics: dict[str, float],
        metric: float,
        *,
        zero_shot: bool,
        steps_used: int,
        generations_used: int,
        runtime: Any,
    ) -> Any:
        from ardevo.strategy import StrategyResult

        solution = RoutedSolution(
            router_version=service.version,
            input_key=view.input_key,
            head_key=view.head_key,
            zero_shot=zero_shot,
            zero_shot_metric=float(metrics.get("routed_zero_shot_metric", metric if zero_shot else 0.0)),
            trained_metric=float(metric),
            steps_used=steps_used,
            expert_usage=dict(view.net.last_gate_stats),
        )
        if metric >= runtime.accept_threshold:
            service.record_task(
                {
                    "task": task.meta.name,
                    "rung": task.meta.rung,
                    "zero_shot": zero_shot,
                    "zero_shot_metric": solution.zero_shot_metric,
                    "metric": float(metric),
                    "steps": steps_used,
                }
            )
            service.save()
        stamped = dict(metrics)
        stamped["routed_steps_used"] = float(steps_used)
        return StrategyResult(strategy=self.name, metric=float(metric), generations_used=generations_used, champion_routed=solution, champion_metrics=stamped)


def build_routed_strategy(config: dict[str, Any]) -> RoutedStrategy:
    table = config.get("orchestrator", {}).get("routed", {}) or {}
    return RoutedStrategy(
        library_dir=str(config.get("orchestrator", {}).get("library_dir", "library")),
        d_model=int(table.get("d_model", 64)),
        top_k=int(table.get("top_k", 2)),
        max_steps=int(table.get("max_steps", 4)),
        train_steps=int(table.get("train_steps", 200)),
        lr=float(table.get("lr", 0.003)),
        weight_decay=float(table.get("weight_decay", 0.0001)),
        adapter_rank=int(table.get("adapter_rank", 0)),
        load_balance_weight=float(table.get("load_balance_weight", 0.01)),
        zero_shot_accept=bool(table.get("zero_shot_accept", True)),
        generation_cost=int(table.get("generation_cost", 10)),
        replay_tasks=int(table.get("replay_tasks", 8)),
        replay_every=int(table.get("replay_every", 4)),
        include_compositions=bool(table.get("include_compositions", True)),
        exclude_temporal=bool(table.get("exclude_temporal", True)),
        persist=bool(table.get("persist", True)),
        persist_strict=bool(table.get("persist_strict", False)),
        expert_ablation=str(table.get("expert_ablation", "none")),
        halting=bool(table.get("halting", False)),
        ponder_epsilon=float(table.get("ponder_epsilon", 0.01)),
        ponder_cost=float(table.get("ponder_cost", 0.001)),
        edge_bias=bool(table.get("edge_bias", False)),
    )
