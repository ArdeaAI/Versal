"""Shared policy for following persistent library references.

The decoded or rendered root is depth 0 and does not consume the budget. Following one
``library:<key>`` edge consumes one level, so ``max_inline_depth=N`` permits exactly ``N``
reference boundaries on any root-to-leaf path. A value of zero therefore permits the root only.
Cycle detection is independent of this budget: revisiting any key on the active path always fails.
"""

from collections.abc import Mapping
from typing import Any

DEFAULT_MAX_INLINE_DEPTH = 4


def configured_max_inline_depth(config: Mapping[str, Any]) -> int:
    """Read the authoritative ``[evolution.composition]`` reference-depth limit."""
    evolution = config.get("evolution", {}) or {}
    composition = evolution.get("composition", {}) or {}
    return max(0, int(composition.get("max_inline_depth", DEFAULT_MAX_INLINE_DEPTH)))
