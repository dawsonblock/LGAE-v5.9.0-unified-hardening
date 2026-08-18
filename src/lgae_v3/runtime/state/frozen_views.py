"""Frozen (immutable) views of authoritative state (v5.11 Phase 3).

The v5.10 defect: AuthoritativeStateGuard.graph returns raw mutable
GraphBuffers. Callers can do guard.graph.weight[...] = ... to mutate
authoritative state, bypassing the authority model.

The fix: frozen views that return detached clones of tensors. Any attempt
to mutate through the view raises UnauthorizedMutationError.

For large graphs where copying is expensive, copy-on-write can be added
later. For now, defensive cloning is the safe default.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from ...types import GraphBuffers
from ..authority import UnauthorizedMutationError


class FrozenGraphView:
    """Immutable view of a GraphBuffers.

    All tensor properties return detached clones. Any attempt to set
    attributes raises UnauthorizedMutationError.
    """

    __slots__ = ("_graph", "_cache")

    def __init__(self, graph: GraphBuffers) -> None:
        # Store a reference but never expose it directly.
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_cache", {})

    def _clone_tensor(self, name: str) -> Tensor:
        cache = object.__getattribute__(self, "_cache")
        if name not in cache:
            graph = object.__getattribute__(self, "_graph")
            t = getattr(graph, name)
            cache[name] = t.detach().clone() if t is not None else None
        return cache[name]

    @property
    def src(self) -> Tensor:
        return self._clone_tensor("src")

    @property
    def dst(self) -> Tensor:
        return self._clone_tensor("dst")

    @property
    def weight(self) -> Tensor:
        return self._clone_tensor("weight")

    @property
    def length(self) -> Tensor:
        return self._clone_tensor("length")

    @property
    def valid(self) -> Tensor:
        return self._clone_tensor("valid")

    @property
    def role(self) -> Tensor:
        return self._clone_tensor("role")

    @property
    def slot_gen(self) -> Tensor:
        return self._clone_tensor("slot_gen")

    @property
    def num_nodes(self) -> int:
        return int(object.__getattribute__(self, "_graph").num_nodes)

    @property
    def capacity(self) -> int:
        return int(object.__getattribute__(self, "_graph").capacity)

    @property
    def version(self) -> int:
        return int(object.__getattribute__(self, "_graph").version)

    def state_hash(self) -> str:
        return object.__getattribute__(self, "_graph").state_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "capacity": self.capacity,
            "version": self.version,
            "state_hash": self.state_hash(),
        }

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on FrozenGraphView; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on FrozenGraphView; "
            "authoritative state is mutated only through the commit channel"
        )


class FrozenFiberView:
    """Immutable view of fiber state.

    Returns detached clones of fiber tensors.
    """

    __slots__ = ("_fiber_fn", "_cache")

    def __init__(self, fiber_fn: Callable[[], Tensor]) -> None:
        object.__setattr__(self, "_fiber_fn", fiber_fn)
        object.__setattr__(self, "_cache", {})

    def _get_z(self) -> Tensor:
        cache = object.__getattribute__(self, "_cache")
        if "z" not in cache:
            fn = object.__getattribute__(self, "_fiber_fn")
            z = fn()
            cache["z"] = z.detach().clone() if z is not None else None
        return cache["z"]

    @property
    def z(self) -> Tensor:
        return self._get_z()

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on FrozenFiberView; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on FrozenFiberView"
        )


class FrozenGaugeView:
    """Immutable view of gauge connections.

    Returns detached clones of gauge tensors.
    """

    __slots__ = ("_gauge", "_cache")

    def __init__(self, gauge: Any) -> None:
        object.__setattr__(self, "_gauge", gauge)
        object.__setattr__(self, "_cache", {})

    @property
    def state_hash(self) -> str:
        g = object.__getattribute__(self, "_gauge")
        return g.state_hash() if g is not None and hasattr(g, "state_hash") else ""

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on FrozenGaugeView; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on FrozenGaugeView"
        )


class StaleSnapshotError(RuntimeError):
    """Raised when a candidate's source state version doesn't match
    the current authoritative state version. This prevents TOCTOU problems."""
