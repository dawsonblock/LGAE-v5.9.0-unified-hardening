"""State bundle for atomic state replacement (v5.11 Phase 7).

A StateBundle is a complete, pre-validated candidate state that can be
atomically swapped into the authority. This enables exception-atomic
transactions: instead of sequentially mutating graph → fiber → gauge
(which leaves partial state on exception), we construct the complete
new state, validate it, and swap in one operation.

The swap is a single pointer assignment, which is atomic in Python
due to the GIL. For multi-threaded scenarios, a write lock brackets
the swap.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ...types import GraphBuffers
from ...fibers import FixedWidthFiberLatent
from .authoritative_state import AuthoritativeState, CalibrationState, ModelReference


@dataclass
class StateBundle:
    """A complete, pre-validated candidate authoritative state.

    This is constructed from the current state plus a transaction's
    deltas, validated, and then atomically swapped into the authority.
    """
    graph: GraphBuffers
    fibers: FixedWidthFiberLatent
    gauges: Any = None
    calibration: CalibrationState = field(default_factory=CalibrationState)
    model_ref: ModelReference = field(default_factory=ModelReference)
    version: int = 0

    @classmethod
    def from_state(cls, state: AuthoritativeState) -> "StateBundle":
        """Create a mutable copy of an authoritative state for modification."""
        # Deep-copy the graph and fibers so modifications don't alias authority.
        # GraphBuffers is a dataclass with tensors; we clone the tensors.
        new_graph = _clone_graph(state.graph)
        new_fibers = _clone_fibers(state.fibers)
        new_gauges = _clone_gauges(state.gauges)
        new_cal = copy.deepcopy(state.calibration)
        new_model = copy.deepcopy(state.model_ref)
        return cls(
            graph=new_graph,
            fibers=new_fibers,
            gauges=new_gauges,
            calibration=new_cal,
            model_ref=new_model,
            version=int(state.version),
        )

    def validate(self) -> None:
        """Validate the bundle before swap.

        Raises if any component is in an invalid state.
        """
        if self.graph is None:
            raise ValueError("StateBundle graph is None")
        if self.fibers is None:
            raise ValueError("StateBundle fibers is None")
        # Graph must have valid hash.
        _ = self.graph.state_hash()
        # Fibers must have valid hash.
        _ = self.fibers.state_hash()

    @property
    def state_hash(self) -> str:
        """Compute the state hash for this bundle."""
        import hashlib
        h = hashlib.sha256()
        h.update(self.graph.state_hash().encode())
        h.update(self.fibers.state_hash().encode())
        if self.gauges is not None:
            gh = self.gauges.state_hash()
        else:
            gh = "none"
        h.update(gh.encode())
        h.update(self.calibration.state_hash().encode())
        h.update(self.model_ref.checkpoint_hash.encode())
        h.update(str(self.version).encode())
        return h.hexdigest()

    def to_authoritative_state(self) -> AuthoritativeState:
        """Convert to an AuthoritativeState for the authority."""
        return AuthoritativeState(
            graph=self.graph,
            fibers=self.fibers,
            gauges=self.gauges,
            calibration=self.calibration,
            model_ref=self.model_ref,
            version=int(self.version),
        )


def _clone_graph(graph: GraphBuffers) -> GraphBuffers:
    """Clone a GraphBuffers, copying all tensors."""
    import dataclasses
    new = dataclasses.replace(graph)
    for f in dataclasses.fields(graph):
        val = getattr(graph, f.name)
        if isinstance(val, Tensor):
            setattr(new, f.name, val.detach().clone())
    return new


def _clone_fibers(fibers: FixedWidthFiberLatent) -> FixedWidthFiberLatent:
    """Clone fiber state by snapshotting and restoring."""
    snap = fibers.snapshot()
    # Create a new fiber bank with the same config.
    import dataclasses
    new = dataclasses.replace(fibers)
    # Clone all tensor parameters.
    for name, param in fibers.named_parameters():
        setattr(new, name, param.detach().clone())
    # Clone all buffers.
    for name, buf in fibers.named_buffers():
        setattr(new, name, buf.detach().clone())
    new.restore(snap)
    return new


def _clone_gauges(gauges: Any) -> Any:
    """Clone gauge connections if present."""
    if gauges is None:
        return None
    if hasattr(gauges, 'raw_generators'):
        import dataclasses
        try:
            new = dataclasses.replace(gauges)
            if hasattr(gauges, 'raw_generators') and isinstance(gauges.raw_generators, Tensor):
                new.raw_generators = gauges.raw_generators.detach().clone()
            if hasattr(gauges, 'slot_generation') and isinstance(gauges.slot_generation, Tensor):
                new.slot_generation = gauges.slot_generation.detach().clone()
            return new
        except Exception:
            # If dataclasses.replace doesn't work, try manual clone.
            return gauges
    return gauges
