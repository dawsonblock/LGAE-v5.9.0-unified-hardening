"""v5.11 Phase 0: regression test documenting the guard mutability defect.

AuthoritativeStateGuard.graph returns raw mutable GraphBuffers.
Callers can do guard.graph.weight[...] = ... to mutate authoritative state,
bypassing the authority model entirely.

This test PASSES against v5.10, proving the defect exists.
After Phase 3, this test should be replaced with one that verifies
the guard is truly immutable.
"""
from __future__ import annotations

import torch

from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.types import make_graph_buffers


def test_guard_graph_is_mutable():
    """The guard's graph property returns a mutable reference.

    A non-commit component can mutate authoritative state through it.
    """
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())

    guard = runtime.guard_for("executive")
    original_weight = guard.graph.weight.clone()

    # DEFECT: we can mutate authoritative state through the guard.
    guard.graph.weight[0] = guard.graph.weight[0] * 2.0

    # The authoritative state was mutated.
    assert not torch.equal(original_weight, guard.graph.weight), (
        "Expected guard.graph to be mutable (the defect), but it was immutable. "
        "This test should FAIL after Phase 3 fixes the state isolation."
    )
