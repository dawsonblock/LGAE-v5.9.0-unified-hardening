"""v5.11 Phase 0: regression test documenting the fake canonical path defect.

LGAERuntime.step() delegates all real work to StructuralLearningLoop.step().
The 8 phase methods (observe, reason, propose, plan, evaluate, authorize,
commit, learn) exist but are never called during step().

This test PASSES against v5.10, proving the defect exists.
After Phase 2, this test should be replaced with one that verifies
all 8 phases ARE called.
"""
from __future__ import annotations

import torch

from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.types import make_graph_buffers


def test_step_does_not_call_eight_phases():
    """The 8 phase methods are not called during step().

    step() calls self.loop.step() which does everything.
    The phase methods are decorative.
    """
    graph = make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32)
    runtime = LGAERuntime(graph, runtime_config=RuntimeConfig())

    # Track which phase methods are called.
    called_phases: list[str] = []
    orig_observe = runtime.observe
    orig_reason = runtime.reason
    orig_propose = runtime.propose
    orig_plan = runtime.plan
    orig_evaluate = runtime.evaluate
    orig_authorize = runtime.authorize
    orig_commit = runtime.commit
    orig_learn = runtime.learn

    def _wrap(name, orig):
        def wrapper(*args, **kwargs):
            called_phases.append(name)
            return orig(*args, **kwargs)
        return wrapper

    runtime.observe = _wrap("observe", orig_observe)
    runtime.reason = _wrap("reason", orig_reason)
    runtime.propose = _wrap("propose", orig_propose)
    runtime.plan = _wrap("plan", orig_plan)
    runtime.evaluate = _wrap("evaluate", orig_evaluate)
    runtime.authorize = _wrap("authorize", orig_authorize)
    runtime.commit = _wrap("commit", orig_commit)
    runtime.learn = _wrap("learn", orig_learn)

    # Run one step.
    runtime.step()

    # DEFECT: not all 8 phase methods are called during step().
    # step() calls self.observe() but then delegates to self.loop.step()
    # which does the real work. The other 7 phases are never called.
    all_phases = {"observe", "reason", "propose", "plan", "evaluate",
                  "authorize", "commit", "learn"}
    called_set = set(called_phases)
    assert called_set != all_phases, (
        f"Expected NOT all 8 phases to be called during step(), but got: {called_phases}. "
        "This test should FAIL after Phase 2 fixes the canonical path."
    )
