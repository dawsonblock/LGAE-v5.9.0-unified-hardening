"""v5.11 Phase 0: regression test documenting the MPC-not-called defect.

MPC planner is instantiated in __init__ when horizon > 1 but step()
delegates to loop.step() which never uses it. MPC is dead code.

This test PASSES against v5.10, proving the defect exists.
After Phase 9, this test should be replaced with one that verifies
MPC actually affects the plan.
"""
from __future__ import annotations

from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.types import make_graph_buffers


def test_mpc_not_used_in_step():
    """MPC is instantiated but never called during step().

    step() delegates to loop.step() which doesn't use the MPC planner.
    """
    runtime_config = RuntimeConfig(mpc_horizon=3)
    runtime = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32),
        runtime_config=runtime_config,
    )

    # MPC is instantiated.
    assert runtime._mpc is not None, "MPC should be instantiated when horizon > 1"

    # Track if MPC.plan is called.
    mpc_called = False
    orig_plan = runtime._mpc.plan

    def _tracking_plan(*args, **kwargs):
        nonlocal mpc_called
        mpc_called = True
        return orig_plan(*args, **kwargs)

    runtime._mpc.plan = _tracking_plan

    # Run step.
    runtime.step()

    # DEFECT: MPC.plan is never called during step().
    assert not mpc_called, (
        "Expected MPC.plan to NOT be called during step() (the defect). "
        "This test should FAIL after Phase 9 wires MPC into plan()."
    )
