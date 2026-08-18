"""v5.11 Phase 0: regression test documenting the IG-not-used defect.

structural_loop.py hardcodes information_gain=0.0, cost=0.0, risk=0.0.
The IG module exists but is never wired into action selection.

This test PASSES against v5.10, proving the defect exists.
After Phase 8, this test should be replaced with one that verifies
IG affects action selection.
"""
from __future__ import annotations

import inspect

from lgae_v3.structural_loop import StructuralLearningLoop


def test_ig_hardcoded_zero():
    """Information gain is hardcoded to 0.0 in the structural loop."""
    source = inspect.getsource(StructuralLearningLoop.step)

    # DEFECT: IG, cost, and risk are hardcoded to 0.0.
    assert "information_gain=0.0" in source, (
        "Expected information_gain=0.0 in structural_loop.py (the defect). "
        "This test should FAIL after Phase 8 activates IG."
    )
    assert "cost=0.0" in source, (
        "Expected cost=0.0 in structural_loop.py (the defect). "
        "This test should FAIL after Phase 8 activates cost."
    )
    assert "risk=0.0" in source, (
        "Expected risk=0.0 in structural_loop.py (the defect). "
        "This test should FAIL after Phase 8 activates risk."
    )
