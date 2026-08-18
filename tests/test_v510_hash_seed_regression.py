"""v5.11 Phase 0: regression test documenting the hash() nondeterminism defect.

curriculum.py uses hash(family.value) which is non-deterministic across
PYTHONHASHSEED values.

This test PASSES against v5.10, proving the defect exists.
After Phase 14, this test should be replaced with one that verifies
deterministic seed derivation.
"""
from __future__ import annotations

import inspect

from lgae_v3.runtime.curriculum import CurriculumGenerator


def test_curriculum_hash_nondeterminism():
    """The curriculum generator uses hash() which is seed-dependent.

    hash() returns different values under different PYTHONHASHSEED settings.
    This makes curriculum generation non-deterministic across processes.
    """
    # The defect: hash() is used for seed derivation.
    # We verify the code path uses hash() by inspecting the source.
    source = inspect.getsource(CurriculumGenerator)
    assert "hash(" in source, (
        "Expected curriculum.py to use hash() (the defect). "
        "This test should FAIL after Phase 14 replaces hash() with SHA-256."
    )
