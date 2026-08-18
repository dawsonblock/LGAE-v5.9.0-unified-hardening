"""v5.11 Phase 0: regression test documenting the production fail-open defect.

Production mode only checks require_signed_receipts + signing_key.
It does not check WAL, evidence store, checkpointing, or qualification.

This test PASSES against v5.10, proving the defect exists.
After Phase 11, this test should be replaced with one that verifies
production mode fails closed without all required components.
"""
from __future__ import annotations

from lgae_v3.runtime import LGAERuntime, RuntimeConfig, RuntimeMode
from lgae_v3.config import ProductionConfig
from lgae_v3.types import make_graph_buffers


def test_production_starts_without_wal():
    """Production mode starts without WAL, evidence, or checkpointing.

    This is a fail-open defect: production should require these.
    """
    config = ProductionConfig()
    runtime_config = RuntimeConfig(
        mode=RuntimeMode.PRODUCTION,
        # No evidence_path, no receipt_path, no signing_key.
        # Only require_signed_receipts is False, so it passes.
    )
    # DEFECT: this succeeds. It should fail.
    runtime = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=32),
        config=config,
        runtime_config=runtime_config,
    )
    # If we got here, production mode started without persistence.
    # This is the defect.
    assert runtime is not None, (
        "Expected production mode to start without WAL (the defect), but it failed. "
        "This test should FAIL after Phase 11 makes production fail-closed."
    )
