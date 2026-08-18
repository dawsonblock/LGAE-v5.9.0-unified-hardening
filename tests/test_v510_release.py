"""v5.10 Phase 50: release verification tests."""
from __future__ import annotations

import pytest

from lgae_v3.version import VERSION, SCHEMA_VERSION, MANIFEST_SCHEMA


def test_version_is_5_10():
    assert VERSION == "5.11.0-dev"


def test_schema_version_updated():
    assert "V5_11_0" in SCHEMA_VERSION


def test_manifest_schema_updated():
    assert "V5_11_0" in MANIFEST_SCHEMA


def test_runtime_imports_complete():
    """Verify all runtime modules are importable."""
    from lgae_v3.runtime import (
        LGAERuntime, RuntimeConfig, RuntimeMode,
        Checkpoint, CheckpointChain,
        WALRecord, WriteAheadLog,
        DecisionRecord, DecisionLedger,
        RealGraphBenchmark, load_benchmark,
        SheafConsistencyResult, certify_sheaf_consistency,
        LieGroup, ManifoldAction, exponential_map,
        InformationGainEstimate, ensemble_disagreement_ig,
        EpistemicUncertaintyEstimate, estimate_epistemic_uncertainty,
        CalibrationMetrics, compute_calibration_metrics,
        MPCPlan, MPCPlanner,
        JointStructuralAction, make_joint_action,
        CreditAssignment, direct_credit,
        ReplayBuffer, ReplayTransition,
        HardNegative, HardNegativeMiner,
        OfflineRLTrainer, OfflineRLConfig,
        CausalCreditAssignment, CausalCreditAssigner,
        compute_degrees, build_sparse_graph,
        get_device, batched_message_passing,
        CounterfactualResult, batched_counterfactual_eval,
    )
    # All imports successful.


def test_phase_count():
    """Verify that all 50 phases (0-49 + 50) are represented."""
    # The git log should have 50+ phase commits.
    import subprocess
    result = subprocess.run(
        ["git", "log", "--oneline", "--grep=Phase"],
        capture_output=True, text=True,
    )
    phase_lines = [l for l in result.stdout.strip().split("\n") if "Phase" in l]
    assert len(phase_lines) >= 49  # at least 49 phase commits
