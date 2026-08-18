"""v5.11-RC Phase 11-12: Internal commit failpoints + SIGKILL crash matrix.

Tests crash recovery at internal commit boundaries using failpoints.
Each failpoint simulates a crash at a specific point in the commit path.

The central invariant: S_restart ∈ {S_t, S_{t+1}} — never a mixed state.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig, make_graph_transaction, StructuralTransaction
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime.wal import replay_committed_transactions
from lgae_v3.types import MutationResult


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


_CRASH_SCRIPT = textwrap.dedent("""
    import sys, os, json, signal
    sys.path.insert(0, {src_path!r})
    import torch
    from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
    from lgae_v3.runtime import LGAERuntime, RuntimeConfig, make_graph_transaction, StructuralTransaction
    from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
    from lgae_v3.types import MutationResult

    torch.manual_seed(42)
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 6; cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = 'exact_lp'; cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0; cfg.audit.bakry_nodes = 0; cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0; cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]; cfg.mutation.curvature_ema_enabled = False

    failpoint = {failpoint!r}
    wal_path = {wal_path!r}
    state_file = {state_file!r}

    rt = LGAERuntime(
        make_graph_buffers(6, [(0,1),(1,2),(2,3),(3,4),(4,5)], capacity=12),
        cfg, runtime_config=RuntimeConfig(wal_path=wal_path),
    )
    pre_hash = rt.authority_hash

    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    txn = make_graph_transaction(
        base_state_version=int(rt.engine.graph.version),
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    )
    full_txn = StructuralTransaction(
        transaction_id=txn.transaction_id,
        base_state_version=txn.base_state_version,
        base_state_hash=txn.base_state_hash,
        graph_delta=txn.graph_delta,
        authorization_id=txn.authorization_binding_hash(),
        delta_hash=txn.delta_hash,
        mutation_result=txn.mutation_result,
    )
    auth = AuthorizationResult(
        snapshot_id='s1', state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=full_txn.transaction_id,
    )

    if failpoint == 'none':
        result = rt.commit_channel.commit(full_txn, auth)
        with open(state_file, 'w') as f:
            json.dump({{'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                       'committed': result.committed}}, f)
    else:
        # Set the failpoint and kill the process when it triggers.
        rt.commit_channel.set_failpoint(failpoint)
        # Replace the failpoint check to SIGKILL instead of raising.
        original_check = rt.commit_channel._check_failpoint
        def kill_check(name):
            if name == failpoint:
                os.kill(os.getpid(), signal.SIGKILL)
            # Don't raise — let the commit continue if not our failpoint.
        rt.commit_channel._check_failpoint = kill_check
        try:
            result = rt.commit_channel.commit(full_txn, auth)
            with open(state_file, 'w') as f:
                json.dump({{'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                           'committed': result.committed}}, f)
        except Exception as e:
            with open(state_file, 'w') as f:
                json.dump({{'pre_hash': pre_hash, 'post_hash': rt.authority_hash,
                           'committed': False, 'error': str(e)}}, f)
""")


def _run_crash_test(tmp_path: Path, failpoint: str) -> dict:
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    wal_path = str(tmp_path / "wal.jsonl")
    state_file = str(tmp_path / "state.json")

    script = _CRASH_SCRIPT.format(
        src_path=src_path, failpoint=failpoint,
        wal_path=wal_path, state_file=state_file,
    )
    script_file = tmp_path / "crash_script.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    script_file.write_text(script)

    proc = subprocess.run(
        [sys.executable, str(script_file)],
        capture_output=True, text=True, timeout=30,
    )

    state = {}
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)

    return {
        "returncode": proc.returncode,
        "state": state,
        "wal_path": wal_path,
    }


def _recover_and_get_hash(wal_path: str) -> str:
    torch.manual_seed(42)
    fresh = LGAERuntime(_graph(), _cfg())
    if os.path.exists(wal_path):
        replay_committed_transactions(wal_path, fresh._engine)
    return fresh.authority_hash


class TestInternalFailpointCrashMatrix:
    """SIGKILL at internal commit failpoints — S_restart ∈ {S_t, S_{t+1}}."""

    @pytest.mark.parametrize("failpoint", [
        "before_prepare",
        "after_wal_commit",
        "before_state_swap",
        "after_state_swap",
    ])
    def test_crash_at_failpoint_recovers_to_pre_or_post(self, tmp_path, failpoint):
        """Crash at each internal failpoint — recovery must give pre or post state."""
        result = _run_crash_test(tmp_path, failpoint)
        state = result["state"]

        # Get pre and post hashes from a normal run.
        normal = _run_crash_test(tmp_path / "normal", "none")
        pre_hash = normal["state"]["pre_hash"]
        post_hash = normal["state"]["post_hash"]

        # Recover from WAL.
        recovered = _recover_and_get_hash(result["wal_path"])

        # The recovered state must be either pre or post — never mixed.
        assert recovered in (pre_hash, post_hash), (
            f"Crash at '{failpoint}': recovered hash {recovered[:16]} "
            f"is neither pre ({pre_hash[:16]}) nor post ({post_hash[:16]})! "
            f"PARTIAL STATE DETECTED!"
        )

    def test_crash_before_prepare_recovers_to_pre(self, tmp_path):
        """Crash before any WAL activity → pre-state."""
        result = _run_crash_test(tmp_path, "before_prepare")
        assert result["returncode"] == -9  # SIGKILL
        normal = _run_crash_test(tmp_path / "normal", "none")
        pre_hash = normal["state"]["pre_hash"]
        recovered = _recover_and_get_hash(result["wal_path"])
        assert recovered == pre_hash

    def test_crash_after_wal_commit_recovers_to_post(self, tmp_path):
        """Crash after WAL COMMIT but before state swap → post-state (replay)."""
        result = _run_crash_test(tmp_path, "after_wal_commit")
        assert result["returncode"] == -9
        normal = _run_crash_test(tmp_path / "normal", "none")
        post_hash = normal["state"]["post_hash"]
        recovered = _recover_and_get_hash(result["wal_path"])
        # After WAL COMMIT, replay should reconstruct the post-state.
        assert recovered == post_hash, (
            f"Crash after WAL COMMIT should recover to post-state. "
            f"Expected: {post_hash[:16]}, Got: {recovered[:16]}"
        )

    def test_crash_before_state_swap_recovers_to_post(self, tmp_path):
        """Crash before state swap (after WAL COMMIT) → post-state."""
        result = _run_crash_test(tmp_path, "before_state_swap")
        assert result["returncode"] == -9
        normal = _run_crash_test(tmp_path / "normal", "none")
        post_hash = normal["state"]["post_hash"]
        recovered = _recover_and_get_hash(result["wal_path"])
        assert recovered == post_hash

    def test_crash_after_state_swap_recovers_to_post(self, tmp_path):
        """Crash after state swap → post-state."""
        result = _run_crash_test(tmp_path, "after_state_swap")
        assert result["returncode"] == -9
        normal = _run_crash_test(tmp_path / "normal", "none")
        post_hash = normal["state"]["post_hash"]
        recovered = _recover_and_get_hash(result["wal_path"])
        assert recovered == post_hash

    def test_no_crash_normal_completion(self, tmp_path):
        """Normal completion (no failpoint) → post-state."""
        result = _run_crash_test(tmp_path, "none")
        assert result["returncode"] == 0
        state = result["state"]
        assert state["committed"]
        post_hash = state["post_hash"]
        recovered = _recover_and_get_hash(result["wal_path"])
        assert recovered == post_hash
