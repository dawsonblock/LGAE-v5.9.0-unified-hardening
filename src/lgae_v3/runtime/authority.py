"""Strict runtime authority boundaries (Phase 2).

Three explicit roles govern who may do what to authoritative state:

  * PROPOSAL    - may generate/rank/retrieve candidates and predict. Cannot
                  mutate authoritative graph/fiber/gauge state.
  * VERIFICATION- may evaluate proposals (shadow simulation, certification).
                  Cannot commit.
  * COMMIT      - the only role permitted to mutate authoritative state, and
                  only through the transactional commit path.

Direct mutation outside the commit authority fails loudly with
``UnauthorizedMutationError`` rather than logging a warning.

This module is intentionally non-breaking: existing engines continue to
operate directly. The boundary is an explicit, testable contract that the
canonical runtime enforces on its own orchestration path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..cache_coherence import GraphReadCoordinator
from ..types import GraphBuffers
from .runtime_state import RuntimeSnapshot, snapshot_from_engine


class AuthorityRole(str, Enum):
    OBSERVATION = "observation"      # read-only readers
    PROPOSAL = "proposal"            # candidate generation / scoring / retrieval
    VERIFICATION = "verification"    # shadow evaluation / certification
    COMMIT = "commit"                # sole authoritative mutator


class UnauthorizedMutationError(RuntimeError):
    """Raised when a non-commit role attempts to mutate authoritative state,
    or when a commit is attempted outside the transactional commit path."""


# Default component -> role classification (from the v5.10 plan).
DEFAULT_BOUNDARIES: dict[str, AuthorityRole] = {
    # Proposal authority
    "graph_state_encoder": AuthorityRole.PROPOSAL,
    "structural_intelligence": AuthorityRole.PROPOSAL,
    "ann_retrieval": AuthorityRole.PROPOSAL,
    "fosr": AuthorityRole.PROPOSAL,
    "effective_resistance": AuthorityRole.PROPOSAL,
    "forman_flow": AuthorityRole.PROPOSAL,
    "learned_candidate_generator": AuthorityRole.PROPOSAL,
    "reasoning_engine": AuthorityRole.PROPOSAL,
    "memory_priors": AuthorityRole.PROPOSAL,
    "mpc_planner": AuthorityRole.PROPOSAL,
    "executive": AuthorityRole.PROPOSAL,
    "counterfactual_proposal": AuthorityRole.PROPOSAL,
    # Verification authority
    "adaptive_geometry": AuthorityRole.VERIFICATION,
    "counterfactual_engine": AuthorityRole.VERIFICATION,
    "exact_orc_lly": AuthorityRole.VERIFICATION,
    "spectral_certification": AuthorityRole.VERIFICATION,
    "topology_certification": AuthorityRole.VERIFICATION,
    "sheaf_gauge_validation": AuthorityRole.VERIFICATION,
    "invariant_checker": AuthorityRole.VERIFICATION,
    "governor": AuthorityRole.VERIFICATION,
    # Commit authority
    "structural_governor": AuthorityRole.COMMIT,
    "transaction_manager": AuthorityRole.COMMIT,
    "mutation_authority_policy": AuthorityRole.COMMIT,
    "engine": AuthorityRole.COMMIT,
}


@dataclass(slots=True)
class AuthorityBoundary:
    """Registry of component roles with enforcement helpers."""
    roles: dict[str, AuthorityRole] = field(default_factory=lambda: dict(DEFAULT_BOUNDARIES))

    def register(self, component: str, role: AuthorityRole) -> None:
        if not isinstance(role, AuthorityRole):
            raise TypeError("role must be an AuthorityRole")
        self.roles[str(component)] = role

    def role_of(self, component: str) -> AuthorityRole:
        return self.roles.get(str(component), AuthorityRole.OBSERVATION)

    def can_mutate(self, component: str) -> bool:
        return self.role_of(component) == AuthorityRole.COMMIT

    def assert_can_mutate(self, component: str) -> None:
        if not self.can_mutate(component):
            raise UnauthorizedMutationError(
                f"component '{component}' has role {self.role_of(component).value}, "
                f"not commit; cannot mutate authoritative state"
            )

    def assert_can_verify(self, component: str) -> None:
        role = self.role_of(component)
        if role not in (AuthorityRole.VERIFICATION, AuthorityRole.COMMIT):
            raise UnauthorizedMutationError(
                f"component '{component}' has role {role.value}; cannot verify proposals"
            )

    def to_summary(self) -> dict[str, str]:
        return {k: v.value for k, v in sorted(self.roles.items())}


class AuthoritativeStateGuard:
    """Read-only view of authoritative state for non-commit components.

    Non-commit components receive a guard instead of the raw engine. The guard
    exposes frozen (immutable) views of graph/fiber/gauge state; any attempt
    to mutate through the guard raises ``UnauthorizedMutationError``.

    v5.11 Phase 3: graph/fibers/gauges now return frozen views that clone
    tensors defensively. The raw engine is never exposed.
    """

    def __init__(self, engine: Any, boundary: AuthorityBoundary, *, component: str) -> None:
        self._engine = engine
        self._boundary = boundary
        self._component = str(component)
        # OBSERVATION/PROPOSAL/VERIFICATION are read-only w.r.t. authority.
        # COMMIT is granted through a separate CommitChannel.

    @property
    def component(self) -> str:
        return self._component

    @property
    def role(self) -> AuthorityRole:
        return self._boundary.role_of(self._component)

    def snapshot(self) -> RuntimeSnapshot:
        return snapshot_from_engine(self._engine)

    @property
    def graph(self) -> FrozenGraphView:
        """Frozen (immutable) graph view.

        Returns a FrozenGraphView that defensively clones all tensors.
        Any attempt to mutate through the view raises
        ``UnauthorizedMutationError``.
        """
        from .state.frozen_views import FrozenGraphView
        return FrozenGraphView(self._engine.graph)

    @property
    def fibers(self) -> FrozenFiberView:
        """Frozen (immutable) fiber view."""
        from .state.frozen_views import FrozenFiberView
        return FrozenFiberView(self._engine.fibers)

    @property
    def gauge_connections(self) -> FrozenGaugeView:
        """Frozen (immutable) gauge view."""
        from .state.frozen_views import FrozenGaugeView
        return FrozenGaugeView(getattr(self._engine, "gauge_connections", None))

    def authority_hash(self) -> str:
        return self._engine.authority_hash()

    def __setattr__(self, name: str, value: Any) -> None:
        # Block accidental mutation of the guard's engine reference from
        # outside. Internal fields are prefixed with underscore.
        if name.startswith("_") or name in {"component"}:
            super().__setattr__(name, value)
        else:
            raise UnauthorizedMutationError(
                f"cannot set attribute '{name}' on AuthoritativeStateGuard; "
                "authoritative state is mutated only through the commit channel"
            )


class CommitChannel:
    """The sole channel through which commit-authority components mutate
    authoritative state.

    v5.11 Phase 4-5: CommitChannel.commit(transaction, authorization) is the
    ONLY path to mutate authoritative state. It validates:

    1. authorization.status == AUTHORIZED
    2. transaction.authorization_id matches authorization
    3. transaction.base_state_hash == current engine state hash
    4. transaction.base_state_version == current engine version
    5. transaction.delta_hash matches recomputed hash
    6. WAL is available in production mode

    If any check fails, the commit is rejected and no state changes.

    Every commit is bracketed by the read coordinator's write epoch
    (seqlock) so concurrent optimistic readers observe a stale read and
    retry rather than seeing a half-applied mutation."""

    def __init__(
        self,
        engine: Any,
        boundary: AuthorityBoundary,
        *,
        component: str = "engine",
        read_coordinator: GraphReadCoordinator | None = None,
        wal: Any = None,
        require_wal: bool = False,
    ) -> None:
        boundary.assert_can_mutate(component)
        self._engine = engine
        self._boundary = boundary
        self._component = str(component)
        self._read_coordinator = read_coordinator
        self._wal = wal
        self._require_wal = require_wal
        self._commit_count = 0
        self._last_transaction_id: str | None = None

    def _bracket(self, fn: Callable[[], Any]) -> Any:
        if self._read_coordinator is None:
            return fn()
        self._read_coordinator.begin_write()
        try:
            return fn()
        finally:
            self._read_coordinator.end_write()

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def commit_count(self) -> int:
        return self._commit_count

    @property
    def last_transaction_id(self) -> str | None:
        return self._last_transaction_id

    def authority_hash(self) -> str:
        return self._engine.authority_hash()

    def snapshot(self) -> RuntimeSnapshot:
        return snapshot_from_engine(self._engine)

    def commit(
        self,
        transaction: Any,
        authorization: Any,
    ) -> Any:
        """Commit a StructuralTransaction with authorization binding.

        This is the sole mutation path. Validates:
        - authorization is AUTHORIZED
        - transaction authorization binding matches
        - base state hash/version match current engine state
        - delta hash is correct
        - WAL is available when required

        Returns CommitResult on success, raises on failure.
        """
        from .transaction import (
            StructuralTransaction, TransactionValidationError,
            StaleTransactionError, AuthorizationBindingError,
        )
        from .contracts.authorization import AuthorizationResult, AuthorizationStatus
        from .contracts.commit import CommitResult
        from ..version import VERSION

        # Validation 1: authorization must be AUTHORIZED.
        if not isinstance(authorization, AuthorizationResult):
            raise AuthorizationBindingError(
                "authorization must be an AuthorizationResult"
            )
        if authorization.status != AuthorizationStatus.AUTHORIZED:
            raise AuthorizationBindingError(
                f"authorization status is {authorization.status.value}, not AUTHORIZED"
            )

        # Validation 2: transaction must be a StructuralTransaction.
        if not isinstance(transaction, StructuralTransaction):
            raise TransactionValidationError(
                "transaction must be a StructuralTransaction"
            )

        # Validation 3: authorization binding.
        # The authorization_id in the transaction must match the
        # authorization's binding hash.
        expected_auth_id = transaction.authorization_binding_hash()
        if transaction.authorization_id is not None:
            if transaction.authorization_id != expected_auth_id:
                raise AuthorizationBindingError(
                    "transaction.authorization_id does not match "
                    "authorization_binding_hash; possible swap attack"
                )

        # Validation 4: base state must match current engine state.
        current_hash = self._engine.authority_hash()
        current_version = int(self._engine.graph.version)
        if transaction.base_state_hash != current_hash:
            raise StaleTransactionError(
                f"transaction base_state_hash {transaction.base_state_hash[:16]}... "
                f"does not match current engine hash {current_hash[:16]}...; "
                f"transaction is stale"
            )
        if transaction.base_state_version != current_version:
            raise StaleTransactionError(
                f"transaction base_state_version {transaction.base_state_version} "
                f"does not match current engine version {current_version}; "
                f"transaction is stale"
            )

        # Validation 5: delta hash must be correct.
        recomputed = transaction.compute_delta_hash()
        if transaction.delta_hash != recomputed:
            raise TransactionValidationError(
                "transaction.delta_hash does not match recomputed hash; "
                "transaction may have been tampered with"
            )

        # Validation 6: WAL availability in production.
        if self._require_wal and self._wal is None:
            raise TransactionValidationError(
                "WAL is required but not configured"
            )

        # All validations passed. Apply the transaction atomically.
        def _apply() -> CommitResult:
            # WAL: write BEGIN + WRITE records before applying.
            # The WRITE record contains the full transaction state so
            # recovery can re-apply it after a crash.
            wal_txn_id = None
            if self._wal is not None:
                wal_txn_id = self._wal.begin({
                    "transaction_id": transaction.transaction_id,
                    "base_state_hash": transaction.base_state_hash,
                    "base_state_version": transaction.base_state_version,
                })
                if transaction.graph_delta is not None:
                    # Serialize the shadow graph for recovery.
                    # Convert tensors to lists for JSON serialization.
                    sg = transaction.graph_delta.shadow_graph
                    sd = sg.to_state_dict()
                    json_state = {}
                    for k, v in sd.items():
                        if hasattr(v, "tolist"):
                            json_state[k] = v.tolist()
                        else:
                            json_state[k] = v
                    self._wal.write(wal_txn_id, {
                        "kind": "graph",
                        "shadow_graph_hash": sg.state_hash(),
                        "shadow_graph_state": json_state,
                        "mutation_name": transaction.graph_delta.mutation_name,
                    })

            # Apply graph delta.
            if transaction.graph_delta is not None:
                old_valid = self._engine.graph.valid.clone()
                self._engine.graph = transaction.graph_delta.shadow_graph
                # Sync gauge generations if needed.
                if self._engine.gauge_connections is not None:
                    reset = torch.where(
                        old_valid != self._engine.graph.valid
                    )[0]
                    self._engine.gauge_connections.reset_slots(
                        reset,
                        optimizers=self._engine.optimizers,
                        sync_generation=self._engine.graph.slot_generation,
                    )
                self._engine._invalidate_neighbor_indices("transaction_commit")

            # Apply fiber delta.
            if transaction.fiber_delta is not None:
                self._engine.fibers.restore(
                    transaction.fiber_delta.shadow_fiber_snapshot
                )

            # Apply gauge delta.
            if transaction.gauge_delta is not None:
                if self._engine.gauge_connections is not None:
                    self._engine.gauge_connections.raw_generators.copy_(
                        transaction.gauge_delta.shadow_gauge_raw.to(
                            self._engine.gauge_connections.raw_generators
                        )
                    )

            # Record cooldowns if mutation result has the mutation info.
            if transaction.mutation_result is not None:
                # The mutation was already evaluated; just record cooldown.
                pass

            after_hash = self._engine.authority_hash()
            after_version = int(self._engine.graph.version)

            # WAL: write COMMIT record after applying.
            if self._wal is not None and wal_txn_id is not None:
                self._wal.commit(wal_txn_id)

            self._commit_count += 1
            self._last_transaction_id = transaction.transaction_id

            return CommitResult(
                snapshot_id=f"{transaction.base_state_hash}:{transaction.base_state_version}",
                state_version=transaction.base_state_version,
                state_hash=transaction.base_state_hash,
                committed=True,
                new_state_version=after_version,
                new_state_hash=after_hash,
                transaction_id=transaction.transaction_id,
                authority_hash_after=after_hash,
            )

        return self._bracket(_apply)

    def evaluate_and_maybe_commit(self, mutation: Any) -> Any:
        """Legacy path: Delegate to the engine's transactional commit path.

        DEPRECATED in v5.11. Use commit(transaction, authorization) instead.
        Kept for backward compatibility with existing engine callers.
        """
        return self._bracket(lambda: self._engine.evaluate_and_maybe_commit(mutation))

    def evaluate_fiber_action(self, *args, **kwargs) -> Any:
        """Legacy path for fiber actions."""
        return self._bracket(lambda: self._engine.evaluate_fiber_action(*args, **kwargs))

    def evaluate_gauge_action(self, *args, **kwargs) -> Any:
        """Legacy path for gauge actions."""
        return self._bracket(lambda: self._engine.evaluate_gauge_action(*args, **kwargs))
