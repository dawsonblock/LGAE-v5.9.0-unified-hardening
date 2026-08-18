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
    authoritative state. It wraps the engine's transactional commit path.

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
    ) -> None:
        boundary.assert_can_mutate(component)
        self._engine = engine
        self._boundary = boundary
        self._component = str(component)
        self._read_coordinator = read_coordinator

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

    def evaluate_and_maybe_commit(self, mutation: Any) -> Any:
        """Delegate to the engine's transactional commit path, bracketed by
        the read-coordinator write epoch."""
        return self._bracket(lambda: self._engine.evaluate_and_maybe_commit(mutation))

    def evaluate_fiber_action(self, *args, **kwargs) -> Any:
        return self._bracket(lambda: self._engine.evaluate_fiber_action(*args, **kwargs))

    def evaluate_gauge_action(self, *args, **kwargs) -> Any:
        return self._bracket(lambda: self._engine.evaluate_gauge_action(*args, **kwargs))

    def authority_hash(self) -> str:
        return self._engine.authority_hash()

    def snapshot(self) -> RuntimeSnapshot:
        return snapshot_from_engine(self._engine)
