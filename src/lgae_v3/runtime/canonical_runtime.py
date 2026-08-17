"""Canonical v5.10 runtime: one authoritative end-to-end governed cycle.

``LGAERuntime`` orchestrates existing engines. It does NOT re-implement any
algorithm. The authority model is strict:

  * Proposal authority  -> learned executive / counterfactual / memory / MPC
  * Verification authority -> governor (shadow evaluation / certification)
  * Commit authority -> ``LGAEEngine`` (the only component that mutates
    authoritative graph/fiber/gauge state, via transactional evaluation)

The runtime's ``step()`` implements the complete governed cycle from the
v5.10 plan and emits immutable evidence + a signed hash-chained receipt on
every authoritative commit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from ..config import LGAEConfig, ResearchConfig, config_governance_hash
from ..cache_coherence import (
    GraphReadCoordinator, run_consistent_read, StaleReadError as _CCStaleReadError,
    CommitEventBus, GraphCommitEvent, ChangeKind,
)
from ..evidence import EvidenceLedger, EvidenceRecord
from ..evolution import LGAEEngine
from ..executive import StructuralExecutive, StructuralAction, ACTION_TO_IDX
from ..receipts import mutation_receipt, append_receipt, ed25519_available, generate_keypair
from ..structural_loop import StructuralLearningLoop, StructuralLoopResult
from ..types import GraphBuffers, MutationDecision, MutationResult
from ..version import VERSION
from .runtime_config import RuntimeConfig, RuntimeMode
from .runtime_events import RuntimeEvent, RuntimePhase
from .runtime_result import RuntimeStepResult
from .runtime_state import RuntimeSnapshot, snapshot_from_engine, StaleReadError
from .authority import (
    AuthorityBoundary, AuthorityRole, AuthoritativeStateGuard,
    CommitChannel, UnauthorizedMutationError,
)
from .cache_coherence import MutationImpact, CacheRegistry


class LGAERuntime:
    """One canonical governed structural-intelligence runtime.

    Parameters
    ----------
    graph:
        Initial authoritative graph buffers.
    config:
        Subsystem ``LGAEConfig`` (or preset). Defaults to ``ResearchConfig``.
    runtime_config:
        Orchestration-level config (mode, evidence/receipt paths, MPC).
    engine:
        Optional pre-built engine. When omitted the runtime constructs one
        bound to ``graph`` and ``config``. The engine is the sole commit
        authority.
    """

    def __init__(
        self,
        graph: GraphBuffers,
        config: LGAEConfig | None = None,
        *,
        runtime_config: RuntimeConfig | None = None,
        engine: LGAEEngine | None = None,
        executive: StructuralExecutive | None = None,
        utility_fn: Callable[[GraphBuffers, Tensor], float] | None = None,
    ) -> None:
        self.config = config or ResearchConfig()
        # Production mode must use the conservative preset. Never silently
        # fall into research behavior. We detect a production-grade config by
        # the safety machinery ProductionConfig() enables; a plain/research
        # config in production mode is a caller error.
        self.runtime_config = runtime_config or RuntimeConfig()
        if self.runtime_config.is_production:
            cfg = self.config
            production_grade = (
                bool(getattr(cfg.mutation, "curvature_ema_enabled", False))
                and bool(getattr(cfg.mutation, "equilibrium_barrier_enabled", False))
                and bool(getattr(cfg.audit, "require_persistent_homology", False))
            )
            if not production_grade:
                raise ValueError(
                    "production runtime mode requires a ProductionConfig-grade LGAEConfig "
                    "(curvature_ema_enabled, equilibrium_barrier_enabled, "
                    "require_persistent_homology must be enabled)"
                )

        self.engine = engine if engine is not None else LGAEEngine(graph, self.config)
        self.executive = executive or StructuralExecutive(self.config)

        util = utility_fn or self.runtime_config.utility_fn or _default_utility
        self.utility_fn = util

        # The governed structural-learning loop. It owns the counterfactual,
        # uncertainty, credit, consolidation, and timescale machinery and
        # delegates commit to the engine (the only commit authority).
        self.loop = StructuralLearningLoop(
            config=self.config,
            executive=self.executive,
            engine=self.engine,
            ensemble_size=self.runtime_config.ensemble_size,
            max_candidates=self.runtime_config.max_candidates,
        )

        # Evidence + receipt ledgers. In-memory when no path is configured.
        self.evidence_ledger = (
            EvidenceLedger(self.runtime_config.evidence_path)
            if self.runtime_config.evidence_path is not None
            else _InMemoryEvidenceLedger()
        )
        self._receipt_path = self.runtime_config.receipt_path
        self._signing_key = self.runtime_config.signing_key
        self._receipt_count = 0

        # Optional MPC planner (Phase 14). Lazily constructed so we do not
        # import heavy planning paths when horizon == 1.
        self._mpc: Any | None = None
        if self.runtime_config.mpc_horizon > 1:
            from ..mpc import StructuralMPC
            self._mpc = StructuralMPC(
                util,
                horizon=int(self.runtime_config.mpc_horizon),
                max_branching=int(self.runtime_config.mpc_max_branching),
                max_sequences=int(self.runtime_config.mpc_max_sequences),
            )

        self._step = 0
        self._events: list[RuntimeEvent] = []
        # Generation is the authoritative step counter bound to snapshots.
        self._generation = int(self.engine.step_index)

        # Strict authority boundaries (Phase 2). The engine is the sole commit
        # authority; proposal/verification components receive read-only guards.
        self.boundary = AuthorityBoundary()
        self.boundary.register("engine", AuthorityRole.COMMIT)
        self.boundary.register("executive", AuthorityRole.PROPOSAL)
        self.boundary.register("counterfactual_engine", AuthorityRole.VERIFICATION)
        self.boundary.register("governor", AuthorityRole.VERIFICATION)
        if self._mpc is not None:
            self.boundary.register("mpc_planner", AuthorityRole.PROPOSAL)
        # Seqlock-style read coordinator (Phase 3): commits are bracketed by
        # a write epoch so optimistic readers retry on stale reads.
        self.read_coordinator = GraphReadCoordinator()
        self._commit_channel = CommitChannel(
            self.engine, self.boundary, component="engine",
            read_coordinator=self.read_coordinator,
        )
        # Mandatory cache coherence (Phase 4): a commit event bus drives
        # selective invalidation of declared-cache dependencies.
        self.commit_event_bus = CommitEventBus()
        self.cache_registry = CacheRegistry(self.commit_event_bus)

    # ------------------------------------------------------------------ #
    # Authority boundary helpers (Phase 2 foundation)
    # ------------------------------------------------------------------ #
    def _assert_commit_authority(self) -> None:
        """Only the engine may mutate authoritative state. The runtime is an
        orchestrator, not a mutator."""
        if self.engine is None:
            raise UnauthorizedMutationError("no commit authority (engine) is bound")
        self.boundary.assert_can_mutate("engine")

    def guard_for(self, component: str) -> AuthoritativeStateGuard:
        """Return a read-only authoritative-state guard for a non-commit
        component. Commit-authority components must use the commit channel."""
        if self.boundary.role_of(component) == AuthorityRole.COMMIT:
            raise UnauthorizedMutationError(
                f"component '{component}' is commit-authority; use the commit channel, not a guard"
            )
        return AuthoritativeStateGuard(self.engine, self.boundary, component=component)

    @property
    def commit_channel(self) -> CommitChannel:
        return self._commit_channel

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def authority_hash(self) -> str:
        return self.engine.authority_hash()

    def snapshot(self) -> RuntimeSnapshot:
        """Capture an immutable authoritative snapshot for readers."""
        return snapshot_from_engine(self.engine, generation=self._generation)

    def consistent_read(self, compute_fn: Callable[[], Any]) -> Any:
        """Run a derived calculation and publish only a generation-consistent
        result. Retries on stale reads up to ``max_stale_read_retries``.

        This is the canonical reader path: every expensive reader should
        operate through ``consistent_read`` so no subsystem silently fetches
        mutable state halfway through a calculation. A read that overlaps a
        commit raises ``StaleReadError`` and is retried from a new snapshot.
        """
        return run_consistent_read(
            self.read_coordinator,
            generation_getter=lambda: int(self.engine.graph.version),
            compute_fn=compute_fn,
            max_retries=int(self.runtime_config.max_stale_read_retries),
        )

    # ------------------------------------------------------------------ #
    # Canonical cycle phases. Each phase delegates to an existing engine.
    # ------------------------------------------------------------------ #
    def observe(self, *, task_loss: float = 0.0, task_loss_delta: float = 0.0,
                epistemic_uncertainty: float = 0.0) -> RuntimeSnapshot:
        """Phase: Observation / Graph State -> Stable Snapshot."""
        snap = self.snapshot()
        self._emit(RuntimePhase.OBSERVE, {"graph_version": snap.graph_version,
                                          "authority_hash": snap.authority_hash,
                                          "task_loss": float(task_loss)})
        self._emit(RuntimePhase.SNAPSHOT, snap.to_summary())
        return snap

    def reason(self, snap: RuntimeSnapshot) -> dict[str, Any]:
        """Phase: Reasoning Graph + Memory (delegated to executive observe)."""
        graph = self.engine.graph
        z = self.engine.fibers().detach().clone()
        audit = self.engine.audit()
        observation = self.executive.observe(
            graph, z, audit, task_loss=0.0, task_loss_delta=0.0,
            epistemic_uncertainty=0.0, fiber_state=self.engine.fibers,
        )
        self._emit(RuntimePhase.REASON, {"observation": observation.to_vector().tolist()})
        return {"observation": observation, "graph": graph, "z": z}

    def propose(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Phase: Candidate Generation + Ranking + Uncertainty + IG/Risk.

        Delegated to the counterfactual engine inside the loop.
        """
        observation = ctx["observation"]
        counterfactual = self.loop.counterfactual.evaluate(observation, None)
        chosen = counterfactual.winner if counterfactual.beats_no_op else StructuralAction.NO_OP
        self._emit(RuntimePhase.PROPOSE, {
            "candidates": len(counterfactual.proposals),
            "beats_no_op": bool(counterfactual.beats_no_op),
            "winner": chosen.value,
        })
        ctx["counterfactual"] = counterfactual
        ctx["chosen_action"] = chosen
        return ctx

    def plan(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Phase: Multi-Step Counterfactual Planning (receding horizon).

        When MPC is enabled, plan a horizon and keep only the first action.
        Otherwise this is a pass-through to the single-action proposal.
        """
        if self._mpc is None:
            self._emit(RuntimePhase.PLAN, {"horizon": 1, "planner": "single_step"})
            return ctx
        graph = self.engine.graph
        z = self.engine.fibers().detach().clone()
        plan_result = self._mpc.plan(graph, z, seed=int(self.config.seed) + self._step)
        self._emit(RuntimePhase.PLAN, {
            "horizon": int(plan_result.horizon),
            "candidates_evaluated": int(plan_result.candidates_evaluated),
            "predicted_utility": float(plan_result.predicted_utility),
            "first_authority": plan_result.first_mutation_authority.value,
        })
        ctx["mpc_plan"] = plan_result
        return ctx

    def evaluate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Phase: Shadow Transaction + Exact/Escalating Verification.

        Delegated to the engine's transactional evaluation, which builds a
        shadow graph and runs the governor's certification horizons. The
        engine is the verification-and-commit authority; this phase only
        triggers it and records the certification level.
        """
        chosen_action: StructuralAction = ctx["chosen_action"]
        if chosen_action == StructuralAction.NO_OP:
            self._emit(RuntimePhase.EVALUATE, {"decision": "no_op", "certification": None})
            ctx["mutation_result"] = None
            ctx["certification_level"] = None
            return ctx
        target = self.executive.select_target(
            chosen_action, self.engine.graph, self.engine.fibers().detach(),
            fiber_state=self.engine.fibers,
        )
        ctx["target"] = target
        result = self.loop._execute_engine_action(chosen_action, target)
        cert = None
        if isinstance(result, MutationResult) and result.metadata:
            cert = result.metadata.get("certification_level")
        self._emit(RuntimePhase.EVALUATE, {
            "decision": result.decision.value,
            "certification": cert,
            "reasons": list(result.reasons),
        })
        ctx["mutation_result"] = result
        ctx["certification_level"] = cert
        return ctx

    def authorize(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Phase: Authority Governor decision (reject/quarantine/commit)."""
        result: MutationResult | None = ctx.get("mutation_result")
        if result is None:
            decision = MutationDecision.ACCEPT.value  # NO_OP is an accepted no-commit
        else:
            decision = result.decision.value
        self._emit(RuntimePhase.AUTHORIZE, {"decision": decision})
        ctx["governance_decision"] = decision
        return ctx

    def commit(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Phase: Atomic State Update + Cache Invalidation + Evidence/Receipt.

        The engine has already performed the atomic commit inside
        ``evaluate`` (it is the commit authority). This phase records the
        immutable evidence and signed receipt for the committed mutation.
        """
        result: MutationResult | None = ctx.get("mutation_result")
        executed = result is not None and result.decision == MutationDecision.ACCEPT
        if executed:
            self._assert_commit_authority()
            before_hash = ctx.get("authority_before") or self.snapshot().authority_hash
            after_hash = self.engine.authority_hash()
            # Immutable evidence record.
            evidence = self.evidence_ledger.append(EvidenceRecord(
                record_type="runtime_mutation_commit",
                graph_hash=after_hash,
                payload={
                    "action": ctx["chosen_action"].value,
                    "target": ctx.get("target", {}),
                    "decision": result.decision.value,
                    "reasons": list(result.reasons),
                    "certification_level": ctx.get("certification_level"),
                    "authority_hash_before": before_hash,
                    "authority_hash_after": after_hash,
                    "step": int(self._step),
                },
                authority_hash=after_hash,
            ))
            ctx["evidence_hash"] = evidence.get("sha256")
            self._emit(RuntimePhase.EVIDENCE, {"evidence_hash": ctx["evidence_hash"]})
            # Signed hash-chained receipt.
            receipt = mutation_receipt(
                result,
                authority_state_hash_before=before_hash,
                authority_state_hash_after=after_hash,
                gauge_authority_hash=(
                    None if self.engine.gauge_connections is None
                    else self.engine.gauge_connections.state_hash()
                ),
                signing_key=self._signing_key,
            )
            if self._receipt_path is not None:
                append_receipt(self._receipt_path, receipt, signing_key=self._signing_key)
            ctx["receipt_hash"] = receipt.get("sha256")
            self._receipt_count += 1
            self._emit(RuntimePhase.COMMIT, {
                "authority_hash_after": after_hash,
                "receipt_hash": ctx["receipt_hash"],
            })
            self._emit(RuntimePhase.CACHE_INVALIDATE, {
                "graph_version": int(self.engine.graph.version),
            })
        else:
            ctx["evidence_hash"] = None
            ctx["receipt_hash"] = None
        return ctx

    def learn(self, ctx: dict[str, Any], loop_result: StructuralLoopResult) -> dict[str, Any]:
        """Phase: Replay / Experience -> Learn.

        Delegated to the structural-learning loop, which already records
        outcomes into the executive, uncertainty ensemble, calibrator, and
        credit tracker. The runtime only surfaces the learning event.
        """
        self._emit(RuntimePhase.LEARN, {
            "executive_experience": len(self.executive._experience),
            "credit_summary": self.loop.credit_tracker.summary(),
        })
        ctx["loop_result"] = loop_result
        return ctx

    # ------------------------------------------------------------------ #
    # The complete governed cycle.
    # ------------------------------------------------------------------ #
    def step(self, *, task_loss: float = 0.0, task_loss_delta: float = 0.0,
             epistemic_uncertainty: float = 0.0) -> RuntimeStepResult:
        """Run one complete governed cycle end-to-end.

        Order:
            observe -> reason -> propose -> plan -> evaluate -> authorize
            -> commit -> learn
        """
        # Bind the authoritative snapshot at the start of the cycle. Readers
        # operate from this immutable snapshot.
        snap_before = self.observe(
            task_loss=task_loss, task_loss_delta=task_loss_delta,
            epistemic_uncertainty=epistemic_uncertainty,
        )
        authority_before = snap_before.authority_hash

        # The structural-learning loop performs the integrated governed step
        # (counterfactual -> uncertainty -> governor -> commit -> credit ->
        # learn). The runtime decomposes the same cycle into canonical phases
        # for observability and evidence, while delegating the actual work to
        # the loop and engine.
        loop_result = self.loop.step(
            self.engine.graph,
            self.engine.fibers().detach().clone(),
            task_loss=task_loss,
            task_loss_delta=task_loss_delta,
            epistemic_uncertainty=epistemic_uncertainty,
            utility_fn=self.utility_fn,
        )

        # Reconstruct canonical-phase context from the loop result so the
        # runtime's phase decomposition stays consistent with the work done.
        ctx: dict[str, Any] = {
            "chosen_action": loop_result.chosen_action,
            "mutation_result": None,
            "authority_before": authority_before,
            "target": loop_result.metadata.get("target", {}),
        }
        # Pull the governor decision and certification from the loop result.
        ctx["governance_decision"] = loop_result.governance_decision
        ctx["certification_level"] = loop_result.metadata.get("certification_level")
        # If the loop committed, record evidence + receipt.
        if loop_result.executed:
            self._assert_commit_authority()
            after_hash = self.engine.authority_hash()
            evidence = self.evidence_ledger.append(EvidenceRecord(
                record_type="runtime_mutation_commit",
                graph_hash=after_hash,
                payload={
                    "action": loop_result.chosen_action.value,
                    "target": loop_result.metadata.get("target", {}),
                    "decision": loop_result.governance_decision,
                    "reasons": loop_result.metadata.get("mutation_reasons", []),
                    "certification_level": ctx.get("certification_level"),
                    "authority_hash_before": authority_before,
                    "authority_hash_after": after_hash,
                    "step": int(self._step),
                    "delta_utility": float(loop_result.delta_utility),
                },
                authority_hash=after_hash,
            ))
            ctx["evidence_hash"] = evidence.get("sha256")
            receipt = mutation_receipt(
                loop_result.metadata,
                authority_state_hash_before=authority_before,
                authority_state_hash_after=after_hash,
                gauge_authority_hash=(
                    None if self.engine.gauge_connections is None
                    else self.engine.gauge_connections.state_hash()
                ),
                signing_key=self._signing_key,
            )
            if self._receipt_path is not None:
                append_receipt(self._receipt_path, receipt, signing_key=self._signing_key)
            ctx["receipt_hash"] = receipt.get("sha256")
            self._receipt_count += 1
            # Publish a MutationImpact on the commit event bus so declared
            # caches are selectively invalidated (Phase 4). Derive the impact
            # from the chosen action's structural dimension.
            impact = _impact_for_action(loop_result.chosen_action)
            self.commit_event_bus.publish(GraphCommitEvent(
                generation=int(self.engine.graph.version),
                changes=impact.to_change_kind(),
                reason="runtime_commit",
            ))
            ctx["mutation_impact"] = impact
            self._emit(RuntimePhase.COMMIT, {
                "authority_hash_after": after_hash,
                "receipt_hash": ctx["receipt_hash"],
                "mutation_impact": impact.to_log(),
            })
            self._emit(RuntimePhase.CACHE_INVALIDATE, {
                "graph_version": int(self.engine.graph.version),
                "invalidated": self.cache_registry.invalidations[-1]["invalidated"] if self.cache_registry.invalidations else [],
                "spared": self.cache_registry.invalidations[-1]["spared"] if self.cache_registry.invalidations else [],
            })
            self._emit(RuntimePhase.EVIDENCE, {"evidence_hash": ctx["evidence_hash"]})
        else:
            ctx["evidence_hash"] = None
            ctx["receipt_hash"] = None

        self.learn(ctx, loop_result)
        snap_after = self.snapshot()
        self._generation = int(self.engine.step_index)
        self._step += 1

        return RuntimeStepResult(
            step=self._step - 1,
            snapshot_before=snap_before,
            snapshot_after=snap_after,
            chosen_action=loop_result.chosen_action.value,
            governance_decision=loop_result.governance_decision,
            executed=bool(loop_result.executed),
            utility_before=float(loop_result.utility_before),
            utility_after=float(loop_result.utility_after),
            delta_utility=float(loop_result.delta_utility),
            certification_level=ctx.get("certification_level"),
            evidence_hash=ctx.get("evidence_hash"),
            receipt_hash=ctx.get("receipt_hash"),
            phases={ev.phase.value: ev.payload for ev in self._events_tail_since(snap_before.generation)},
            metadata={
                "version": VERSION,
                "target": loop_result.metadata.get("target", {}),
                "authority_hash_before": authority_before,
                "authority_hash_after": snap_after.authority_hash,
                "uncertainty": loop_result.metadata.get("uncertainty", {}),
            },
        )

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    def _emit(self, phase: RuntimePhase, payload: dict[str, Any]) -> None:
        self._events.append(RuntimeEvent(phase=phase, step=self._step, payload=payload))

    def _events_tail_since(self, generation: int) -> list[RuntimeEvent]:
        # Events are append-only per step; return all events emitted this step.
        return [e for e in self._events if e.step == self._step]

    def events(self) -> list[dict[str, Any]]:
        return [e.to_log() for e in self._events]

    def summary(self) -> dict[str, Any]:
        return {
            "step": int(self._step),
            "generation": int(self._generation),
            "authority_hash": self.authority_hash,
            "receipt_count": int(self._receipt_count),
            "evidence_root": self.evidence_ledger.root_hash,
            "runtime_config": self.runtime_config.to_summary(),
            "loop_summary": self.loop.summary(),
            "version": VERSION,
        }


class _InMemoryEvidenceLedger(EvidenceLedger):
    """In-memory evidence ledger for research/non-persistent runs."""

    def __init__(self) -> None:
        # Bypass file-based construction.
        self.path = None  # type: ignore[assignment]
        self._records: list[dict[str, Any]] = []
        self._previous: str | None = None
        self._index = -1

    def append(self, record: EvidenceRecord) -> dict[str, Any]:
        from ..evidence import EVIDENCE_SCHEMA, _safe, _canonical
        import hashlib
        self._index += 1
        envelope = {
            "schema": EVIDENCE_SCHEMA,
            "build_version": VERSION,
            "index": self._index,
            "previous_hash": self._previous,
            "record": _safe(record),
        }
        envelope["sha256"] = hashlib.sha256(_canonical(envelope)).hexdigest()
        self._records.append(envelope)
        self._previous = envelope["sha256"]
        return envelope

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def verify(self) -> tuple[bool, list[str]]:
        ok, errors = super().verify() if self.path is not None else (True, [])
        return ok, errors

    @property
    def root_hash(self) -> str | None:
        return self._previous


def _default_utility(graph: GraphBuffers, z: Tensor) -> float:
    """Default structural utility: negative sum of squared latent distances
    over active edges (a smooth connectivity proxy)."""
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        if src.numel() == 0:
            return 0.0
        d = (z[src] - z[dst]).pow(2).sum(-1)
        w = graph.weight[graph.valid]
        return float(-(w * d).sum().item())


def _impact_for_action(action: StructuralAction) -> MutationImpact:
    """Map a structural action to the state dimensions it can change.

    This is a conservative over-approximation used for cache invalidation.
    The authoritative impact is the one observed by the transaction itself;
    this helper gives the runtime a declarative impact for the commit event.
    """
    from ..executive import StructuralAction as A
    if action in (A.ADD_EDGE, A.PRUNE_EDGE):
        return MutationImpact(topology=True, weights=True, metric=True)
    if action in (A.REWEIGHT_AFFINITY,):
        return MutationImpact(weights=True)
    if action in (A.REWEIGHT_LENGTH,):
        return MutationImpact(metric=True)
    if action == A.COUPLED_REWEIGHT:
        return MutationImpact(weights=True, metric=True)
    if action in (A.SPAWN_FIBER, A.PRUNE_FIBER):
        return MutationImpact(fibers=True, latents=True)
    if action == A.CHANGE_GAUGE:
        return MutationImpact(gauges=True)
    return MutationImpact()  # NO_OP
