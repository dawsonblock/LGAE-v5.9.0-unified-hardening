<div align="center">

# LGAE v5.9.0

### Governed Adaptive Geometry Engine

**A multi-timescale geometric controller for self-evolving graph and fiber-bundle latent spaces**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![CI Status](https://github.com/dawsonblock/LGAE/actions/workflows/ci.yml/badge.svg)](https://github.com/dawsonblock/LGAE/actions)
[![Tests](https://img.shields.io/badge/tests-719%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gauge: SO(d)](https://img.shields.io/badge/gauge-SO(d)%20invariance-purple.svg)]()
[![Validation: synthetic-only](https://img.shields.io/badge/validation-synthetic%20only-orange.svg)]()

</div>

---

## v5.9.0 — Unified Hardening Merge

v5.9.0 keeps the v5.8.4 reasoning, cache-coherence, adaptive-geometry, structural-intelligence, and joint topology/gauge runtime as the trunk while restoring the production hardening that existed in v5.3.3. Restored capabilities include domain-separated deterministic RNG streams, reproducibility metadata and qualification IDs, configurable feature-based credit baselines, generator-norm-bounded dynamic gauges, residual-aware equilibrium barriers, Bayesian curvature hysteresis, explicit audit certification levels, mutation authority levels, tensor-native topology/bridge analysis, bounded structural MPC, information-gain benchmarking, Ed25519-signed mutation receipts, and checkpoint Merkle roots.

The information-gain task is retained as a benchmark but remains separate from the legacy structural-diagnosis release gate so the historical qualification threshold is not silently redefined. The default long-horizon credit baseline remains `GraphHashBaseline` for v5.8 compatibility; `GraphFeatureBaseline` is available explicitly where generalization across structurally similar states is desired.

## v5.5 evidence-grounded reasoning memory

v5.5 adds a four-layer control architecture around the existing geometry engine:

1. **Structural graph (`G_S`)** — the authoritative topology, fibers, gauges and geometric state.
2. **Reasoning graph (`G_R`)** — typed dependency nodes that can fan out independent diagnostics in parallel and deterministically reduce their evidence.
3. **Experience-memory graph (`G_M`)** — a derived, reorganizable graph of states, actions, predictions and outcomes used only as a ranking prior.
4. **Immutable evidence ledger (`L_E`)** — append-only, SHA-256 hash-chained JSONL records. Memory can evolve; evidence cannot be rewritten without detection.

The learned reasoner therefore scores `Q(S,a,M)` but remains non-authoritative: every actual topology mutation still passes through the exact governor and transaction path. v5.5 also removes the v5.4 random-exploration `O(N^2)` non-edge materialization path and replaces it with bounded direct sampling.

**Scientific boundary:** this release strengthens architecture, provenance and experience reuse. It still does not claim that the learned reasoner beats strong rewiring baselines on unseen topology families; that remains an empirical training/qualification gate.


## Table of contents

- [Overview](#overview)
- [Validation status](#validation-status)
- [Quickstart](#quickstart)
- [Code examples](#code-examples)
- [CLI](#cli)
- [Architecture](#architecture)
- [Key features](#key-features)
- [Mathematical oracles](#mathematical-oracles)
- [Baseline comparison](#baseline-comparison)
- [Real-world experiment](#real-world-experiment)
- [Validation boundaries](#validation-boundaries)
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Release history](#release-history)
- [License](#license)

---

## Overview

LGAE is a research-grade governed geometric learning engine and structural controller. It operates over graph-structured data and continuous fiber bundles, combining:

- **Continuous field diffusion** — sparse row-stochastic Markov operators with `SO(d)` gauge parallel transport
- **Lie-algebra gauge connections** — exact native `SO(d)` transport via skew generators and Cayley/exp maps
- **Discrete Ricci-flow surgery** — log-conformal multiplicative weight updates with anti-thrashing hysteresis
- **Multi-operator curvature diagnostics** — Bakry–Émery, Ollivier (log-Sinkhorn), LLY, Forman, entropic
- **Transactional structural evolution** — shadow rollout, dual actuation/diagnostic audits, accept/reject/quarantine
- **Counterfactual structural reasoning (v5.4)** — bounded multi-channel concrete candidate generation, permutation-equivariant graph encoding, probabilistic `Q(S,a)` scoring, uncertainty/risk/information-gain estimates, governor-grounded replay generation, and exact-authority runtime bridging

The core principle: **field dynamics are sparse and compiled; discrete evolution is transactional and eager; curvature diagnoses rather than directly dictates topology.**

> **Naming.** The release is **LGAE v5.9.0**, the repo is [`dawsonblock/LGAE`](https://github.com/dawsonblock/LGAE). The Python distribution/module is `lgae-v3` / `lgae_v3` — a historical name kept for import stability (renaming would touch every import in 60 source + 36 test files for no functional gain). `pip install lgae-v3` gives you `import lgae_v3` at version `5.9.0`. The CLI is `lgae-v3`.

---

## Validation status

> **Read this first.** The geometry/numerical oracles (Bakry–Émery, LLY, Ollivier, SO(d), log-Sinkhorn, LOBPCG) are verified against analytic ground truth and are the strongest part of this codebase. The *structural-policy* qualification (diagnosis accuracy / mutation regret) is measured on **six hand-authored synthetic tasks** with no external dataset, and the first baseline comparison shows the learned policy **does not yet transfer** to structurally held-out tasks as well as a cheap non-learned heuristic. See [Validation boundaries](#validation-boundaries).

| What | Status | Evidence |
| :--- | :--- | :--- |
| Geometry/numerical oracles | **9/9 PASS** vs analytic ground truth | `scripts/qualify.py`, `qualification_report.json` |
| Production-dynamics checks | **8/8 PASS** | `scripts/qualify_production.py` |
| Structural-policy gate | **PASS** (83.3% acc / 0.0176 regret, deterministic) | `scripts/qualify_policy.py` |
| Test suite | **601/601 passing** (~15s single invocation) | `pytest -q` |
| Baseline comparison | Learned beats baselines in-distribution, **loses to heuristic held-out** | `scripts/compare_baselines.py` |
| Real-world experiment | **Negative result** on Karate Club (governor commits nothing) | `scripts/run_real_experiment.py` |
| `torch.compile` perf | **Not qualified** this release (Inductor smoke timed out) | `release_verification.json` |

---

## v5.4 structural reasoning

The v5.4 reasoner changes the learning target from action-class classification to concrete intervention valuation:

```text
(graph state S, concrete action a) -> predicted delta utility, uncertainty, risk, information gain
```

The runtime path is deliberately bounded and safety-separated:

```text
encode graph -> generate <=64 candidates -> learned Q ranking -> exact governor precheck -> transactional engine commit
```

`NO_OP` is always present, candidate generation mixes heuristic/latent/exploration channels, and learned code cannot commit graph state directly. `CounterfactualFactory` can generate governor-grounded training tuples for edge mutations, while `StructuralReasoningLoop` re-certifies the selected concrete mutation through `LGAEEngine.evaluate_and_maybe_commit`.

Quick smoke:

```bash
PYTHONPATH=src python scripts/qualify_reasoning.py
```

The new reasoning qualification is an architectural smoke test, not a claim of general structural intelligence. The existing held-out benchmark limitation remains the scientific gate: v5.4 must be trained on a substantially larger counterfactual dataset and must beat the spectral heuristic on unseen topology families before the learned reasoner is considered qualified.

---

## Quickstart

```bash
# Clone
git clone https://github.com/dawsonblock/LGAE.git
cd LGAE

# Install (editable, with dev deps)
python -m pip install -e '.[dev]' --no-build-isolation

# Run the self-evolving graph demo
lgae-v3 demo --nodes 10 --steps 4

# Verify geometric oracles against analytic ground truth
python scripts/qualify.py

# Run the deterministic policy qualification gate
python scripts/qualify_policy.py

# Compare learned executive vs baselines (random / spectral-heuristic / oracle)
python scripts/compare_baselines.py

# Run the real-world Karate Club experiment
python scripts/run_real_experiment.py
```

---

## Code examples

### Gauge parallel transport on fiber bundles

```python
import torch
from lgae_v3 import LGAEConfig, LGAEEngine, make_bucketed_graph_buffers

cfg = LGAEConfig()
cfg.fiber.d_base = 8
cfg.fiber.d_max = 16
cfg.fiber.gauge_dim = 8
cfg.fiber.gauge_parameterization = "cayley"  # or "exp"

graph = make_bucketed_graph_buffers(num_nodes=4, edges=[(0,1),(1,2),(2,3),(3,0)], bucket_size=256)
engine = LGAEEngine(graph, cfg)
engine.diffuse_(eta=0.01)

# Verify SO(d) invariants to machine precision
orth_err, det_err = engine.gauge_connections.invariant_error()
print(f"Max orthogonality error: {orth_err.max():.2e}")
print(f"Max determinant error:   {det_err.max():.2e}")
```

### Differentiable training with optimizer isolation

```python
import torch
from torch import nn
from lgae_v3 import LGAEConfig, LGAEEngine, LGAETrainCore
from lgae_v3.training import padded_markov_edges_with_slots, train_step

cfg = LGAEConfig(); cfg.fiber.d_base = 4; cfg.fiber.d_max = 8; cfg.fiber.gauge_dim = 4
graph = make_bucketed_graph_buffers(4, [(0,1),(1,2),(2,3)], bucket_size=32)
engine = LGAEEngine(graph, cfg)

core = LGAETrainCore(engine.fibers, nn.Linear(8, 2), gauge_bank=engine.gauge_connections, gauge_dim=4)
optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)

src, dst, w, valid, slot, reverse = padded_markov_edges_with_slots(graph, max_edges=32)
metrics = train_step(core, engine, optimizer, target=torch.randn(4, 2),
                     src=src, dst=dst, weight=w, valid=valid,
                     bottleneck_pressure=torch.zeros(4), edge_slot=slot,
                     reverse=reverse, step=0, spawn_interval=50)
print("Step Loss:", metrics["loss"].item())
```

### Curvature auditing & governed surgery

```python
mutation = engine.propose_midpoint_edge()
result = engine.evaluate_and_maybe_commit(mutation)
print("Decision:", result.decision.value)  # 'accept', 'reject', or 'quarantine'
print("Reasons:", result.reasons)
print("Authority hash after:", result.metadata.get("authority_hash_after"))
```

### Safe checkpoints & hash-chained receipts

```python
# Safe format (safetensors + JSON) for untrusted interchange
engine.save_checkpoint("checkpoint_dir/")
engine2.load_checkpoint_("checkpoint_dir/",
    allow_governance_mismatch=True,
    optimizer_load_policy="restore",  # "restore" | "reset" | "reject"
)
engine2.assert_generation_sync()  # raises on graph/gauge generation divergence

# Tamper-evident receipt ledger
from lgae_v3.receipts import mutation_receipt, append_receipt, verify_receipt_chain
append_receipt("ledger.jsonl", mutation_receipt(result,
    authority_state_hash_before=engine.authority_hash(),
    gauge_authority_hash=engine.gauge_connections.state_hash()))
is_valid, errors = verify_receipt_chain("ledger.jsonl")
assert is_valid
```

---

## CLI

```bash
lgae-v3 demo --nodes 10 --steps 4              # self-evolving graph demo
lgae-v3 qualify-lly --graph cycle --nodes 6    # cross-validate LLY curvature
lgae-v3 qualify-lly --graph path --nodes 4
lgae-v3 qualify-lly --graph complete --nodes 4

python scripts/qualify.py                       # geometric oracles (9/9)
python scripts/qualify_policy.py                # deterministic policy gate
python scripts/qualify_production.py            # production-dynamics checks (8/8)
python scripts/compare_baselines.py             # baseline comparison
python scripts/run_real_experiment.py           # real-world Karate Club experiment
python scripts/generate_manifest.py --check     # verify SHA-256 integrity manifest
```

---

## Architecture

```
       +-------------------------------------------------------------+
       |                  Continuous Field Dynamics                  |
       |  - Latent states z in R^{N x D} with dynamic fiber gating   |
       |  - Sparse row-stochastic Markov diffusion (O(E x D))        |
       |  - SO(d) gauge parallel transport: U_e in SO(d_g)           |
       +------------------------------+------------------------------+
                                      |
                         Fast Geometric Signals (Gamma, r, Var)
                                      v
       +-------------------------------------------------------------+
       |               Transaction & Shadow Evaluation               |
       |  - Eager shadow rollout with dual actuation/diagnostics     |
       |  - Log-conformal Ricci flow: w' = clamp(w * exp(-dt * dk))  |
       |  - Graph surgery: add, reweight, prune with hysteresis      |
       +------------------------------+------------------------------+
                                      |
                      Curvature & Topological Audits
                                      v
       +-------------------------------------------------------------+
       |                   Authoritative Governor                    |
       |  - Exact LLY & Log-Sinkhorn Ollivier (W1 optimal transport) |
       |  - Reversible Bakry-Émery CD(K, N) with Schur complements   |
       |  - Sparse LOBPCG spectral certificate & beta_0 protection   |
       |  - Accept, Reject, or Quarantine with SHA-256 state locks   |
       +-------------------------------------------------------------+
```

The authoritative loop (v5.1.1+):

```
observe → learned proposal/target → counterfactual comparison → uncertainty gate
→ LGAEEngine transaction → ACCEPT / QUARANTINE / REJECT
→ outcome receipt → long-horizon credit → executive/ensemble update
```

`LGAEEngine` is the sole structural commit authority; governor-only use is read-only certification. `QUARANTINE` is never treated as execution and never receives committed-action credit. Graph, fiber, and gauge changes all have transactional shadow/rollback/quarantine paths.

---

## Key features

### SO(d) gauge connection bank

- **Lie-algebra parameterization**: generator parameters in unconstrained space, mapped through skew-symmetric algebra $\mathfrak{so}(d)$ via $A_e = \tfrac{1}{2}(R_e - R_e^T)$, then to $\mathrm{SO}(d)$ via Cayley retraction or matrix exponential.
- **Guaranteed invariance**: $U_e^T U_e = I$ and $\det(U_e) = +1$ to machine precision across arbitrary optimizer steps.
- **Slot generation lifecycle**: monotonic generation counters track slot allocation/retirement; cryptographically committed in the graph state hash.
- **Optimizer momentum isolation**: retired/reused edge slots zero all tensor-valued optimizer state (Adam, AdamW, SGD, Adagrad, RMSProp).

### Stable optimal transport (log-Sinkhorn Ollivier)

- **Log-domain scaling**: eliminates probability-space underflow at small $\epsilon$.
- **Zero-mass pruning**: exact support removal for unvisited states.
- **Marginal-residual certification**: convergence validated against recovered coupling marginals.
- **Exact ground-truth oracle**: high-precision LP retained for qualification.

### Reversible Γ-calculus & Bakry–Émery CD(K, N)

- **Continuous-time reversible Markov generators** with detailed-balance volume measure.
- **Float64 conditioning**: precision row re-normalization and diagonal ULP cancellation.
- **Γ-nullspace Schur complement**: eliminates uncoupled higher-hop coordinates, preventing false-positive curvature anomalies.

### Log-conformal Ricci flow & surgery hysteresis

- **Weight positivity**: $w \leftarrow \mathrm{clamp}(w \cdot \exp(-\Delta t(\kappa - \kappa^*)), w_{\min}, w_{\max})$ — weights never cross zero.
- **Anti-thrashing cooldown**: canonical edge cooldown tracker separates addition, deadband, and pruning regions.
- **O(V+E) bridge filter**: rejects disconnecting edge removals before expensive global audits.

### torch.compile compatibility

- **Fixed-shape buffer bucketing**: `GraphBuffers` round capacity to fixed-size buckets with in-place refresh.
- **Dormant fiber channel suppression**: inactive latent coordinates zeroed post-diffusion.

---

## Mathematical oracles

All oracles verified against analytic ground truth. Run with `python scripts/qualify.py`.

| Oracle | Test case | Theoretical target | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| Bakry–Émery $K_\infty$ | Path $P_4$ (interior) | $1 - \tfrac{\sqrt{2}}{2} \approx 0.292893$ | `0.2928932188` | PASS |
| Bakry–Émery $K_\infty$ | Path $P_4$ (endpoints) | $1.0$ | `1.0000000000` | PASS |
| Bakry–Émery $K_\infty$ | Complete $K_2$ | $2.0$ | `2.0000000000` | PASS |
| Exact LLY agreement | $K_2, P_4, C_4, K_3$ | $\kappa_{\text{LP}} = 2\kappa_{1/2}$ | Max error: `0.0` | PASS |
| Weak entropic curvature | $K_3$ (empty 2-hop shell) | $+\infty$ | `Infinity` | PASS |
| Log-Sinkhorn vs LP | Large metric / small $\epsilon$ | $998.0$ | `997.999999999` | PASS |
| $SO(d)$ invariance | Post-Adam steps | $\|U^T U - I\|_F < 10^{-10}$ | `Pass` | PASS |
| Sparse LOBPCG spectral gap | Cycle $C_{24}$ | Matches exact $\lambda_2$ | `0.03407417` | PASS |

---

## Baseline comparison

`scripts/compare_baselines.py` runs four controllers — random, spectral-heuristic, learned, oracle — on the same tasks and reports diagnosis accuracy and mean regret, on both in-distribution and truly held-out structurally-distinct task variants.

```
controller             split                  accuracy     regret
random                 in_distribution          0.1000     2.6918
spectral_heuristic     in_distribution          0.4667     2.2669
learned                in_distribution          0.8333     0.0176
oracle                 in_distribution          1.0000     0.0000
random                 held_out_structure       0.1000     3.0092
spectral_heuristic     held_out_structure       0.6000     2.4000
learned                held_out_structure       0.3000     0.6175
oracle                 held_out_structure       1.0000     0.0000
```

**Reading.** The learned executive beats random and the spectral heuristic in-distribution. On structurally held-out tasks it **loses to the spectral heuristic on accuracy** (30% vs 60%) but has lower regret (0.62 vs 2.40), suggesting it defaults to safe NO_OP on unseen structures rather than misdiagnosing. This is the first honest generalization signal the benchmark has produced.

---

## Real-world experiment

`scripts/run_real_experiment.py` uses Zachary's Karate Club (a real 34-node social network with two ground-truth communities, shipped with NetworkX) and a real downstream task: recovering the two communities from a latent embedding via clustering.

```
condition                accuracy    lambda2    edges
raw                        0.7059     0.1323       78
random_add                 0.7353     0.1632       84
spectral_heuristic         0.7941     0.1249       84
lgae_governed              0.6765     0.1323       78
```

**Reading.** On this real task the LGAE governance loop does **not** improve community recovery over the raw baseline: the governor commits zero mutations on a graph this small (all proposals rejected by shadow audits), and the engine's fiber latent is a slightly worse representation for clustering than the raw spectral embedding. A simple spectral heuristic does best. This is an honest negative result — it does not support a "production-ready controller" claim, and is the starting point for understanding *when* the governance machinery helps.

---

## Validation boundaries

The oracles above are verified against analytic ground truth and are the strongest claim this codebase makes. The following are **not** claims:

- **No real-world generalization claim.** The structural-policy qualification uses six hand-authored synthetic tasks. "Held-out" seeds 101–105 reused the *same task structures* as training seeds 0–15, differing only in latent noise. The new `compare_baselines.py` adds truly held-out structurally-distinct variants, where the learned policy loses to a non-learned heuristic.
- **Benchmark utility is constructed, not discovered.** Each task's `utility()` is written so the labeled "correct" action is the argmax of Δ-utility. The learned executive is graded on whether it recovers the action the utility was built to reward — a training-loop consistency check, not evidence that the governor diagnoses real structure.
- **No baseline comparison until v5.3.1.** The new `compare_baselines.py` is the first comparison to random-action, spectral-heuristic, and oracle controllers. The learned executive does not yet beat the spectral heuristic on held-out tasks.
- **No deployment-safety proof.** The governor's transactional shadow/rollback/quarantine and fail-closed numerical behavior are engineering safeguards, not a formal safety argument.
- **`torch.compile` not performance-qualified this release.** The fresh CPU Inductor smoke timed out during compilation; the compiled-kernel architecture is inherited from v5.2 and not newly measured. See `release_verification.json`.

---

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.6+
- NumPy, SciPy, NetworkX, PyYAML

### From source

```bash
git clone https://github.com/dawsonblock/LGAE.git
cd LGAE
python -m pip install -e '.[dev]' --no-build-isolation
```

### From wheel

```bash
pip install lgae-v3
```

Optional persistent-homology backend:

```bash
pip install 'lgae-v3[ph]'    # installs ripser
```

---

## Repository structure

```
.
├── .github/workflows/ci.yml     # CI: tests, oracles, policy gate, baselines, real experiment
├── src/lgae_v3/
│   ├── core/                    # compatibility layer and engine entrypoints
│   ├── curvature/               # Bakry-Émery, CDE, Entropic, Forman, LLY, Ollivier
│   ├── training/                # LGAETrainCore, padded buffers, train loops
│   ├── benchmark/               # synthetic tasks, metrics, baselines, policy qualification
│   ├── evolution.py             # authoritative LGAEEngine
│   ├── governor.py              # GeometryGovernor & transition audits
│   ├── fibers.py                # FixedWidthFiberLatent & SOConnectionBank
│   ├── executive.py             # learned structural executive (proposal only)
│   ├── mutations.py             # log-conformal Ricci flow & graph surgeries
│   ├── operators.py             # actuation & diagnostic Markov operators
│   ├── receipts.py              # cryptographic receipt logging
│   └── topology.py              # NetworkX conversion, Betti numbers & PH
├── scripts/
│   ├── qualify.py               # geometric oracles (9/9)
│   ├── qualify_policy.py        # deterministic policy gate
│   ├── qualify_production.py    # production-dynamics checks (8/8)
│   ├── compare_baselines.py     # baseline comparison (random/heuristic/learned/oracle)
│   ├── run_real_experiment.py   # real-world Karate Club experiment
│   └── generate_manifest.py     # SHA-256 integrity manifest
├── tests/                       # 37 test modules, 601 tests
├── docs/                        # architecture, mathematics, reading list, release notes
├── configs/default.yaml         # default engine and audit configuration
└── pyproject.toml
```

---

## Release history

### v5.3.1 — Integrity & baseline-comparison fixes (current)

- **Fixed nondeterministic policy qualification** (release-gate bug): seed was set after network init, making accuracy a random draw (86.7%–100% on identical runs). Now deterministic at 83.3%.
- **Removed circular benchmark utility** in Task A (rewarded the correct action's signature directly); replaced with pure spectral gap.
- **Added baseline controllers** (random / spectral-heuristic / oracle) and `compare_baselines.py`.
- **Added truly held-out task variants** with different graph structures (original "held-out" seeds reused identical structures).
- **Added real-world experiment** (`run_real_experiment.py`): Karate Club community recovery — honest negative result.
- **Corrected false "bounded batches" claim**: full suite passes in ~15s single invocation.
- **Regenerated stale reports** and `example_output.txt`.
- **Toned down README overclaims**; added Validation boundaries section.
- See `CHANGELOG.md` for full details.

### v5.3.0 — Production dynamics hardening

Exact native `SO(d)` gauge transport retained; external sheaf maps gain non-expansive transport guards; automatic surgery gains curvature EMA/variance hysteresis; directed kernels gain stationary-measure Γ₂ symmetrization; ANN/neighbor caches become transaction-generation aware; slow structural updates can wait for latent equilibrium; mutation learning uses graph-conditioned counterfactual advantages.

### v5.2.0 — Structural policy qualification

Compact latent-state diagnostics, bounded learned mutation magnitudes, explicit risk learning from REJECT/QUARANTINE, ensemble-variance information-gain proxy, persistent pending credit, supervised policy-prior head.

### v5.1.1 — Closed-loop authority integration

`LGAEEngine` as sole structural commit authority; `QUARANTINE` never treated as execution; transactional shadow/rollback/quarantine for graph/fiber/gauge; safe checkpoints with safetensors; bootstrap ensemble uncertainty read-only; conformal residual calibration; long-horizon credit; sheaf-Laplacian sign fix; persistent-homology bottleneck minimax matching; Bakry–Émery two-hop neighborhood requirement; ANN self-neighbor removal; dynamic-gauge antisymmetric generators.

### v5.0.0 — Structural learning loop

Learned structural executive, long-term mutation credit assignment, calibrated uncertainty (LCB gate), stability/plasticity + consolidation, task-grounded benchmark harness (6 synthetic tasks), structural counterfactual engine, closed loop.

### v4.x — Sparse weighted geometry, metric-measure separation, authority/persistence hardening

Sparse dual operators (O(Nk)), weighted curvature backends, metric–measure separation (affinity vs length), multi-horizon shadow certification, canonical authority hash, slot-generation binding, safetensors checkpoints, hash-chained receipts.

### v3.x — Original geometry governor

SO(d) gauge bank, log-Sinkhorn Ollivier, reversible Γ-calculus, log-conformal Ricci flow, sparse LOBPCG, transactional dry-run, fixed-capacity `torch.compile` boundaries.

---

## License

[MIT](LICENSE)


## v5.8.0 concurrent snapshot hardening

v5.8.0 extends commit-boundary cache coherence with an optimistic seqlock-style read epoch. Long-running derived computations can capture a stable `GraphReadView`, run without holding a global reader lock, and publish only if the mutation epoch and authoritative graph generation are unchanged at completion. In-place graph transactions mark the epoch unstable for their entire mutation window; both commits and rollbacks invalidate overlapping reads. `run_consistent_read()` provides bounded retry semantics for curvature, spectral, candidate-generation, and reasoning workloads.

## v5.6.1 cache-coherence hardening

The v5.6.1 release centralizes derived-cache lifecycle at the authoritative transaction commit boundary. `CommitEventBus` publishes a single monotonic `GraphCommitEvent` only after successful commits; rollback does not publish an authoritative event. Caches can declare dependencies (`TOPOLOGY`, `WEIGHTS`, `LENGTHS`, `LATENTS`, `FIBERS`, `GAUGE`, `ROLES`), retain valid results across unrelated mutations, and fail closed when a generation stamp is stale. Delta transactions report touched nodes and edges without scanning the complete graph.

## v5.8.0 adaptive geometry runtime

v5.8.0 adds operator-declared structural dependency footprints, automatic local/global cache-scope derivation, orthogonality drift telemetry with threshold-triggered polar repair, and an adaptive curvature cascade. Runtime screening can stop at inexpensive Forman geometry when confidence is high, escalate through LLY and the existing log-domain Sinkhorn Ollivier solver under ambiguity, and require exact OT for high-risk or certification paths. Exact geometry and the governor remain authoritative.

## v5.8.0 Structural Intelligence Qualification

v5.8.0 moves the development focus from adding geometric operators to measuring whether the learned structural executive can select useful interventions on topology families excluded from training. The new `structural_intelligence.py` layer adds state-grouped replay, an ensemble candidate-value model with epistemic/aleatoric uncertainty decomposition, effective-resistance candidate retrieval, procedural graph families, and exact candidate-regret/calibration metrics.

The qualification boundary is intentionally strict: `structural_intelligence_qualification_report.json` may report the infrastructure `PASS` while `scientific_generalization_status` remains `NOT_YET_QUALIFIED`. The bundled smoke-scale training run is not evidence of production-level structural intelligence; it exists to prove that the data-generation, grouped-training, held-out-family, regret, and calibration pipelines execute deterministically and expose failure rather than hide it. Exact shadow evaluation and the geometry governor remain authoritative.

## v5.8.1 Structural Intelligence Hardening

v5.8.1 hardens the learned decision layer rather than changing the geometric authority path. It adds randomized-prior Q ensembles for persistent epistemic diversity, structural-regime stratified state-grouped replay, a trainable contrastive nonedge retriever, effective-resistance and exact spectral-gap reference proposal channels, a BORF-style AF3/Forman-flow proxy channel, and conservative lower-confidence-bound arbitration.

The qualification boundary is explicit: the bundled smoke-scale training run measures candidate recall, held-out regret, and ID/OOD uncertainty but does **not** claim unseen-topology superiority unless the generated hardening report marks `scientific_generalization_status` as `PASS`. FoSR is represented by an exact spectral-gap-gain reference on bounded qualification graphs; the Forman-flow channel is explicitly a BORF-style proxy rather than a paper-exact BORF implementation.



## v5.8.3 Joint Structural Action & Local Credit

v5.8.3 couples ADD_EDGE proposals with low-rank Lie-algebra connection initialization.  Candidate gauge maps are evaluated only in shadow rollouts until the GeometryGovernor accepts the topology mutation; authoritative gauge parameters are initialized only after commit.  The release also adds normalized connection-Dirichlet credit and graph-distance attribution so local repairs are not trained solely from graph-global returns.

Scientific claim boundary: this release qualifies joint-action plumbing, SO(d) invariants, shadow-only certification, and conservative local-credit accounting.  It does **not** yet claim that learned joint actions outperform topology-only proposals on unseen graph families.

## v5.8.2 Scalable Structural Intelligence

Adds ANN-backed learned candidate inference without dense N×N logits, a production first-order FoSR/Fiedler proposal channel, monotone online epistemic-scale calibration, contextual action-dependent LCB conservatism, and 1-WL-assisted replay redundancy gating. Exact FoSR remains a bounded qualification oracle.

## v5.8.4 — Joint Action Runtime Optimization

v5.8.4 replaces the joint proposal head's matrix-exponential hot path with a solve-based Cayley retraction, adds explicit dual restriction-map pairing for undirected cellular edges, qualifies self-adjoint `B^T B` connection-Laplacian assembly, and extends localized credit to the two-sided sheaf Dirichlet flux. The bundled CPU benchmark does not show a universal Cayley speedup; see `joint_action_runtime_qualification_report.json`.
