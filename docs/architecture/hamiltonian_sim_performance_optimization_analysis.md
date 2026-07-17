# Hamiltonian Simulation Performance: Optimization Design Analysis

> **Status:** Design investigation only — no implementation.  
> **Date:** 2026-07-17  
> **Scope:** Bottleneck in Atlas dynamics execution path  
> **Evidence base:** Instrumented pipeline profiles (`ATLAS_PROFILE=1`) and
> controlled micro-benchmarks against Qiskit 2.4.2

---

## 1. Summary of the Bottleneck

### Measured pipeline hotspot

Across the instrumented sweep, approximately **97%** of simulated-evolution time
is spent in:

```text
HamiltonianSimulation.run
  └── StatevectorEvolver.evolve   (span: sim.evolver_evolve)
        └── Statevector(circuit)
```

### Measured evolver-internal hotspot

A diagnostic walk mirroring Qiskit’s `Statevector.from_instruction` path shows:

| Stage | Share of evolver internals |
|-------|----------------------------|
| `PauliEvolutionGate.to_matrix()` via `Operator._instruction_to_matrix` | **~98%** |
| `Statevector._evolve_operator` (apply dense unitary) | ~1–2% |
| `circuit.to_instruction`, `|0…0⟩` init, `np.asarray`, result pack | ≲1% |

### What that `to_matrix()` call does

From Qiskit `PauliEvolutionGate.to_matrix()`:

1. Take the (typically sparse) Pauli operator.
2. Build a sparse matrix.
3. Call `scipy.sparse.linalg.expm(-i t A)`.
4. Densify to a full **`2ⁿ × 2ⁿ`** NumPy array.

Atlas then (via Qiskit) wraps that dense matrix as an `Operator` and applies it
to the statevector — which is cheap compared with building the matrix.

### Scale of repeated work (TFIM Strang example)

For the profiled `n=4`, Strang, 10 steps circuit:

- **140** `PauliEvolution` instructions
- **140** dense matrix exponentials
- **140** dense `2ⁿ×2ⁿ` allocations

Only ~11 unique `(Pauli, Δt)` pairs appear in that circuit; without caching or
decomposition, Qiskit still pays the full dense-expm cost **per gate instance**.

Micro-benchmark (`n=6`, Strang, 10 steps, 220 gates):

| Path | Wall time |
|------|-----------|
| `Statevector(PauliEvolution circuit)` | **2.50 s** |
| `Statevector(circuit.decompose())` (basis gates) | **0.008 s** (~300×) |
| Unique-gate `to_matrix` cache only (11 unique) | **0.11 s** |
| One-shot `expm_multiply(-iHt, |ψ₀⟩)` on full H | **0.001 s** |

---

## 2. Root-Cause Analysis

### Where the cost lives

| Layer | Role in the hotspot | Verdict |
|-------|---------------------|---------|
| **Atlas `StatevectorEvolver`** | Thin wrapper: `Statevector(circuit)` | **Not** the algorithmic cost; chooses a slow *simulation path* |
| **Atlas product formulas** | Emit many `PauliEvolutionGate`s | **Amplifies** the cost (gate count × steps) |
| **Qiskit `Statevector(circuit)`** | For each gate: `to_matrix` then evolve | **Execution mechanism** that triggers the hotspot |
| **Qiskit `PauliEvolutionGate.to_matrix`** | Sparse `expm` → dense unitary | **Dominant computational kernel** |
| **SciPy `expm`** | Matrix exponential | Called inside Qiskit’s gate matrix path |
| **Dense `Operator` apply** | Matrix–vector / unitary apply | Minor (~1%) |

### Root cause (precise)

The bottleneck is **not** “statevector simulation is intrinsically slow.”

It is:

> **Simulating a product-formula circuit by converting every
> `PauliEvolutionGate` into a dense `2ⁿ×2ⁿ` unitary via
> `PauliEvolutionGate.to_matrix()`, once per gate instance.**

Atlas currently:

1. Builds the right *logical* Trotter circuit (`circuits/evolution`).
2. Executes it with the most general Qiskit Quantum Info path
   (`Statevector(circuit)`), which is a poor fit for long sequences of
   `PauliEvolutionGate`s.

So:

- **Fundamental expensive operation:** Qiskit / SciPy dense gate-matrix expm.
- **Architectural choice that exposes it:** Atlas evolver uses
  `Statevector(circuit)` on undecomposed Pauli-evolution circuits.
- **Workload multiplier:** Strang/Lie emit O(steps × terms) such gates.

Evidence:

- Profile: ~98% of evolver time in `instruction_to_matrix` / `to_matrix`.
- Same circuit after `decompose()` simulates ~300× faster with identical state
  (error ~1e-15 on the micro-benchmark).
- Exact full-Hamiltonian `expm_multiply` is orders of magnitude cheaper than
  per-term dense gate matrices for small `n` — showing the issue is the
  *per-gate dense unitary construction*, not “evolving a 2ⁿ state.”

---

## 3. Optimization Strategies

For each technique: description, expected gain, memory, complexity,
correctness, maintainability, Atlas vs Qiskit, backend-agnostic fit.

### A. Decompose `PauliEvolutionGate` before statevector simulation

**Description:** Before `Statevector(...)`, call `circuit.decompose()` (or
synthesize to `{rx, ry, rz, rzz, cx, …}`) so simulation uses elementary gates
with cheap built-in matrices / kernels.

**Expected gain:** **Very high** (micro-benchmark ~300× on evolver path).  
**Memory:** Lower peak (avoid many dense `2ⁿ×2ⁿ` unitaries).  
**Complexity:** Low (one-line or small helper in evolver / circuit finalize).  
**Correctness:** High if using Qiskit’s definition; verify phases/global phase.  
**Maintainability:** High.  
**Locus:** Mostly **Qiskit path selection**; Atlas chooses when to decompose.  
**Backend-agnostic:** Yes for statevector backends; hardware may want
undecomposed or differently transpiled forms.

### B. Cache dense matrices for identical `(Pauli, time)` gates

**Description:** Memoize `PauliEvolutionGate.to_matrix()` (or `Operator`) by
canonical Pauli string + coefficient-scaled time within a run / process.

**Expected gain:** High when uniqueness ≪ gate count (TFIM: ~11 unique vs 140–220
gates → order-of-magnitude). Less than full decompose for large unique sets.  
**Memory:** One dense `2ⁿ×2ⁿ` per unique key (grows as `u · 4ⁿ`).  
**Complexity:** Low–medium (cache layer; careful keying of time floats).  
**Correctness:** Exact reuse if keys match; watch float keying / parameter
expressions.  
**Maintainability:** Medium (cache invalidation / thread safety).  
**Locus:** Qiskit-facing cache, ideally behind Atlas evolver.  
**Backend-agnostic:** Optimization of a specific simulation strategy, not a
new physics API.

### C. Custom evolver: apply Pauli rotations without dense unitaries

**Description:** For each Trotter term, apply `exp(-i θ P)` directly to the
statevector (sparse `expm_multiply`, or Clifford conjugation + diagonal phase
for Pauli strings), never materializing `2ⁿ×2ⁿ` matrices.

**Expected gain:** **Very high**; scales far better with `n` than dense
`to_matrix`.  
**Memory:** O(2ⁿ) state only (plus sparse H_j).  
**Complexity:** Medium–high (correct Pauli conjugation / sparse apply).  
**Correctness:** Must match Qiskit phases; needs tests vs current path.  
**Maintainability:** Medium; Atlas-owned kernel.  
**Locus:** **Atlas-specific** execution backend.  
**Backend-agnostic:** Still an “evolver” implementation; algorithms unchanged.

### D. Circuit template reuse across sweep times

**Description:** Build one parameterized / structured Trotter circuit and only
bind time (or rebuild only time-dependent angles) across `evolution_time`
sweep points.

**Expected gain:** Medium for **circuit construction** (currently ~2% of time);
small unless combined with A/B/C. Does not fix per-gate `to_matrix` alone.  
**Memory:** Low.  
**Complexity:** Medium (parameterized product formulas).  
**Correctness:** High if angles mapped correctly.  
**Maintainability:** Medium.  
**Locus:** `circuits/evolution` + algorithm orchestration.  
**Backend-agnostic:** Yes.

### E. Reuse Hamiltonian / observables / evolution method across sweep

**Description:** Stop rebuilding invariant objects every sweep iteration
(already measured as rebuilt every point but cheap today).

**Expected gain:** Low for current hotspot (sub-ms). Important for cleanliness
and larger models later.  
**Memory:** Low (hold references).  
**Complexity:** Low.  
**Correctness:** High if parameters truly fixed.  
**Maintainability:** High.  
**Locus:** `experiments/hamiltonian_sim_experiment.py`.  
**Backend-agnostic:** Yes.

### F. Fuse commuting Pauli terms / larger chunks

**Description:** Group commuting terms into fewer `PauliEvolutionGate`s (or one
per mutually commuting set) to reduce gate count and `to_matrix` calls.

**Expected gain:** Medium (fewer gates); each gate’s `to_matrix` may be denser /
harder. Trade-off depends on grouping.  
**Memory:** Similar or higher per fused gate.  
**Complexity:** Medium (commutation graphs).  
**Correctness:** Exact if groups commute.  
**Maintainability:** Medium.  
**Locus:** `circuits/evolution`.  
**Backend-agnostic:** Yes.

### G. Gate fusion / transpile with optimization

**Description:** Run Qiskit transpiler / `optimize_1q_gates` etc. after
decomposition.

**Expected gain:** Medium on top of A; small alone on PauliEvolution circuits.  
**Memory:** Low.  
**Complexity:** Low–medium.  
**Correctness:** High at `optimization_level` appropriate for exact sim.  
**Maintainability:** High.  
**Locus:** execution / optional pass.  
**Backend-agnostic:** Transpile is backend-aware by nature; for exact SV sim use
basis-gate optimization without coupling map constraints.

### H. Qiskit Aer statevector / GPU methods

**Description:** Optional `qiskit-aer` `AerSimulator(method="statevector")` (or
GPU/density variants).

**Expected gain:** Potentially high for **large** `n` and elementary-gate
circuits; **does not by itself** fix PauliEvolution `to_matrix` if Aer also
expands gates similarly — must verify Aer’s handling of `PauliEvolutionGate`.
Likely still want decompose first.  
**Memory:** Aer-dependent.  
**Complexity:** Medium (new optional dependency + backend).  
**Correctness:** High for ideal SV methods.  
**Maintainability:** Medium (extra dependency).  
**Locus:** `execution/` new evolver.  
**Backend-agnostic:** Fits multi-backend Atlas design; Aer not required for all
users.

### I. Estimator / primitives

**Description:** Use `StatevectorEstimator` or Sampler-style primitives.

**Expected gain:** **Low / wrong tool** for returning a full statevector for
fidelity vs exact evolution. Estimators target expectation values, not
`|ψ(t)⟩`.  
**Memory:** N/A.  
**Complexity:** N/A for this goal.  
**Correctness:** Not applicable to current `SimulationResult.statevector` API.  
**Recommendation:** Out of scope for this bottleneck.

### J. Single dense / sparse full-H exact evolution as “simulation”

**Description:** Replace Trotter circuit simulation with `expm_multiply(-iHt, ψ)`
on the full Hamiltonian.

**Expected gain:** Huge vs current path for small `n` — but this **changes the
algorithm under test** (no longer product-formula approximation). Useful as a
reference, not as a drop-in replacement for Trotter validation.  
**Memory:** Sparse H + state.  
**Complexity:** Low.  
**Correctness:** Exact Schrödinger evolution, not Trotter.  
**Locus:** Already exists as `Hamiltonian.exact_time_evolution`.  
**Architecture:** Must remain separate from approximate evolver.

### K. Lazy / delayed matrix construction

**Description:** Avoid building matrices until apply time; use
`expm_multiply` on sparse generators per term.

**Expected gain:** High (related to C).  
**Memory:** Better than dense unitaries.  
**Complexity:** Medium.  
**Correctness:** High with care.  
**Locus:** Custom evolver.  
**Backend-agnostic:** Atlas execution detail.

### L. Alternate Qiskit APIs (`HamiltonianGate`, evolution libraries)

**Description:** `HamiltonianGate` still centers on matrix data; Qiskit
algorithms evolution modules historically target circuits/estimators. None
clearly replace per-term dense `to_matrix` for QI `Statevector` without
decomposition or a custom apply path.

**Expected gain:** Uncertain / likely low without A or C.  
**Complexity:** Medium research.  
**Recommendation:** Prefer A/C over chasing alternate gate wrappers.

### M. Batched evolution / vectorization across times

**Description:** Evolve many times in one pass or batch state copies.

**Expected gain:** Medium for sweeps if kernel supports batching; does not
remove per-gate dense expm unless combined with A/C.  
**Complexity:** High.  
**Locus:** execution / experiment.  
**Scalability:** Good for parameter sweeps later.

### N. Avoid Strang’s doubled term applications at simulation level

**Description:** Strang uses forward/backward half-steps → more gates than Lie.
Using Lie for speed demos reduces gate count but changes convergence order.

**Expected gain:** Modest constant factor; not a root-cause fix.  
**Architecture:** Algorithm choice, not an evolver optimization.

---

## 4. Trade-off Matrix (condensed)

| ID | Technique | Perf | Effort | Clean arch | Scales with n | Arbitrary H | Future methods |
|----|-----------|------|--------|------------|---------------|-------------|----------------|
| A | Decompose then SV | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| B | Cache gate matrices | ★★★★ | ★★★★ | ★★★ | ★★ | ★★★★ | ★★★★ |
| C | Direct Pauli apply / expm_multiply | ★★★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| D | Circuit templates | ★★ | ★★★ | ★★★★ | ★★★ | ★★★★ | ★★★★ |
| E | Reuse sweep invariants | ★ | ★★★★★ | ★★★★★ | ★★★ | ★★★★★ | ★★★★★ |
| F | Commuting fusion | ★★★ | ★★★ | ★★★★ | ★★★ | ★★★ | ★★★ |
| G | Post-decompose optimize | ★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| H | Aer backend | ★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| I | Estimator primitives | ★ | — | — | — | — | Wrong API |
| J | Full-H expm as “sim” | ★★★★★* | ★★★★★ | ★★ | ★★★ | ★★★★★ | Breaks Trotter meaning |
| K | Sparse apply (no dense U) | ★★★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| L | Alt Qiskit gates | ★★ | ★★ | ★★★ | ★★ | ★★★ | ★★★ |
| M | Batched times | ★★★ | ★★ | ★★★ | ★★★ | ★★★★ | ★★★★ |
| N | Fewer Trotter gates (method) | ★★ | ★★★★★ | ★★★★ | ★★ | ★★★★★ | Changes algo |

\*Only if replacing Trotter simulation — usually unacceptable for validation.

---

## 5. Framework Placement

| Concern | Prefer module |
|---------|----------------|
| “How to execute a bound evolution circuit fast” | `execution/` (evolver implementations) |
| “What circuit represents the product formula” | `circuits/evolution/` |
| “Orchestrate build → evolve → `SimulationResult`” | `algorithms/hamiltonian_sim.py` |
| “Reuse H / observables across sweep” | `experiments/hamiltonian_sim_experiment.py` |
| Optional Aer / GPU | `execution/` + builders/YAML backend name |
| Caching policy | Inside evolver or thin `execution/cache` helper — **not** in physics |

**Do not** put matrix caches inside TFIM Hamiltonian classes or inside VQE
estimator paths.

Preserve:

```text
EvolutionMethod.build_circuit(...) → EvolutionCircuitSpec
Evolver.evolve(circuit) → EvolutionResult
```

Multiple evolvers (`StatevectorEvolver`, `DecomposingStatevectorEvolver`,
`AerStatevectorEvolver`, `PauliTermEvolver`) can share that contract.

---

## 6. Is Atlas Using the Intended Qiskit Path?

**For returning an exact statevector of a Quantum Info circuit:**  
`Statevector(circuit)` is a supported API.

**For efficiently simulating long Pauli-evolution product formulas:**  
No — this is a **general but inefficient** path. Qiskit’s own gate definitions
already lower `PauliEvolutionGate` to basis rotations (`rzz`, `rx`, …). The
intended practical approach for QI statevector simulation is closer to:

1. synthesize / decompose evolution gates, then simulate, or  
2. apply generators without dense unitaries.

Atlas is **not** using Aer; `qiskit-aer` is not in `requirements.txt`. Aer may
help later but is not required to fix the current `to_matrix` pathology.

Estimator primitives are the right tool for **expectation values**, not for
Atlas’s current fidelity workflow that needs full statevectors.

---

## 7. Recommended Implementation Order

1. **A — Decompose (or synthesize) before `Statevector` simulation**  
   Fastest win, tiny change, preserves Trotter meaning, backend-shaped as an
   evolver option or default for the QI path.

2. **E — Reuse sweep-invariant objects**  
   Cheap hygiene; small runtime win now, clearer contracts.

3. **G — Light optimization after decompose**  
   Optional 1q consolidation once A lands.

4. **B or C — Cache dense matrices *or* sparse/direct Pauli apply**  
   Choose based on scaling goals:
   - **B** if staying matrix-oriented and unique terms stay few.
   - **C/K** if targeting larger `n` (avoid `4ⁿ` memory).

5. **D — Parameterized / templated Trotter circuits**  
   After execution path is healthy.

6. **H — Optional Aer evolver**  
   After elementary-gate path exists; measure before committing dependency.

7. **F / M** — Fusion and batching as later scalability work.

**Explicitly defer / avoid as “the fix”:** I (estimators), J (replace Trotter
with exact H evolution for the approximate branch), N alone.

---

## 8. Risks and Limitations

- **Decomposition correctness:** Must preserve global phase and endianness;
  add golden tests vs current `Statevector(undecomposed)` on small systems.
- **Caching float keys:** Evolution times / Δt float equality can miss or false-hit;
  prefer rationalized keys from `(coeff, step_index)` where possible.
- **Dense cache memory:** Caching unitaries is O(u · 4ⁿ); dangerous above ~12–14
  qubits even with few unique terms.
- **Aer dependency:** Optional extra install; keep pure QI path working.
- **Hardware path:** Optimizations for local SV must not silently alter circuits
  intended for later hardware submission without an explicit policy.
- **Profiling double-count:** Diagnostic walks currently *add* work under
  `ATLAS_PROFILE`; production timings should disable diagnostics.

---

## 9. Clear Recommendation — First Optimization to Implement

### Implement first: **Decompose Pauli-evolution circuits before statevector simulation** (Strategy A)

**Why:**

1. Profiling proves the cost is **dense `PauliEvolutionGate.to_matrix`**, not
   applying unitaries and not Atlas’s wrapper boilerplate.
2. Micro-benchmarks show **~300×** evolver speedup after `decompose()` with
   state agreement at numerical noise.
3. Engineering effort is minimal and fits `execution/` without changing
   product-formula math or experiment APIs.
4. Remains compatible with arbitrary Hamiltonians expressed as Pauli sums and
   with future evolution methods that still emit circuits.
5. Keeps Trotter semantics intact (unlike replacing simulation with full-H
   `expm`).

**Suggested shape (for a future PR, not now):**

- Add an evolver variant or a flag, e.g. decompose-on-evolve defaulting on for
  local QI simulation.
- Keep `EvolutionMethod` output unchanged.
- Validate fidelity / state equality on small TFIM fixtures against today’s path.

**Second step after A:** either sparse/direct Pauli apply (C) for scaling, or
object reuse (E) for sweep hygiene — depending on whether the next goal is
“larger `n`” or “cleaner experiment loops.”

---

## 10. Bottom Line

| Question | Answer |
|----------|--------|
| What dominates runtime? | `PauliEvolutionGate.to_matrix()` dense expm per Trotter term gate |
| Is Atlas’s evolver “wrong”? | Too thin / too naive a Qiskit path, not a heavy Atlas algorithm bug |
| Is Qiskit “broken”? | General QI path is correct but inefficient for this circuit shape |
| Best first fix? | Decompose (synthesize to elementary gates) before `Statevector` |
| What not to do first? | Estimators; replacing Trotter with exact H evolution; large caches of dense `2ⁿ×2ⁿ` unitaries at high `n` |

---

*End of design analysis. No code was modified for this document’s recommendations.*
