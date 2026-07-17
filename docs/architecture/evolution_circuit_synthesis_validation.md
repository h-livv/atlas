# Pre-Synthesize Evolution Circuits: Design & Validation Report

> **Status:** Investigation complete — **no implementation**.  
> **Date:** 2026-07-17  
> **Question:** Is decomposing/synthesizing `PauliEvolution` circuits before
> `Statevector` simulation the correct first optimization for Atlas?  
> **Verdict:** **Yes.** Proceed as an optional evolver-side preprocess with
> **default on** for the local statevector evolver.

---

## 1. Does decomposition remove the bottleneck?

### Current path

```text
Hamiltonian
  → EvolutionMethod.build_circuit()
  → QuantumCircuit[PauliEvolutionGate × O(steps × terms)]
  → Statevector(circuit)
  → for each gate: PauliEvolutionGate.to_matrix()
        → scipy.sparse.linalg.expm(...) → dense 2ⁿ×2ⁿ Operator
  → apply Operator to statevector
```

Empirical counts (TFIM, Strang, 10 steps, `n=4`): **140**
`PauliEvolutionGate.to_matrix()` calls per evolve. That path is ~98% of
evolver time in prior profiles.

### Proposed path

```text
Hamiltonian
  → EvolutionMethod.build_circuit()   (unchanged: still emits PauliEvolution)
  → QuantumCircuit
  → circuit.decompose()               (Qiskit definition → {rzz, rx, …})
  → Statevector(circuit)
  → for each gate: cheap fixed-size to_matrix (2×2 / 4×4) → apply
```

### Does this avoid `PauliEvolutionGate.to_matrix()`?

**Yes — entirely.** After one `decompose()` pass on TFIM product-formula
circuits:

| Metric | Native | Decomposed |
|--------|--------|------------|
| `PauliEvolution` ops | 70–220 | **0** |
| `PauliEvolutionGate.to_matrix` calls | = gate count | **0** |
| Simulation matrix path | dense `2ⁿ×2ⁿ` expm | `rzz` (4×4), `rx` (2×2) |

Statevector still uses `Operator._instruction_to_matrix` for elementary
gates, but those matrices are tiny and closed-form — not sparse full-system
exponentials.

**Conclusion:** Decomposition removes the identified bottleneck (repeated
dense `PauliEvolutionGate.to_matrix` / `expm`).

---

## 2. Expected performance improvement

Measured wall times for `Statevector` alone (and for decompose+SV including
`decompose()` cost):

| Case | Native SV | Decomposed SV | Speedup (SV only) |
|------|-----------|---------------|-------------------|
| `n=4` Lie 10 steps | 0.23 s | 1.2 ms | ~186× |
| `n=4` Strang 10 | 0.43–0.45 s | ~2 ms | ~210× |
| `n=6` Strang 10 | 2.96 s | 5.9 ms | ~500× |
| `n=6` Lie 20 | 2.80 s | 4.6 ms | ~613× |
| `n=8` Strang 5 | 5.64 s | 3.4 ms | ~1600× |

Including `decompose()` once per time point (`n=4` Strang, 5 times):

| Path | Mean per point |
|------|----------------|
| Native `Statevector` | 0.34 s |
| `decompose()` + `Statevector` | **0.010 s** (~35×) |

`decompose()` itself is ~10 ms on the `n=4` Strang circuit — negligible vs
native SV, still the minority of the optimized path.

**Why speedup grows with `n`:** native cost is dominated by dense
`2ⁿ×2ⁿ` expm per gate; decomposed cost is O(gates × poly(2ⁿ)) with tiny
local unitaries. Peak memory also drops (no stack of full-system dense
unitaries).

**Estimate for Atlas sweeps:** evolver-dominated runs should see
**~10²–10³×** reduction in `sim.evolver_evolve` time at small-to-moderate
`n` used today; end-to-end sweep speedup will approach that as long as
exact evolution / I/O remain secondary.

---

## 3. Numerical correctness assessment

Validation compared native vs decomposed states for Lie and Strang across
times `{0.1, 1.0, 5.0}` and sizes `n ∈ {4,6,8}`.

| Check | Result |
|-------|--------|
| State fidelity ⟨ψ_n\|ψ_d⟩ | **1 − O(10⁻¹⁵)** |
| Max amplitude \|Δ\| | ≤ ~1.5×10⁻¹⁵ |
| Global observables (`zz`, `x`) | Δ ≤ ~10⁻¹⁴ |
| Site `⟨Zᵢ⟩` | max Δ ≤ ~10⁻¹⁵ |
| Fidelity vs exact (same Trotter) | identical to ~10⁻¹⁵ |
| Full unitary `‖U_n − U_d‖₂` (`n=4` Strang) | **1.3×10⁻¹⁵** |
| Process fidelity | **1 − O(10⁻¹⁴)** |

**Physics preserved:**

- Unitary evolution: yes (operator-level match).
- Trotter ordering / Strang symmetry: yes — decomposition lowers each
  `PauliEvolutionGate` independently; product order is unchanged.
- Observables, fidelity, operator error: unchanged beyond float noise.

**Caveat:** Correctness relies on Qiskit’s `PauliEvolutionGate` definition
(synthesis to Pauli rotations). Atlas should treat that as the contract and
add regression tests (state + observable deltas) when implementing — not
re-implement synthesis in physics code.

---

## 4. Architectural placement

| Location | Fit |
|----------|-----|
| **`execution/` / `StatevectorEvolver.evolve`** | **Best.** Execution concern: how to simulate a bound circuit. Algorithms stay unaware. |
| `sim.evolver_evolve` / `HamiltonianSimulation` | Possible but leaks execution policy into the algorithm. |
| `sim.circuit_build` / `circuits/evolution` | **Wrong.** Would mutate the logical product-formula representation and couple synthesis to every method. Hardware / inspection may want undecomposed circuits. |
| Optional preprocessing stage outside evolver | Acceptable if still owned by execution builders; prefer inside evolver for encapsulation. |

**Recommended shape:**

```text
EvolutionMethod.build_circuit → EvolutionCircuitSpec  (unchanged)
StatevectorEvolver.evolve(circuit):
    if synthesize_evolution:
        circuit = circuit.decompose()   # or bounded until no PauliEvolution
    return Statevector(circuit) → EvolutionResult
```

Algorithms continue to call `evolver.evolve(spec.circuit)` only.

---

## 5. Configuration recommendations

Make it **optional and evolver-scoped**, not a global physics switch.

Preferred YAML (extends existing backend parameters):

```yaml
backend:
  name: statevector_evolver
  parameters:
    synthesize_evolution: true   # default: true for this backend
```

Equivalent explicit mode (if clearer later):

```yaml
backend:
  name: statevector_evolver
  parameters:
    evolution_execution:
      mode: synthesized   # native | synthesized
```

| Choice | Rationale |
|--------|-----------|
| **Default `true` / `synthesized`** for local SV evolver | Matches Atlas goals (correct Trotter + practical performance). Native path is a debug/compatibility escape hatch. |
| Keep logical circuits undecomposed until evolve | Preserves circuit inspection, future hardware transpile, and method authorship. |
| Do **not** put this under `physics` or `algorithm` | Algorithms must not know about synthesis. |

`native` remains useful for: debugging gate-level identity with Qiskit’s
dense path, or A/B correctness tests.

---

## 6. Future compatibility

| Concern | Compatible? |
|---------|-------------|
| Lie-Trotter / Strang | Yes (validated). |
| Higher-order Suzuki | Yes — still sequences of `PauliEvolutionGate` (or equivalent) that Qiskit can define/decompose. |
| QDrift | Yes — randomized Pauli evolutions still lower to elementary rotations if built from `PauliEvolutionGate` (or similar definable gates). |
| Hamiltonian type | Yes — synthesis is gate-local; independent of TFIM vs other Pauli sums. |
| New product formulas | Compatible if they emit decomposable evolution gates; if a future method emits opaque custom gates without definitions, synthesis must be skipped or custom-handled. |

Optimization stays **method-agnostic** and **Hamiltonian-agnostic** as long as
circuits are built from gates with Qiskit definitions.

---

## 7. Risks and trade-offs

| Risk | Assessment |
|------|------------|
| Increased circuit depth (reported) | Depth *label* may drop (parallelizable 1q/2q) or change; **gate count** ≈ same term count mapped to `rzz`/`rx`. Not a correctness issue for SV. |
| Increased circuit build / preprocess time | ~10 ms decompose vs hundreds of ms–seconds of native SV — net win. |
| Memory footprint | **Lower** peak vs native (no many dense `2ⁿ×2ⁿ` mats). Decomposed circuit object is larger as instruction list of small gates — fine. |
| Future execution backends | Hardware / Aer should receive logical or backend-transpiled forms; synthesis-for-SV must stay behind the **statevector evolver**, not mutate shared circuit specs permanently. |
| Loss of flexibility | Mitigated by `synthesize_evolution: false`. |
| Interaction with hardware | Do not force SV synthesis into IBM Runtime paths; hardware wants transpile to device basis, not QI `decompose()` for SV. |
| Multi-term / non-single-Pauli evolution gates | Current Atlas emits **single-Pauli** `PauliEvolutionGate`s → clean `{rzz,rx}`. If fused multi-Pauli gates appear later, `decompose()` still applies but may need multi-pass or explicit synthesis; re-validate. |

---

## 8. Final recommendation

**Pre-synthesizing / decomposing the evolution circuit before local
statevector simulation is the correct first optimization.**

### Why preferable to alternatives

| Alternative | Why not first |
|-------------|---------------|
| **Cache dense `to_matrix` results** | Still allocates/reuses `2ⁿ×2ⁿ` unitaries; memory scales as `unique × 4ⁿ`; weaker asymptotics; more cache-keying complexity. Good *second* lever if needed. |
| **Cache Operators / circuit templates** | Helps reuse across times; does not remove per-gate dense expm on first use; orthogonal, smaller win alone. |
| **Replace execution backend (Aer, custom Pauli apply)** | Higher effort/dependency; Aer may still need elementary gates; custom apply is the *best long-term kernel* but not the smallest correct win. |
| **Replace Trotter with full-H `expm_multiply`** | Changes the algorithm under test — unacceptable as a drop-in for product-formula validation. |

Decomposition is preferable **now** because it:

1. Eliminates the measured hotspot completely (`to_matrix` count → 0).
2. Preserves Trotter/Strang numerics to float precision.
3. Lives cleanly in `execution/` with algorithms unaware.
4. Is method- and Hamiltonian-agnostic.
5. Is a one-line-class change with an optional config escape hatch.
6. Leaves room for later Aer / direct Pauli-apply evolvers without API churn.

**Next step (implementation, after approval):** add
`synthesize_evolution` (default `true`) to `StatevectorEvolver`, wire through
`BackendConfig.parameters` / builders, and add equivalence unit tests
(state fidelity + observables Lie/Strang).
