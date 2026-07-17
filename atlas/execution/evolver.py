"""Exact local simulation of parameter-free evolution circuits.

Distinct from estimators: an evolver runs a *complete* circuit once and
returns the final statevector. That matches Hamiltonian simulation, where
the product-formula circuit is already fully bound (no variational θ).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.quantum_info.operators.operator import Operator as OperatorClass

from atlas import profiling as profile


@dataclass
class EvolutionResult:
    """Final amplitudes after simulating an evolution circuit.

    Responsibility:
        Carry the dense state produced by ``StatevectorEvolver`` so algorithms
        can package a ``SimulationResult`` without depending on Qiskit types.

    State:
        statevector: Complex amplitude array of length ``2 ** num_qubits``.
        num_qubits: Register size taken from the simulated circuit.

    Usage:
        Returned by ``StatevectorEvolver.evolve``; consumed immediately by
        ``HamiltonianSimulation.run``.
    """

    statevector: np.ndarray
    num_qubits: int


def _profile_statevector_simulation_path(circuit: QuantumCircuit) -> None:
    """Time the internal stages of ``Statevector(circuit)`` without replacing it.

    Mirrors Qiskit's ``Statevector.from_instruction`` path for diagnostics:
    ``to_instruction`` → ``|0…0⟩`` → per-op ``to_matrix`` / ``_evolve_operator``.
    Results are discarded; this does not affect the evolver return value.
    """

    with profile.span("evolver.diag.circuit_inspect"):
        _ = circuit.num_qubits
        _ = circuit.size()
        _ = circuit.depth()
        op_counts: dict[str, int] = defaultdict(int)
        for item in circuit.data:
            op_counts[item.operation.name] += 1

    with profile.span("evolver.diag.to_instruction"):
        instruction = circuit.to_instruction()

    with profile.span("evolver.diag.init_zero_statevector"):
        init = np.zeros(2**instruction.num_qubits, dtype=complex)
        init[0] = 1.0
        vec = Statevector(init, dims=instruction.num_qubits * (2,))

    # Walk the circuit definition exactly as ``_evolve_instruction`` does for
    # a top-level circuit instruction.
    definition = instruction.definition
    qubits = {qubit: i for i, qubit in enumerate(definition.qubits)}

    n_matrix_ops = 0
    n_definition_ops = 0

    for item in definition:
        op = item.operation
        qargs = [qubits[q] for q in item.qubits]
        op_name = op.name

        with profile.span("evolver.diag.instruction_to_matrix", op=op_name):
            mat = OperatorClass._instruction_to_matrix(op)

        if mat is not None:
            n_matrix_ops += 1
            with profile.span("evolver.diag.operator_wrap", op=op_name):
                operator = Operator(mat)
            with profile.span("evolver.diag.evolve_operator", op=op_name):
                Statevector._evolve_operator(vec, operator, qargs=qargs)
        else:
            n_definition_ops += 1
            with profile.span("evolver.diag.evolve_via_definition", op=op_name):
                Statevector._evolve_instruction(vec, op, qargs=qargs)

    # Record structural facts as zero-duration meta events for the report.
    with profile.span(
        "evolver.diag.summary",
        num_qubits=circuit.num_qubits,
        circuit_size=circuit.size(),
        circuit_depth=circuit.depth(),
        op_counts=dict(op_counts),
        matrix_path_ops=n_matrix_ops,
        definition_path_ops=n_definition_ops,
    ):
        pass


class StatevectorEvolver:
    """Simulate a fully bound circuit and return its final statevector.

    Responsibility:
        Provide the dynamics execution primitive (ideal, noiseless).

    State:
        Stateless — no instance fields; safe to reuse across many runs.

    Usage:
        Built when ``backend.name == \"statevector_evolver\"``. Injected into
        ``HamiltonianSimulation``.
    """

    def evolve(self, circuit: QuantumCircuit) -> EvolutionResult:
        """Classically simulate ``circuit`` and return final amplitudes.

        Purpose:
            Evaluate a Trotterized evolution circuit exactly (as a statevector).

        Inputs:
            circuit: Parameter-free preparation + evolution circuit.

        Process:
            Construct Qiskit ``Statevector(circuit)`` and convert to ``numpy``.
            When ``ATLAS_PROFILE=1``, also time an internal diagnostic walk that
            mirrors Qiskit's simulation path (does not replace the return value).

        Outputs:
            ``EvolutionResult`` with amplitudes and qubit count.

        Side effects:
            Ideal simulation; memory scales as ``2 ** n``. No I/O.
        """

        # Stages absent from this function (occur upstream in circuit build):
        # PauliEvolutionGate construction, SparsePauliOp creation, Trotter
        # circuit composition, initial-state preparation, transpilation.
        with profile.span("evolver.absent.transpile_backend_primitives"):
            # Explicit no-op marker: no transpile, backend, or Sampler/Estimator.
            pass

        with profile.span("evolver.statevector_from_circuit"):
            # Authentic production path — numerical behavior unchanged.
            qiskit_state = Statevector(circuit)

        with profile.span("evolver.numpy_asarray"):
            statevector = np.asarray(qiskit_state)

        with profile.span("evolver.result_construct"):
            result = EvolutionResult(
                statevector=statevector,
                num_qubits=circuit.num_qubits,
            )

        if profile.PROFILER.enabled:
            # Diagnostic-only decomposition (extra work while profiling).
            with profile.span("evolver.diagnostic_decomposition"):
                _profile_statevector_simulation_path(circuit)

        return result
