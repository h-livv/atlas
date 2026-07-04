"""IBM Runtime hardware estimator execution.

Encapsulates the hardware-specific details currently in
`vqe_legacy/tfim_hardware_run.py`: transpilation, layout application,
resilience configuration, and `EstimatorV2` job execution. This module never
rebuilds the ansatz, Hamiltonian, or observables -- it only consumes the
objects passed into `evaluate()`.

Backend selection/authentication (e.g. `QiskitRuntimeService`) is not this
module's responsibility; callers are expected to pass in an already-resolved
backend object (real or fake).
"""

from __future__ import annotations

from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from atlas.experiments.results import HardwareEvaluationResult
from atlas.physics.hamiltonians import Hamiltonian
from atlas.physics.observables import ObservableSpec
from atlas.circuits.ansatzes.hardware_efficient import AnsatzSpec


class IBMRuntimeEstimator:
    """Evaluates a bound ansatz on an IBM Runtime backend via `EstimatorV2`."""

    def __init__(self, backend, resilience_level: int = 1):
        self.backend = backend
        self.resilience_level = resilience_level

    def evaluate(
        self,
        ansatz: AnsatzSpec,
        hamiltonian: Hamiltonian,
        parameter_values,
        observables: list[ObservableSpec],
    ) -> HardwareEvaluationResult:
        """Bind, transpile, and run the ansatz against `hamiltonian` and `observables`.

        Raises `ValueError` if `self.backend` does not have enough qubits for
        `ansatz.num_qubits`, per the n-qubit generalization policy.
        """

        if self.backend.num_qubits < ansatz.num_qubits:
            raise ValueError(
                f"Backend '{self.backend.name}' has {self.backend.num_qubits} qubits, "
                f"which is fewer than the {ansatz.num_qubits} qubits required by the ansatz."
            )

        bound_circuit = ansatz.bind(parameter_values)

        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        isa_circuit = pm.run(bound_circuit)

        isa_hamiltonian = hamiltonian.operator().apply_layout(isa_circuit.layout)
        isa_observables = [
            spec.operator.apply_layout(isa_circuit.layout) for spec in observables
        ]

        estimator = Estimator(mode=self.backend)
        estimator.options.resilience_level = self.resilience_level

        pub = (isa_circuit, [isa_hamiltonian] + isa_observables)
        job = estimator.run([pub])
        result = job.result()

        evs = result[0].data.evs

        energy = float(evs[0])
        observables_dict = {
            spec.name: float(ev) for spec, ev in zip(observables, evs[1:])
        }

        return HardwareEvaluationResult(
            backend=self.backend.name,
            energy=energy,
            observables=observables_dict,
            job_id=job.job_id(),
        )
