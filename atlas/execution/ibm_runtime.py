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
    """Evaluates a bound ansatz on an IBM Runtime backend via `EstimatorV2`.

    Responsibility:
        Take optimized variational parameters, transpile to the device ISA,
        apply the resulting layout to operators, and submit one Estimator job
        that measures energy plus named observables.

    State:
        backend: Already-resolved IBM (or fake) backend object.
        resilience_level: Forwarded to Estimator options each evaluation.

    Usage:
        Constructed inside ``TFIMExperiment`` when ``mode == \"hardware\"``.
        Not used on the dynamics path today.
    """

    def __init__(self, backend, resilience_level: int = 1):
        """Store the target backend and resilience setting.

        Inputs:
            backend: Qiskit backend with ``.num_qubits`` and ``.name``.
            resilience_level: EstimatorV2 resilience option (error mitigation).
        """

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

        Purpose:
            Obtain hardware estimates of energy and analysis observables for
            already-optimized variational parameters.

        Inputs:
            ansatz: Parameterized circuit template.
            hamiltonian: Model whose ``operator()`` is measured as energy.
            parameter_values: Optimal (or candidate) ansatz parameters.
            observables: Named operators measured in the same pub after energy.

        Process:
            1. Reject backends with too few qubits.
            2. Bind parameters to produce a concrete circuit.
            3. Transpile with preset pass manager at optimization level 3.
            4. Apply the ISA layout to H and each observable.
            5. Run EstimatorV2 with resilience configured; energy is pub index 0.
            6. Map remaining expectations onto observable names.

        Outputs:
            ``HardwareEvaluationResult`` with backend name, energy, observable
            dict, and job id.

        Side effects:
            Submits a Runtime job (network I/O, queue time, QPU usage).

        Raises:
            ValueError: If ``backend.num_qubits < ansatz.num_qubits``.
        """

        if self.backend.num_qubits < ansatz.num_qubits:
            raise ValueError(
                f"Backend '{self.backend.name}' has {self.backend.num_qubits} qubits, "
                f"which is fewer than the {ansatz.num_qubits} qubits required by the ansatz."
            )

        bound_circuit = ansatz.bind(parameter_values)

        # Map the abstract circuit onto the device's native gate set and coupling map.
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        isa_circuit = pm.run(bound_circuit)

        # Operators must follow the same qubit layout as the transpiled circuit.
        isa_hamiltonian = hamiltonian.operator().apply_layout(isa_circuit.layout)
        isa_observables = [
            spec.operator.apply_layout(isa_circuit.layout) for spec in observables
        ]

        estimator = Estimator(mode=self.backend)
        estimator.options.resilience_level = self.resilience_level

        # Single pub: energy first, then analysis observables in list order.
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
