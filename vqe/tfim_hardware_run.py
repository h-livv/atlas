from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2 as Estimator,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from physics.tfim_phy import get_tfim_hamiltonian


def get_hardware_observables(backend, J, h, optimal_thetas):

    print(f"Executing on: {backend.name}")

    # -------------------------------------------------------
    # Build Hamiltonian + observables
    # -------------------------------------------------------

    hamiltonian = get_tfim_hamiltonian(J, h)

    zz_op = SparsePauliOp.from_list([
        ("ZZ", 1.0)
    ])

    x_op = SparsePauliOp.from_list([
        ("XI", 1.0),
        ("IX", 1.0)
    ])

    # -------------------------------------------------------
    # Build ansatz
    # -------------------------------------------------------

    params = ParameterVector("θ", 4)

    ansatz = QuantumCircuit(2)

    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)

    ansatz.cx(0, 1)

    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)

    bound_circuit = ansatz.assign_parameters(optimal_thetas)

    # -------------------------------------------------------
    # ISA transpilation
    # -------------------------------------------------------

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3
    )

    isa_circuit = pm.run(bound_circuit)

    isa_hamiltonian = hamiltonian.apply_layout(
        isa_circuit.layout
    )

    isa_zz = zz_op.apply_layout(
        isa_circuit.layout
    )

    isa_x = x_op.apply_layout(
        isa_circuit.layout
    )

    # -------------------------------------------------------
    # Runtime Estimator
    # -------------------------------------------------------

    estimator = Estimator(mode=backend)

    estimator.options.resilience_level = 1

    pub = (
    isa_circuit,
    [
        isa_hamiltonian,
        isa_zz,
        isa_x,
    ]
    )

    job = estimator.run([pub])

    print(f"Job ID: {job.job_id()}")

    result = job.result()

    evs = result[0].data.evs

    energy = float(evs[0])
    zz = float(evs[1])
    transverse_mag = float(evs[2])

    print(
        f"Energy = {energy:.6f}, "
        f"<ZZ> = {zz:.6f}, "
        f"<X0+X1> = {transverse_mag:.6f}"
    )

    return {
    "backend": backend.name,
    "energy": energy,
    "zz": zz,
    "x": transverse_mag,
    "job_id": job.job_id(),
}