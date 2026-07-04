import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from physics.tfim_phy import get_tfim_hamiltonian

def run_vqe_sim(J, h, num_starts=10):
    """Runs a multi-start VQE loop to find the ground state."""
    estimator = StatevectorEstimator()
    
    # Import the exact same physical system
    hamiltonian = get_tfim_hamiltonian(J, h)
    
    # Build the 2-qubit Hardware-Efficient Ansatz
    params = ParameterVector('θ', 4)
    ansatz = QuantumCircuit(2)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cx(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)
    
    def cost_function(param_values):
        pub = (ansatz, hamiltonian, param_values)
        return float(estimator.run([pub]).result()[0].data.evs)

    global_best_energy = float('inf')
    global_best_thetas = None
    global_best_iterations = 0 
    
    for _ in range(num_starts):
        random_guess = np.random.uniform(0, 2 * np.pi, size=4)
        result = minimize(cost_function, random_guess, method='COBYLA', options={'maxiter': 200})
        if result.fun < global_best_energy:
            global_best_energy = result.fun
            global_best_thetas = result.x
            global_best_iterations = result.nfev
            
    return global_best_energy, global_best_thetas, global_best_iterations

def run_vqe_hardware(J, h, thetas):
    """New: A single-shot execution on hardware."""
    # This calls your hardware_run.py logic
    from tfim_hardware_run import get_hardware_energy
    return get_hardware_energy(J, h, thetas)
        