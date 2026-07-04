import numpy as np
from qiskit.quantum_info import SparsePauliOp
from numpy.linalg import eigh

def get_tfim_hamiltonian(J, h):
    """Returns the 2-qubit Transverse Field Ising Model Hamiltonian."""
    return SparsePauliOp.from_list([
        ("ZZ", -J),
        ("XI", -h),
        ("IX", -h)
    ])


def calculate_exact_solution(J, h):
    """Classically finds the exact ground state energy and statevector."""
    hamiltonian = get_tfim_hamiltonian(J, h)
    
    # eigh returns both eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(hamiltonian.to_matrix())
    
    # Return the lowest energy and its corresponding statevector (column 0)
    return np.min(eigenvalues), eigenvectors[:, 0]