import numpy as np
from physics.tfim_phy import calculate_exact_solution 
from optimizer.tfim_optimizer import run_vqe

if __name__ == "__main__":
    # ---------------------------------------------------------
    # System Parameters
    # ---------------------------------------------------------
    J_val = 1.0  # Coupling constant
    h_val = 0.5  # Transverse field
    starts = 100 # Multi-start iterations
    
    print("==================================================")
    print(f" 2-Qubit TFIM Simulator (Single-Shot) ")
    print(f" Parameters: J = {J_val}, h = {h_val}")
    print("==================================================")
    
    print(f"Running Exact Classical Diagonalization...")
    # UPDATE 2: Unpack the exact statevector so exact_E remains a pure float
    exact_E, exact_state = calculate_exact_solution(J_val, h_val)
    
    print(f"Running Quantum VQE (Multi-Start: {starts})...")
    # UPDATE 3: Unpack the new iteration count
    vqe_E, optimal_thetas, iters = run_vqe(J_val, h_val, num_starts=starts)
    
    # Calculate absolute error
    error = abs(exact_E - vqe_E)
    
    # ---------------------------------------------------------
    # Final Output
    # ---------------------------------------------------------
    print("\n--- Final Results ---")
    print(f"Analytical Ground State: {exact_E:.6f}")
    print(f"VQE Computed Energy:     {vqe_E:.6f}")
    print(f"Absolute Error:          {error:.6e}")
    print(f"Optimizer Iterations:    {iters}")
    print(f"Optimal Thetas (rads):   {np.round(optimal_thetas, 4)}")