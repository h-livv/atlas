import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector, state_fidelity, SparsePauliOp
from physics.tfim_phy import calculate_exact_solution
from optimizer.tfim_optimizer import run_vqe_sim
from tfim_hardware_run import get_hardware_observables
import pandas as pd
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2 as Estimator,
)

'''service = QiskitRuntimeService()
backend = service.least_busy(
        operational=True,
        simulator=False,    
        min_num_qubits=2
    )'''

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    print("Initializing TFIM Physics Benchmark...\n")
    
    # Define the save directory
    load_dir = os.path.join('vqe')
    save_dir = os.path.join('vqe', 'data')
    ensure_dir(save_dir)
    
    # Define the parameter sweep
    J_val = 1.0
    # Sweep h from 0.05 to 2.0 (avoiding pure 0 to prevent trivial math states)
    # h_values = np.linspace(0.05, 2.0, 20) 
    hardware_df = pd.read_csv(os.path.join(load_dir, 'hardware_results_20_pts.csv'))
    hardware_h_vals = hardware_df["h"].to_numpy()
    h_values = hardware_df["h"].to_numpy()
    
    # Data storage arrays
    exact_energies = []
    vqe_energies = []
    magnetizations = []
    transverse_magnetizations = []
    fidelities = []
    iterations = []
    hardware_energies = []
    hardware_zz = []
    hardware_x = []
    #hardware_h_vals = []
    optimal_thetas = {}
    hardware_backends = []
    hardware_job_ids = []
    
    # Define the magnetization observable (Spin-Spin Correlation <ZZ>)
    zz_op = SparsePauliOp.from_list([("ZZ", 1.0)])
    x_op = SparsePauliOp.from_list([("XI", 1.0), ("IX", 1.0)])
    
    # Rebuild the Ansatz locally to construct the final statevector
    params = ParameterVector('θ', 4)
    ansatz = QuantumCircuit(2)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cx(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)
    
    print(f"{'h/J Ratio':<10} | {'Delta E':<12} | {'Fidelity':<10} | {'Iters'}")
    print("-" * 50)
    
    for h in h_values:
        # 1. Get exact math truth
        exact_E, exact_state = calculate_exact_solution(J_val, h)
        exact_energies.append(exact_E)
        
        # 2. Run VQE (Using 5 starts to balance speed and accuracy for a dense graph)
        vqe_E, opt_thetas, iters = run_vqe_sim(J_val, h, num_starts=5)
        vqe_energies.append(vqe_E)
        iterations.append(iters)
        optimal_thetas[h] = opt_thetas
        
        # 3. Construct the VQE Quantum State
        vqe_circuit = ansatz.assign_parameters(opt_thetas)
        vqe_state = Statevector(vqe_circuit)
        
        # 4. Calculate Fidelity (How perfectly do the vectors align?)
        fid = state_fidelity(vqe_state, exact_state)
        fidelities.append(fid)
        
        # 5. Calculate Magnetization <ZZ>
        mag = vqe_state.expectation_value(zz_op).real
        magnetizations.append(mag)

        # 6. Calculate Transverse Magnetization <X1 + X2>
        transverse_mag = vqe_state.expectation_value(x_op).real
        transverse_magnetizations.append(transverse_mag)
        
        print(f"{h/J_val:<10.3f} | {abs(exact_E - vqe_E):<12.2e} | {fid:<10.4f} | {iters}")


    '''print("\nStarting Hardware Validation (20 points)...")
    for h in h_values:

        theta = optimal_thetas[h]

        results = get_hardware_observables(
            backend,
            J_val,
            h,
            theta
        )

        hardware_h_vals.append(h)

        hardware_energies.append(results["energy"])

        hardware_zz.append(results["zz"])

        hardware_x.append(results["x"])

        hardware_backends.append(results["backend"])
        hardware_job_ids.append(results["job_id"])'''


    hardware_energies = hardware_df["energy"].to_numpy()

    hardware_zz = hardware_df["zz"].to_numpy()

    hardware_x = hardware_df["x"].to_numpy()

    energy_stderr = hardware_df["energy_stderr"].to_numpy()

    zz_stderr = hardware_df["zz_stderr"].to_numpy()

    x_stderr = hardware_df["x_stderr"].to_numpy()

    print()

    print("Hardware Validation")

    print("-----------------------------------------------")

    for h,E_exact,E_sim,E_hw in zip(
        hardware_h_vals,
        exact_energies,
        vqe_energies,
        hardware_energies
    ):

        abs_err = abs(E_hw-E_exact)

        rel_err = abs_err/abs(E_exact)

        print(
            f"h={h:.2f} | "
            f"Exact={E_exact:.4f} | "
            f"Sim={E_sim:.4f} | "
            f"HW={E_hw:.4f} | "
            f"Abs={abs_err:.4f} | "
            f"Rel={100*rel_err:.2f}%"
        )



    print("\nSimulation and hardware run complete. Generating and saving plots...")

    # ---------------------------------------------------------
    # Plot 1: Ground State Energy
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(h_values, exact_energies, 'k--', label='Exact Analytical', linewidth=2)
    plt.plot(h_values, vqe_energies, 'ro', label='VQE Approximation', alpha=0.7)
    if hardware_energies.size > 0:
        plt.errorbar(
    hardware_h_vals,
    hardware_energies,
    yerr=energy_stderr,
    fmt='kx',
    markersize=10,
    capsize=4,
    label="IBM Hardware"
)
    plt.title('Energy vs Transverse Field (h/J)')
    plt.xlabel('h/J Ratio (Transverse Field Strength)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'energy_vs_hJ.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Combined Magnetization Comparison Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # Plot both datasets on the same axes
    plt.plot(h_values, magnetizations, 'b-s', label=r'$\langle ZZ \rangle$ (Longitudinal)', linewidth=2)
    plt.plot(h_values, transverse_magnetizations, 'c-o', label=r'$\langle X_0+X_1 \rangle$ (Transverse)', linewidth=2)
    plt.errorbar(
    hardware_h_vals,
    hardware_zz,
    yerr=zz_stderr,
    fmt='ks',
    markersize=8,
    capsize=4,
    label="Hardware <ZZ>"
)

    plt.errorbar(
    hardware_h_vals,
    hardware_x,
    yerr=x_stderr,
    fmt='k^',
    markersize=8,
    capsize=4,
    label="Hardware <X0+X1>"
)
    plt.title('Magnetization Phase Competition vs h/J')
    plt.xlabel('h/J Ratio (Transverse Field Strength)')
    plt.ylabel('Expectation Value')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5) # Reference line at 0
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(os.path.join(save_dir, 'magnetization_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Hardware error plot

    hardware_error = np.array(hardware_energies) - np.array(exact_energies)
    sim_error = np.abs(
    np.array(vqe_energies)
    -
    np.array(exact_energies)
)
    hardware_abs_error = np.abs(hardware_error)

    plt.figure(figsize=(8,5))

    eps = 1e-12

    plt.semilogy(
        h_values,
        sim_error+eps,
        'bo-', 
        label='Simulator'
    )

    plt.semilogy(
        hardware_h_vals,
        np.abs(hardware_abs_error)+eps,
        'ro-',
        label='Hardware'
    )

    plt.legend()

    plt.xlabel("h/J")

    plt.ylabel("Absolute Energy Error")

    plt.title("Energy Error Comparison")

    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'sim_vs_hardware_error.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Simulator vs Hardware parity plot

    plt.figure(figsize=(6,6))

    plt.scatter(
    vqe_energies,
    hardware_energies,
    c=hardware_h_vals,
    cmap="viridis",
    s=80
)

    plt.colorbar(label="h/J")

    all_energy = np.concatenate([
    vqe_energies,
    hardware_energies
    ])

    lims = [
        np.min(all_energy),
        np.max(all_energy)
    ]

    plt.plot(
        lims,
        lims,
        'k--'
    )

    plt.axis('equal')
    plt.xlim(lims)
    plt.ylim(lims)

    plt.xlabel("Simulator Energy")

    plt.ylabel("Hardware Energy")

    plt.title("Simulator vs Hardware")

    plt.savefig(os.path.join(save_dir, 'sim_vs_hardware_parity.png'), dpi=300, bbox_inches='tight')

    plt.close()

    # Relative error

    relative_error = np.abs(
    np.array(hardware_energies) -
    np.array(exact_energies)
    ) / np.abs(exact_energies)

    relative_error *= 100

    plt.figure(figsize=(8, 5))
    plt.plot(
    hardware_h_vals,
    relative_error,
    'ro-'
    )
    plt.xlabel("h/J")
    plt.ylabel("Relative Error (%)")
    plt.title("Relative Energy Error (Hardware vs Exact)")
    plt.grid(True)
    plt.savefig(
    os.path.join(save_dir, "relative_error.png"),
    dpi=300,
    bbox_inches="tight"
)


    plt.close()



    # ---------------------------------------------------------
    # Plot 3: State Fidelity
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(h_values, fidelities, 'g-', marker='^')
    plt.title('VQE State Fidelity vs Exact Ground State')
    plt.xlabel('h/J Ratio (Transverse Field Strength)')
    plt.ylabel('Fidelity (0 to 1)')
    plt.ylim([0.8, 1.05]) # Zoom in on the top to see small deviations
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'fidelity_vs_hJ.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 4. Optimizer Iterations (Mean and Std Dev)
    # ---------------------------------------------------------
    # We need to run the VQE multiple times for each 'h' to get stats
    '''num_trials = 5 
    mean_iters = []
    std_iters = []
    

    for h in h_values:
        trial_iters = []
        for _ in range(num_trials):
            # We only care about the iteration count here
            _, _, iters = run_vqe_sim(J_val, h, num_starts=5) 
            trial_iters.append(iters)
            print(f"Current h: {h}, Iterations: {trial_iters}")
        
        mean_iters.append(np.mean(trial_iters))
        std_iters.append(np.std(trial_iters))

    plt.figure(figsize=(8, 5))
    plt.errorbar(h_values, mean_iters, yerr=std_iters, fmt='o-', 
                color='m', ecolor='gray', capsize=5, label='Mean NFEV')
    plt.title('Mean Optimizer Iterations vs h/J (with Error Bars)')
    plt.xlabel('h/J Ratio')
    plt.ylabel('Function Evaluations (NFEV)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'iterations_vs_hJ_errorbars.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()'''

    df = pd.DataFrame({
    "h": h_values,
    "exact_energy": exact_energies,
    "vqe_energy": vqe_energies,
    "hardware_energy": hardware_energies,

    "sim_zz": magnetizations,
    "hardware_zz": hardware_zz,

    "sim_x": transverse_magnetizations,
    "hardware_x": hardware_x,

    "fidelity": fidelities,

    "abs_error": hardware_abs_error,
    "relative_error_percent": relative_error,

    "energy_stderr": energy_stderr,
    "zz_stderr": zz_stderr,
    "x_stderr": x_stderr

    })

    
    '''"backend": hardware_backends,
    "job_id": hardware_job_ids'''

    csv_path = os.path.join(save_dir, "tfim_hardware_benchmark.csv")

    df.to_csv(csv_path, index=False)

    print(f"Benchmark data saved to {csv_path}")

    print(f"All plots and CSV saved successfully in: {save_dir}")

    print()

    print("Benchmark Summary")

    print("----------------")

    print(
        f"Mean hardware abs error: "
        f"{np.mean(hardware_abs_error):.5f}"
    )

    print(
        f"Max hardware abs error: "
        f"{np.max(hardware_abs_error):.5f}"
    )

    print(
        f"Mean relative error: "
        f"{np.mean(relative_error):.2f}%"
    )

    print(
        f"Max relative error: "
        f"{np.max(relative_error):.2f}%"
    )

    