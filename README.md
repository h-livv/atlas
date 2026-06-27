# Atlas

A modular framework for quantum simulation, providing tools for variational eigensolvers, quantum dynamics, hardware execution, and benchmarking.

## Features

- Variational Quantum Eigensolver (VQE)
- Exact diagonalization benchmarks
- IBM Quantum hardware execution
- Quality visualizations

## Variational Quantum Eigensolver

- VQE was implemented on a 2-qubit Transverse Field Ising Model
- The algorithm was run for several values of magentization ratios
- The results were benchmarked against analytical diagonalization and actual hardware execution

<img width="800" height="500" alt="energy_vs_hJ" src="https://github.com/user-attachments/assets/e16cde82-4e27-4bb2-be78-d92840b6ca82" />

## Roadmap

- [x] VQE
- [x] IBM Quantum Runtime
- [x] TFIM benchmarking
- [ ] Variational Quantum Deflation (VQD)
- [ ] Qubit count expansion
- [ ] Quantum dynamics (Trotterization)
- [ ] Error mitigation
- [ ] Additional Hamiltonians
