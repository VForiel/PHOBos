# Characterization Results Analysis

## Overview
We have successfully implemented a numerical solver to characterize the 4-Port MMI component using the provided phase scan data.
The solver optimizes the matrices $M$ (Recombination), $C$ (Crosstalk), input vector $I$, and switching parameters $\alpha_{max}, \alpha_{off}$ to match the measured interference patterns (DC and Fundamental Harmonics).

## Model Parameters
*   **M (Recombination)**: Modeled as a general $4 \times 4$ complex matrix.
*   **C (Crosstalk)**: Modeled as a general $4 \times 4$ complex matrix.
*   **Inputs**: Single complex vector $I$, modulated by diagonal $E$.
*   **Switching**: $\alpha_{off}$ and $\alpha_{max}$ determine the transmission of blocked/active channels.

## Key Findings

### 1. Switching Performance
The solver estimated the effective transmission values for "off" and "max" states:
*   **$\alpha_{off}$**: $\approx 0.87 + 0.08j$ (Magnitude $\approx 0.88$)
*   **$\alpha_{max}$**: $\approx 3.59 + 0.00j$ (Magnitude $\approx 3.59$)

**Extinction Ratio**:
$$ \text{ER} = \frac{|\alpha_{max}|^2}{|\alpha_{off}|^2} \approx \frac{12.9}{0.77} \approx 16.7 $$
The "off" state is not perfectly dark, suppressing power by a factor of ~16. This explains the significant crosstalk observed in "single input" experiments.

### 2. Matrix M (Recombination)
The reconstructed M matrix shows significant mixing. The values are presented in Magnitude/Phase format ($|z| e^{j\phi}$).

$$
M = \begin{bmatrix}
0.70 e^{j-0.65} & 0.71 e^{j0.24} & 0.71 e^{j0.67} & 0.59 e^{j0.18} \\
0.68 e^{j-0.15} & 0.69 e^{j2.29} & 0.66 e^{j-0.33} & 0.59 e^{j-2.53} \\
0.64 e^{j0.45} & 0.65 e^{j-0.18} & 0.69 e^{j-2.86} & 0.63 e^{j-1.82} \\
0.63 e^{j0.49} & 0.66 e^{j-1.81} & 0.71 e^{j-1.29} & 0.64 e^{j1.32} \\
\end{bmatrix}
$$

Inspection of the unitarity constraint ($M M^H$) suggests the component is roughly unitary but with some loss or gain imbalance, likely absorbed into the $I$ vector or due to measurement calibration.

### 3. Matrix C (Crosstalk)
The crosstalk matrix captures the coupling between channels before/after the MMI.

$$
C = \begin{bmatrix}
1.22 e^{j0.28} & 0.01 e^{j1.48} & 0.01 e^{j3.05} & 0.01 e^{j2.63} \\
0.02 e^{j0.66} & 1.15 e^{j-0.24} & 0.01 e^{j-1.34} & 0.02 e^{j2.26} \\
0.01 e^{j-2.64} & 0.03 e^{j-0.95} & 1.12 e^{j0.10} & 0.03 e^{j-0.22} \\
0.02 e^{j-3.14} & 0.01 e^{j-2.46} & 0.03 e^{j-2.87} & 1.17 e^{j-0.28} \\
\end{bmatrix}
$$

### 4. Input Parameters
The model separates the scalar input amplitude $I_{ampl}$ from the channel-specific transmission coefficients (Matrix E).

**Scalar Input Amplitude:**
$$ I_{ampl} \approx 20.03 $$

**Matrix E (Input Weights - "Max" State)**:
$$
E = \begin{bmatrix}
1.22 e^{j0.30} & 0.00 e^{j0.00} & 0.00 e^{j0.00} & 0.00 e^{j0.00} \\
0.00 e^{j0.00} & 1.18 e^{j-0.36} & 0.00 e^{j0.00} & 0.00 e^{j0.00} \\
0.00 e^{j0.00} & 0.00 e^{j0.00} & 1.18 e^{j0.36} & 0.00 e^{j0.00} \\
0.00 e^{j0.00} & 0.00 e^{j0.00} & 0.00 e^{j0.00} & 1.24 e^{j-0.41} \\
\end{bmatrix}
$$

### 5. Solver Convergence
### 5. Solver Convergence
*   **Residual Cost**: The optimizer achieved a low residual cost ($1.56 \cdot 10^5$) with the corrected input mapping.
*   **Fit Quality**: The model captures the main harmonic behavior, but residual error remains. This suggests either:
    *   The "Unitary + Lossless" assumption is too strict.
    *   Higher-order nonlinearities or non-diagonal crosstalk exist.
    *   Phase drifts during measurement (the dataset spans multiple configs).

## Deliverables
All analysis files have been organized in `E:\PHOBos\analysis\`:
*   `characterization_report.md`: Initial feasibility study.
*   `extract_data.py`: Script to process raw `.npz` files.
*   `solve_matrices.py`: Numerical optimizer (SciPy based).
*   `characterization_results.npy`: Saved matrices ($M, C, I, \alpha$).
*   `inspect_data.py`: Utility to view raw data structure.

## Recommendations
1.  **Improve "Off" State**: An ER of 16 is low for precise characterization. Improvement in the physical blocking mechanism would decouple the unknowns better.
2.  **Longer Optimization**: The solver can be run for more iterations (increase `max_nfev` in `solve_matrices.py`) to refine the solution.
3.  **Regularization**: Adding a penalty for non-unitarity of M and C might constrain the physical solution better if the current result is drifting.
