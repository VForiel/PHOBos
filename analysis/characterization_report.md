# Characterization Feasibility Report: 4-Port MMI

## 1. Mathematical Model

The system is modeled as:
$$ O = M \cdot T \cdot C \cdot E \cdot I $$

Where:
*   **$O$**: Output vector $(4,)$. We measure intensities $|O_i|^2$.
*   **$I$**: Input vector $(4,)$. Unknown.
*   **$E$**: Input switching matrix (Diagonal $4 \times 4$).
    *   $E_{ii} \in \{ \alpha_{\text{max}}, \alpha_{\text{off}} \}$
    *   Used to select active inputs.
    *   **Note**: Index inversion: Data "Input $k$" $\rightarrow$ Physical $E$ blocks component $5-k$ (or similar map, assuming 1-based indexing $1 \leftrightarrow 4, 2 \leftrightarrow 3$).
*   **$C$**: Crosstalk matrix $(4 \times 4)$. Unknown. Ideally identity. Assumed Unitary (lossless approx).
*   **$T$**: Phase shifter matrix (Diagonal $4 \times 4$). Known/Controlled.
    *   $T_{kk} = e^{i \phi_k}$
*   **$M$**: Recombination matrix (MMI) $(4 \times 4)$. Unknown. Assumed Unitary.

## 2. Data Structure

The available data consists of phase scans. For a given configuration of active inputs (defined by $E$):
1.  One shifter $k$ is scanned: $\phi_k \in [0, 2\pi]$. Other phases are 0.
2.  Intensity $|O|^2$ is recorded for all 4 outputs.

This yields curves of the form:
$$ I_{out}(\phi_k) = A + B \cos(\phi_k + \psi) $$
or more generally, interference patterns.

The dataset includes:
*   **n1**: Single input active (3 inputs blocked).
*   **n2**, **n3**, **n4**: Combinations of active inputs.

## 3. Feasibility Analysis

### 3.1. Single Phase Scan Analysis

Let $u = C \cdot E \cdot I$ satisfy the effective input to the $T$-M stage.
Since $T$ is diagonal, $w = T u$ has elements $w_j = u_j$ for $j \neq k$, and $w_k = u_k e^{i \phi_k}$ (if scanning shifter $k$).

The output is $O = M w$.
$$ O_m = \sum_j M_{mj} w_j = M_{mk} u_k e^{i \phi_k} + \sum_{j \neq k} M_{mj} u_j $$
Let $a_{mk} = M_{mk} u_k$ and $b_{mk} = \sum_{j \neq k} M_{mj} u_j$.
$$ O_m(\phi_k) = a_{mk} e^{i \phi_k} + b_{mk} $$

The intensity is:
$$ |O_m|^2 = |a_{mk}|^2 + |b_{mk}|^2 + 2 |a_{mk} b_{mk}| \cos(\phi_k + \arg(a_{mk}) - \arg(b_{mk})) $$

From the Fourier transform of the measured intensity $|O_m|^2$ vs $\phi_k$, we can extract:
*   DC component: $|a_{mk}|^2 + |b_{mk}|^2$
*   First Harmonic: $X_{mk} = a_{mk} b_{mk}^*$ (complex value).

### 3.2. Solving Strategy

We have $X_{mk} = (M_{mk} u_k) (\sum_{j \neq k} M_{mj} u_j)^*$ for each output $m$ and each scanned shifter $k$.

#### Step 1: Characterize $M$ magnitudes (approx)
Using **n1** data (single input active):
Ideally, if crosstalk $C \approx Identity$ and $E$ is perfect, only one $u_j$ is non-zero.
However, $C$ mixes inputs, so all $u_j$ are non-zero but one is dominant.
Scanning all shifters for all 'n1' configs gives us a rich set of constraints.

If we assume $M$ is unitary, $\sum_m |M_{mj}|^2 = 1$.
The DC components and modulation depths gives info on $|M_{mj}|$.

#### Step 2: Use n1 data to find $M$ rows/cols up to phases
For a specific input configuration (fixed $E, I$), $u$ is fixed.
Scanning shifter $k$ gives harmonics $X_{mk} \propto M_{mk} u_k (\dots)^*$.
Scanning shifter $p$ gives $X_{mp} \propto M_{mp} u_p (\dots)^*$.
Ratios of these harmonics across different outputs $m$ eliminate the $(\dots)$ terms.
$$ \frac{X_{mk}}{X_{nk}} = \frac{M_{mk} b_{mk}^*}{M_{nk} b_{nk}^*} $$
This is hard because $b$ depends on $m$.

Alternative:
Look at $O_m = \sum_j M_{mj} u_j$.
Ideally, solving for $M$ and $u$ generally is an Iterative Least Squares or Gradient Descent problem.
Since we have data for many $u$ (different $E$) and many modulations ($T$), the system is overconstrained if we assume conservation of energy (Unitary matrices).

**Unknowns**:
*   $M$: 16 complex parameters (unitarty $\approx$ 16 real params for $U(4)$ minus global phase).
*   $C$: 16 complex parameters (unitary).
*   $E \cdot I$: For each config, essentially a vector.

**Constraints**:
Each scan gives 3 real numbers per output channel per shifter per configuration (DC, Amp, Phase).
$4 \text{ outputs} \times 4 \text{ shifters} \times 16 \text{ configs} = 256$ independent scans.
Information is abundant.

### 3.3. Algorithm Proposal

I propose a numerical optimization approach (Gradient Descent) rather than pure analytical inversion, due to the non-linear coupling and "off" leakage.

**Loss Function**:
$$ L(M, C, I, \alpha_{max}, \alpha_{off}) = \sum_{configs, shifters, steps} || |M T(\phi) C E_{cfg} I|^2 - \text{Data}_{intensity} ||^2 $$

**Initialization**:
*   $M_{init} = $ DFT matrix or identity (depending on device geometry).
*   $C_{init} = I$.
*   $I_{init} = [1, 1, 1, 1]$.
*   $\alpha_{max} = 1, \alpha_{off} = 0$.

**Refinement**:
1.  Extract Fourier coefficients from data (DC, 1st harmonic) to reduce data size and noise.
2.  Optimize cost function against these coefficients analytically derived from model.
    *   Model Harmonic $H_1 = (M u)_k \cdot (M u)_{rest}^*$.
3.  Optimize using PyTorch or SciPy `least_squares`.

### 4. Interpretation of 'max' and 'off'
By comparing **n1** (single input) vs **n4** (all active), we can calculate the effective extinction ratio.
If $C$ is significant, 'off' inputs will still couple into other channels. The solver will naturally find the best $\alpha_{off}$ that explains the background interference.

## 5. Conclusion
**Feasibility**: **High**.
The dataset is comprehensive ($n=1$ to $4$ with full phase scans).
Possibility to separate $M$ and $C$ relies on the variation of $E$. Since $E$ changes the vector $u$ entering $C^{-1} T C$ (wait, order is $M T C E I$), actually $u=CEI$ enters $T$.
Since $T$ is between $M$ and $C$, $M$ and $C$ are separable.
*   $C$ mixes $E I$ before phase encoding.
*   $M$ mixes phase encoded signals.

If $C$ were after $T$, it would be degenerate with $M$. But it's before.
Therefore, unique determination is possible (up to diagonal phases).

## 6. Recommended Next Steps
1.  **Pre-processing**: Load all `npz` data, perform FFT on axis `phase_steps` to extract DC and Fundamental Harmonic ($k=1$) for each curve.
2.  **Numerical Solve**: Implement the model in PyTorch/JAX.
    *   Variables: $M$ (unitary param), $C$ (unitary param), $I$, $\alpha$.
    *   Target: Match Extracted Harmonics.
3.  **Validation**: Reconstruct signals and compare with `n4` data.
