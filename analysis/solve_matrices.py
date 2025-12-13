import numpy as np
from scipy.optimize import least_squares
import time

# Load Data
try:
    data = np.load("processed_harmonics.npy", allow_pickle=True)
except FileNotFoundError:
    print("Data file not found. Run extract_data.py first.")
    exit(1)

# Pre-process data
# N samples
# Each sample: mask (4,), scanned_idx (int), dc (4,), fund (4,) complex

# Convert to fixed arrays for speed
items = [d for d in data]
N = len(items)

scanned_indices = np.array([d['scanned_input_idx'] for d in items], dtype=int) # (N,)
active_masks = np.array([d['active_mask'] for d in items], dtype=bool) # (N, 4)

measured_dc = np.array([d['dc'] for d in items], dtype=float) # (N, 4)
measured_fund = np.array([d['fundamental'] for d in items], dtype=complex) # (N, 4)

# --- Parameter Encoding/Decoding ---
# M: 4x4 complex -> 32 params
# C: 4x4 complex -> 32 params
# I: fixed to [1,1,1,1]
# Alpha_off: 4 complex -> 8 params
# Alpha_max: 4 complex -> 8 params
# Total 80 params

# --- Parameter Encoding/Decoding ---
# M: 4x4 complex -> 32 params
# C: 4x4 complex -> 32 params
# I_ampl: 1 scalar real -> 1 param
# Alpha_off: 4 complex -> 8 params
# Alpha_max: 4 complex -> 8 params
# Total 81 params

def pack_params(M, C, I_ampl, alpha_off, alpha_max):
    return np.hstack([
        M.real.ravel(), M.imag.ravel(),
        C.real.ravel(), C.imag.ravel(),
        np.array([I_ampl]),
        alpha_off.real.ravel(), alpha_off.imag.ravel(),
        alpha_max.real.ravel(), alpha_max.imag.ravel()
    ])

def unpack_params(x):
    # Offsets
    idx = 0
    
    M_real = x[idx:idx+16].reshape(4,4); idx += 16
    M_imag = x[idx:idx+16].reshape(4,4); idx += 16
    M = M_real + 1j * M_imag
    
    C_real = x[idx:idx+16].reshape(4,4); idx += 16
    C_imag = x[idx:idx+16].reshape(4,4); idx += 16
    C = C_real + 1j * C_imag
    
    # I Ampl (scalar)
    I_ampl = x[idx]; idx += 1
    
    # Alpha Off (4,)
    a_off_real = x[idx:idx+4]; idx += 4
    a_off_imag = x[idx:idx+4]; idx += 4
    alpha_off = a_off_real + 1j * a_off_imag

    # Alpha Max (4,)
    a_max_real = x[idx:idx+4]; idx += 4
    a_max_imag = x[idx:idx+4]; idx += 4
    alpha_max = a_max_real + 1j * a_max_imag
    
    return M, C, I_ampl, alpha_off, alpha_max

# --- Initial Guess ---
# User Description:
# 1. Constructive (DFT row 0)
# 2. Double Bracewell (DFT row 2: 1, -1, 1, -1)
# 3. Quadrature (DFT row 3: 1, j, -1, -j)
# 4. Symmetric Quadrature (DFT row 1: 1, -j, -1, j)

M_physical = np.array([
    [1,  1,  1,  1],      # Uniform
    [1, -1,  1, -1],      # Alternating
    [1, 1j, -1, -1j],     # Quadrature +
    [1, -1j, -1, 1j]      # Quadrature -
], dtype=complex) * 0.5   # Normalize

M0 = M_physical.copy() 
M0 += 0.05 * (np.random.randn(4,4) + 1j*np.random.randn(4,4))

C0 = np.eye(4, dtype=complex) # approx identity
C0 += 0.01 * (np.random.randn(4,4) + 1j*np.random.randn(4,4))

# Initial Alphas per channel
# Off ~ 0, Max ~ 1
alpha_off0 = np.zeros(4, dtype=complex) + 0.1
alpha_max0 = np.ones(4, dtype=complex)

# Initial Amplitude
# Data suggests magnitudes around 3-4 per channel output, so reasonable starting point.
I_ampl0 = 20.0 

x0 = pack_params(M0, C0, I_ampl0, alpha_off0, alpha_max0)

# --- Residual Function ---
def residuals(x):
    M, C, I_ampl, alpha_off, alpha_max = unpack_params(x)
    
    # I is scalar ampl * ones
    I_vec = np.ones(4, dtype=complex) * I_ampl
    
    # Construct E values for each measurement
    # active_masks (N, 4) boolean
    # alpha_off (4,), alpha_max (4,)
    
    # We broadcast alphas to (N, 4)
    # If mask is True use max, else off
    
    # Expand alphas to (1, 4)
    a_off_exp = alpha_off[None, :]
    a_max_exp = alpha_max[None, :]
    
    # Select based on mask
    E_vals = np.where(active_masks, a_max_exp, a_off_exp) # (N, 4)
    
    # EI = E * I (elementwise)
    EI = E_vals * I_vec[None, :] # (N, 4)
    
    # u = C @ EI.T
    # (4, 4) @ (4, N) -> (4, N) -> T -> (N, 4)
    u = (C @ EI.T).T 
    
    # Calculate M contributions
    # M (4, 4)
    # Mu = u @ M.T
    Mu = u @ M.T # (N, 4)
    
    # Modulated part (scanned input k)
    # u_k = u[n, k]
    # M_k = M[:, k]
    # O_mod = u_k * M_k
    
    # Advanced indexing
    row_idx = np.arange(N)
    k_idx = scanned_indices
    u_k = u[row_idx, k_idx] # (N,)
    
    M_k = M[:, k_idx].T # (N, 4)
    
    O_mod = u_k[:, None] * M_k # (N, 1) * (N, 4) -> (N, 4)
    O_static = Mu - O_mod
    
    # Predictions
    pred_dc = np.abs(O_mod)**2 + np.abs(O_static)**2
    
    # Fundamental: conj(O_mod) * O_static
    pred_fund = np.conj(O_mod) * O_static
    
    # Residuals
    res_dc = (pred_dc - measured_dc).ravel()
    
    # Fund error
    diff_fund = pred_fund - measured_fund
    res_fund_real = diff_fund.real.ravel()
    res_fund_imag = diff_fund.imag.ravel()
    
    return np.concatenate([res_dc, res_fund_real, res_fund_imag])

# --- Optimization ---
print(f"Starting optimization with {len(x0)} parameters...")
print(f"Number of residuals: {residuals(x0).shape[0]}")

start_time = time.time()
res = least_squares(residuals, x0, verbose=2, method='lm', max_nfev=2000)
end_time = time.time()

print(f"Optimization done in {end_time - start_time:.2f}s")
print(f"Success: {res.success}, Cost: {res.cost}")

x_final = res.x
M_f, C_f, I_ampl_f, a_off_f, a_max_f = unpack_params(x_final)

# Save
results = {
    'M': M_f,
    'C': C_f,
    'I_ampl': I_ampl_f,
    'alpha_off': a_off_f,
    'alpha_max': a_max_f
}
np.save("characterization_results.npy", results)

print(f"\n--- Estimated I Amplitude: {I_ampl_f:.3f} ---")
print("--- Estimated Alphas Off ---")
print(a_off_f)
print("--- Estimated Alphas Max ---")
print(a_max_f)
