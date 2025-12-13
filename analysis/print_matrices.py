import numpy as np
import os
import sys

# Monkeypatch for numpy 2.0 -> 1.x compatibility if needed
try:
    import numpy.core 
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = numpy.core
except ImportError:
    pass

# Path to the data file
data_path = 'characterization_results.npy'

if not os.path.exists(data_path):
    # Try finding it in the same directory as script if run from elsewhere
    data_path = os.path.join(os.path.dirname(__file__), 'characterization_results.npy')

if not os.path.exists(data_path):
    print(f"Error: Could not find {data_path}")
    exit(1)

data = np.load(data_path, allow_pickle=True).item()

M = data['M']
C = data['C']

def complex_to_latex(z):
    # Format: |z| e^{j phi}
    mag = abs(z)
    phase = np.angle(z) # Radians
    
    # Optional: Normalize phase to (-pi, pi] or (0, 2pi]
    # np.angle does (-pi, pi] by default.
    
    return f"{mag:.2f} e^{{j{phase:.2f}}}"

print("--- Matrix M (Magnitude * exp(j Phase)) ---")
print(r"M = \begin{bmatrix}")
for row in M:
    line = " & ".join([complex_to_latex(x) for x in row])
    print(line + r" \\")
print(r"\end{bmatrix}")

print("\n--- Matrix C (Magnitude * exp(j Phase)) ---")
print(r"C = \begin{bmatrix}")
for row in C:
    line = " & ".join([complex_to_latex(x) for x in row])
    print(line + r" \\")
print(r"\end{bmatrix}")

if 'alpha_max' in data:
    alpha_max = data['alpha_max'] # (4,) vector
    # E is effectively diag(alpha_max) for the fully ON state
    E = np.diag(alpha_max)
    print("\n--- Matrix E_max (Diagonal of alpha_max) ---")
    print(r"E_{max} = \begin{bmatrix}")
    for row in E:
        line = " & ".join([complex_to_latex(x) for x in row])
        print(line + r" \\")
    print(r"\end{bmatrix}")

if 'I_ampl' in data:
    print(f"\n--- Input Amplitude I ---")
    print(f"I = {data['I_ampl']:.4f}")
