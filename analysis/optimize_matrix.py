import numpy as np
from scipy.optimize import least_squares
import time
import os
import matplotlib.pyplot as plt

# Configuration
# Parameters matching the notebook
DATA_PATH = r"../tests/generated/architecture_characterization/4-Port_MMI_Active/20251208_164250/characterization_data.npz"
PROCESSED_DATA_PATH = "processed_harmonics.npy"
PLOTS_DIR = "plots"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def run_optimization():
    print("Chargement des données...")
    try:
        data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"Fichier {PROCESSED_DATA_PATH} introuvable.")
        return None

    # Préparation des données (per-channel)
    items = [d for d in data]
    
    # (N_scans, 4) - Intensités mesurées pour chaque sortie
    measured_dc = np.array([d['dc'] for d in items])
    measured_fund = np.array([d['fundamental'] for d in items])
    
    active_masks = np.array([d['active_mask'] for d in items], dtype=bool)
    scanned_indices = np.array([d['scanned_input_idx'] for d in items], dtype=int)
    N_sub = len(items)
    
    print(f"Nombre de points de mesure : {N_sub}")
    
    # --- Fonctions du Modèle ---
    def compute_residuals(A, I_ON, I_OFF):
        # Calcul direct O = A . T . I pour garder l'info spatialle
        # On veut prédire le DC et le Fondamental pour CHAQUE port de sortie (4)
        
        # Construction des vecteurs d'entrée de base (sans phase)
        Is_ON = I_ON[None, :]
        Is_OFF = I_OFF[None, :]
        I_base = np.where(active_masks, Is_ON, Is_OFF) # (N, 4)
        
        # Dans le modèle : I_scan(phi) = I_base + (e^(iphi)-1)*I_k
        # O(phi) = A * I_scan(phi)
        # O(phi) = O_static + e^(iphi) * O_mod
        # Avec O_mod = A[:, k] * I_k (la colonne k de A multipliée par l'entrée k)
        # Et O_static = A * I_base - O_mod
        
        # 1. Calcul de O pour l'entrée de base I_base
        # A est (4,4), I_base.T est (4, N) -> O_base est (4, N)
        O_base = (A @ I_base.T).T # (N, 4)
        
        # 2. Identification de la partie modulée (liée à l'entrée k scannée)
        # Pour chaque scan n, k = scanned_indices[n]
        # On a besoin de la colonne k de A et de l'élément k de I_base
        k_indices = scanned_indices
        I_k = I_base[np.arange(N_sub), k_indices] # (N,)
        A_cols = A[:, k_indices].T # (N, 4) - Colonnes correspondantes de A
        
        O_mod = A_cols * I_k[:, None] # (N, 4) Broadcasting scale proper column
        O_static_part = O_base - O_mod # (N, 4)
        
        pred_dc = np.abs(O_static_part)**2 + np.abs(O_mod)**2 # (N, 4)
        pred_fund = O_static_part * np.conj(O_mod) # (N, 4)
        
        # Résidus (N, 4)
        diff_dc = (pred_dc - measured_dc).ravel() # (N*4,)
        diff_fund = (pred_fund - measured_fund).ravel() # (N*4,)
        
        return np.concatenate([diff_dc, diff_fund.real, diff_fund.imag])

    # --- Initialisation  ---
    # MMI Théorique 4x4 (pour guider A)
    A_init = np.array([
        [1,  1,  1,  1],
        [1, 1j, -1,-1j],
        [1, -1,  1, -1],
        [1, -1j, -1, 1j]
    ], dtype=complex) * 0.5
    
    I_ON_init = np.ones(4, dtype=float) * 4.0
    I_OFF_init = np.zeros(4, dtype=complex) + 0.1
    
    # Helpers
    def pack(A, I_ON, I_OFF):
        return np.hstack([
            A.real.ravel(), A.imag.ravel(),
            I_ON.real.ravel(), I_ON.imag.ravel(),
            I_OFF.real.ravel(), I_OFF.imag.ravel()
        ])
        
    def unpack(x):
        idx = 0
        A_real = x[idx:idx+16].reshape(4,4); idx += 16
        A_imag = x[idx:idx+16].reshape(4,4); idx += 16
        A = A_real + 1j * A_imag
        
        ion_real = x[idx:idx+4]; idx += 4
        ion_imag = x[idx:idx+4]; idx += 4
        I_ON = ion_real + 1j * ion_imag
        
        ioff_real = x[idx:idx+4]; idx += 4
        ioff_imag = x[idx:idx+4]; idx += 4
        I_OFF = ioff_real + 1j * ioff_imag
        return A, I_ON, I_OFF
    
    # --- Optimisation Globale ---
    print("\n--- Optimisation Globale (A, I) / Channel-wise ---")
    x0 = pack(A_init, I_ON_init, I_OFF_init)
    
    def residuals(x):
        A, I_ON, I_OFF = unpack(x)
        return compute_residuals(A, I_ON, I_OFF)
        
    # 'lm' works well when n_residuals > n_params. 
    # Params: 32 (A) + 8 (ION) + 8 (IOFF) = 48.
    res = least_squares(residuals, x0, verbose=1, method='lm', max_nfev=5000)
    A_final, I_ON_final, I_OFF_final = unpack(res.x)

    # --- Normalisation Spectrale (Norme = 1) ---
    norm_A = np.linalg.norm(A_final, 2)
    print(f"\nNorme spectrale avant normalisation : {norm_A}")
    
    scale_factor = norm_A
    A_final /= scale_factor
    I_ON_final *= scale_factor
    I_OFF_final *= scale_factor
    
    print(f"Norme spectrale après normalisation : {np.linalg.norm(A_final, 2)}")
    
    print("Optimisation terminée. Coût final :", res.cost)
    return {'A': A_final, 'I_ON': I_ON_final, 'I_OFF': I_OFF_final}

if __name__ == "__main__":
    results = run_optimization()
    if results:
        print("\nRésultats finaux (Normalisés):")
        print("A:\n", results['A'])
        print("I_ON:\n", results['I_ON'])
        print("I_OFF:\n", results['I_OFF'])
