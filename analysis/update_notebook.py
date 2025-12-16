import json
import os

NOTEBOOK_PATH = "/mnt/e/PHOBos/analysis/Analysis.ipynb"

# The new source code for the run_optimization function
# Note: Indentation must match the notebook cell context if any, but since it's the whole cell content, standard python indentation is fine.
# We must split it into lines with \n at the end.
new_source_code = r"""def run_optimization():
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
        
        # Développement de |O|^2 = |O_static + e^(iphi) O_mod|^2
        # = |O_static|^2 + |O_mod|^2 + 2 Re( e^(-iphi) O_static* . O_mod )
        # Attention: la phase scan est e^(-i phi) dans nos conventions de mesure habituellement
        # mais vérifions: si T = diag(e^iphi), c'est e^iphi.
        # Le terme fondamental mesuré est le coeff devant e^(-i phi) ?? ou e^(i phi) ?
        # "fundamental" dans les données est généralement le coefficient complexe C1 de la série de Fourier.
        # I(phi) ~ DC + 2 Re( Fund * e^(-i phi) ) = DC + Fund * e^(-i phi) + Fund* * e^(i phi)
        # Notre terme croisé est : O_static* . O_mod . e^(i phi) + O_static . O_mod* . e^(-i phi)
        # Donc le coef de e^(-i phi) est O_static . O_mod*
        # Et le coef de e^(i phi) est O_static* . O_mod
        # Si la mesure "fundamental" correspond à e^(-i phi) (standard FFT numpy souvent...)
        # On va assumer Fund_pred = O_static * conj(O_mod)
        
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
    # Residuals: 60 scans * 4 outputs * 3 (DC, Re, Im) = 720. 
    # 720 >> 48 -> LM est approprié.
    res = least_squares(residuals, x0, verbose=1, method='lm', max_nfev=5000)
    A_final, I_ON_final, I_OFF_final = unpack(res.x)
    
    # --- Normalisation Spectrale (Norme = 1) ---
    norm_A = np.linalg.norm(A_final, 2)
    print(f"Norme spectrale avant normalisation : {norm_A}")
    
    scale_factor = norm_A
    A_final /= scale_factor
    I_ON_final *= scale_factor
    I_OFF_final *= scale_factor
    
    print(f"Norme spectrale après normalisation : {np.linalg.norm(A_final, 2)}")
    
    print("Optimisation terminée. Coût final :", res.cost)
    return {'A': A_final, 'I_ON': I_ON_final, 'I_OFF': I_OFF_final}
"""

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    updated = False
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if "def run_optimization():" in source and "least_squares" in source:
                print(f"Found optimization function in cell {i}. Updating...")
                
                # Verify it is the target cell roughly
                if "A_init = np.array" in source:
                    # Replace source
                    # Need to format lines with \n
                    new_lines = [line + "\n" for line in new_source_code.splitlines()]
                    # The splitlines eats \n, avoiding double \n issues if we re-add
                    # But wait, splitlines() creates a list.
                    # notebook expects a list of strings where each string ends with \n usually.
                    
                    cell['source'] = new_lines
                    updated = True
                    break
    
    if updated:
        print(f"Writing updated notebook to {NOTEBOOK_PATH}")
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
    else:
        print("Could not find the cell to update!")

if __name__ == "__main__":
    update_notebook()
