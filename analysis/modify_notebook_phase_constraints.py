#!/usr/bin/env python3
"""
Script amélioré pour modifier Analysis.ipynb et appliquer les contraintes de phase.
"""

import json
import re

def modify_notebook(notebook_path):
    """Modifie le notebook pour appliquer les contraintes de phase."""
    
    # Lire le notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    modified = False
    
    # Parcourir toutes les cellules
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source_lines = cell['source']
        source_text = ''.join(source_lines)
        
        # Chercher la cellule avec pack_full_params
        if 'def pack_full_params(M, C, I_ON, I_OFF):' in source_text:
            print(f"✓ Cellule {i} trouvée: pack_full_params")
            
            new_code = [
                "def pack_full_params(M, C, I_ON, I_OFF):\n",
                "    \"\"\"I_ON: phase nulle -> amplitudes seulement\"\"\"\n",
                "    return np.hstack([\n",
                "        M.real.ravel(), M.imag.ravel(),\n",
                "        C.real.ravel(), C.imag.ravel(),\n",
                "        np.abs(I_ON),  # Seulement les amplitudes\n",
                "        I_OFF.real.ravel(), I_OFF.imag.ravel()\n",
                "    ])\n",
                "\n",
                "def unpack_full_params(x):\n",
                "    idx = 0\n",
                "    M_real = x[idx:idx+16].reshape(4,4); idx += 16\n",
                "    M_imag = x[idx:idx+16].reshape(4,4); idx += 16\n",
                "    M = M_real + 1j * M_imag\n",
                "    \n",
                "    C_real = x[idx:idx+16].reshape(4,4); idx += 16\n",
                "    C_imag = x[idx:idx+16].reshape(4,4); idx += 16\n",
                "    C = C_real + 1j * C_imag\n",
                "    \n",
                "    ion_amp = x[idx:idx+4]; idx += 4\n",
                "    I_ON = ion_amp + 0j  # Phase nulle\n",
                "    \n",
                "    ioff_real = x[idx:idx+4]; idx += 4\n",
                "    ioff_imag = x[idx:idx+4]; idx += 4\n",
                "    I_OFF = ioff_real + 1j * ioff_imag\n",
                "    \n",
                "    return M, C, I_ON, I_OFF\n",
                "\n",
                "# Step 1: Optimize I (Fix M, C)\n",
                "def pack_step1(I_ON, I_OFF):\n",
                "    \"\"\"I_ON: phase nulle -> amplitudes seulement\"\"\"\n",
                "    return np.hstack([\n",
                "        np.abs(I_ON),\n",
                "        I_OFF.real.ravel(), I_OFF.imag.ravel()\n",
                "    ])\n",
                "\n",
                "def unpack_step1(x, fixed_M, fixed_C):\n",
                "    idx = 0\n",
                "    ion_amp = x[idx:idx+4]; idx += 4\n",
                "    I_ON = ion_amp + 0j  # Phase nulle\n",
                "    \n",
                "    ioff_real = x[idx:idx+4]; idx += 4\n",
                "    ioff_imag = x[idx:idx+4]; idx += 4\n",
                "    I_OFF = ioff_real + 1j * ioff_imag\n",
                "    \n",
                "    return fixed_M, fixed_C, I_ON, I_OFF\n",
                "\n",
                "# Step 2: Optimize C (Fix M, I)\n"
            ]
            
            # Trouver où finit pack_step1 et commence pack_step2
            step2_start = None
            for j, line in enumerate(source_lines):
                if '# Step 2: Optimize C' in line or 'def pack_step2' in line:
                    step2_start = j
                    break
            
            if step2_start:
                # Garder le reste du code après Step 2
                new_code.extend(source_lines[step2_start+1:])
                cell['source'] = new_code
                modified = True
                print("  → Fonctions modifiées")
        
        # Chercher l'initialisation
        if 'I_ON_init = np.ones(4, dtype=complex)' in source_text:
            print(f"✓ Cellule {i} trouvée: I_ON_init")
            for j, line in enumerate(source_lines):
                if 'I_ON_init = np.ones(4, dtype=complex)' in line:
                    source_lines[j] = line.replace(
                        'dtype=complex',
                        'dtype=float  # Réels purs, phase nulle'
                    )
                    modified = True
                    print("  → Initialisation modifiée")
                    break
    
    if not modified:
        print("⚠ Aucune modification effectuée")
        return False
    
    # Sauvegarder
    backup_path = notebook_path.replace('.ipynb', '_backup.ipynb')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"\n✓ Backup: {backup_path}")
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"✓ Modifié: {notebook_path}")
    
    return True

if __name__ == '__main__':
    notebook_path = '/mnt/e/PHOBos/analysis/Analysis.ipynb'
    
    print("Modification du notebook Analysis.ipynb")
    print("=" * 60)
    
    success = modify_notebook(notebook_path)
    
    if success:
        print("=" * 60)
        print("✓ Modifications appliquées!\n")
        print("Changements:")
        print("  • I_ON: phase commune = 0 (réels purs)")
        print("  • I_OFF: phases variables (complexes)")
        print("  • -4 paramètres d'optimisation")
    else:
        print("\n✗ Échec de la modification")
