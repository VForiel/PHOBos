import json
import os
import re

NOTEBOOK_PATH = "/mnt/e/PHOBos/analysis/Analysis.ipynb"

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    modified_markdown = False
    modified_code = False

    for cell in cells:
        # Patch Markdown Section 3.1
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            if "## 3. Hypothèses et Approximations" in source and "Passivité du Crosstalk" in source:
                print("Found Section 3.1 cell. Patching...")
                new_source = []
                for line in cell['source']:
                    if "Passivité du Crosstalk (C)" in line:
                         # Replace strictly
                         new_source.append("2.  **Conservation de l'Énergie (Unitarité de C)** : $ C C^\\dagger = I $ (Matrice unitaire générale)\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                modified_markdown = True

        # Patch Code Block 5 (Results Display)
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "from IPython.display import display, Math" in source and "matrix_to_latex" in source:
                print("Found Block 5 code cell. Patching...")
                # New code source logic
                new_code_lines = [
                    "from IPython.display import display, Math\n",
                    "import numpy as np\n",
                    "\n",
                    "def complex_to_latex(z):\n",
                    "    return f\"{np.abs(z):.2f} e^{{j{np.angle(z):.2f}}}\"\n",
                    "\n",
                    "def matrix_to_latex(mat):\n",
                    "    latex_str = r\"\\begin{bmatrix}\"\n",
                    "    for row in mat:\n",
                    "        latex_str += \" & \".join([complex_to_latex(x) for x in row]) + r\" \\\\\"\n",
                    "    latex_str += r\"\\end{bmatrix}\"\n",
                    "    return latex_str\n",
                    "\n",
                    "if results:\n",
                    "    M = results['M']\n",
                    "    C = results['C']\n",
                    "    \n",
                    "    display(Math(r\"\\textbf{Matrix M (5x5):} \" + matrix_to_latex(M)))\n",
                    "    display(Math(r\"\\textbf{Matrix C (5x5):} \" + matrix_to_latex(C)))\n",
                    "    \n",
                    "    I_ON = results['I_ON']\n",
                    "    I_OFF = results['I_OFF']\n",
                    "    I_stray_val = results['I_stray']\n",
                    "    \n",
                    "    display(Math(r\"\\textbf{Intensities Inputs:}\"))\n",
                    "    display(Math(r\"\\alpha_{i,max} = [\" + \", \".join([complex_to_latex(x) for x in I_ON]) + \"]\"))\n",
                    "    display(Math(r\"\\alpha_{i,off} = [\" + \", \".join([complex_to_latex(x) for x in I_OFF]) + \"]\"))\n",
                    "    display(Math(r\"\\alpha_{5,stray} = \" + complex_to_latex(I_stray_val)))\n"
                ]
                cell['source'] = new_code_lines
                cell['outputs'] = [] # Clear outputs to force re-run feeling and avoid old data mismatch
                modified_code = True

    if modified_markdown and modified_code:
        print("Writing modifications back to file...")
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Success.")
    else:
        print(f"Warning: Modifications incomplete. Markdown: {modified_markdown}, Code: {modified_code}")

if __name__ == "__main__":
    patch_notebook()
