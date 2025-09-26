#!/usr/bin/env python3
"""
Test simple des modifications
"""

import kbench
from kbench.classes.deformable_mirror import DM
from kbench.classes.filter_wheel import FilterWheel
from kbench.classes.pupil_mask import PupilMask

print("=== Test simple des modifications ===\n")

try:
    print("1. Test DM:")
    dm = DM("27BW007#051")
    print("   DM créé avec succès")
    dm[0].set_ptt(100.0, 0.001, -0.002)
    print("   Segment modifié avec succès")
    del dm
    print("   DM supprimé avec succès\n")
    
    print("2. Test Filter Wheel:")
    fw = FilterWheel("/dev/ttyUSB2")
    print("   FilterWheel créé avec succès")
    fw.move(3)
    print("   Mouvement effectué avec succès")
    fw.close()
    print("   FilterWheel fermé avec succès\n")
    
    print("3. Test Pupil Mask:")
    pm = PupilMask("/dev/ttyUSB0", "/dev/ttyUSB1")
    print("   PupilMask créé avec succès")
    pm.move_h(1000, abs=False)
    print("   Mouvement horizontal effectué avec succès")
    pm.rotate(45.0, abs=False)
    print("   Rotation effectuée avec succès\n")
    
    print("✅ Tous les tests ont réussi !")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    
print("\n=== Fin des tests ===")