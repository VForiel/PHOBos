#!/usr/bin/env python3
"""
Test simple de la fonction get_delays()
"""

import numpy as np
import sys
import os

# Ajouter le chemin du module kbench
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kbench.modules.atmosphere import get_delays, atmo_screen_kolmogorov


def test_kolmogorov_screen():
    """Test de génération d'écran de Kolmogorov."""
    print("Test 1: Génération d'écran de phase Kolmogorov")
    print("-" * 50)
    
    screen = atmo_screen_kolmogorov(
        size=256,
        physical_size=50.0,
        r0=0.16,
        L0=25.0
    )
    
    print(f"  ✓ Écran généré: {screen.shape}")
    print(f"  ✓ Type: {screen.dtype}")
    print(f"  ✓ RMS: {np.std(screen):.3f} rad")
    print(f"  ✓ Moyenne: {np.mean(screen):.3e} rad (doit être ~0)")
    
    assert screen.shape == (256, 256), "Mauvaise taille d'écran"
    assert np.abs(np.mean(screen)) < 0.1, "Moyenne non nulle"
    print("  ✓ Test réussi!\n")


def test_get_delays_basic():
    """Test basique de get_delays()."""
    print("Test 2: Fonction get_delays() - configuration basique")
    print("-" * 50)
    
    delays, times = get_delays(
        n_telescopes=4,
        telescope_diameter=1.8,
        r0=0.16,
        wind_speed=10.0,
        n_steps=10,
        demo=False
    )
    
    print(f"  ✓ Forme des retards: {delays.shape}")
    print(f"  ✓ Forme des temps: {times.shape}")
    print(f"  ✓ Durée: {times[-1]:.1f} s")
    print(f"  ✓ RMS global: {np.std(delays):.2f} nm")
    
    assert delays.shape == (10, 4), "Mauvaise forme de tableau"
    assert times.shape == (10,), "Mauvaise forme de temps"
    assert np.all(np.isfinite(delays)), "Valeurs non finies détectées"
    print("  ✓ Test réussi!\n")


def test_custom_positions():
    """Test avec positions personnalisées."""
    print("Test 3: Positions personnalisées des télescopes")
    print("-" * 50)
    
    positions = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ])
    
    delays, times = get_delays(
        telescope_positions=positions,
        n_steps=5,
        demo=False
    )
    
    print(f"  ✓ Nombre de télescopes détecté: {delays.shape[1]}")
    print(f"  ✓ Positions utilisées: {positions.shape}")
    
    assert delays.shape[1] == 4, "Mauvais nombre de télescopes"
    print("  ✓ Test réussi!\n")


def test_different_n_telescopes():
    """Test avec différents nombres de télescopes."""
    print("Test 4: Différents nombres de télescopes")
    print("-" * 50)
    
    for n in [3, 4, 6]:
        delays, _ = get_delays(
            n_telescopes=n,
            n_steps=5,
            demo=False
        )
        print(f"  ✓ {n} télescopes: {delays.shape}")
        assert delays.shape[1] == n, f"Erreur pour {n} télescopes"
    
    print("  ✓ Test réussi!\n")


def test_wavelength_scaling():
    """Test de la mise à l'échelle avec la longueur d'onde."""
    print("Test 5: Mise à l'échelle r0 avec longueur d'onde")
    print("-" * 50)
    
    # Test à deux longueurs d'onde différentes
    delays_h, _ = get_delays(
        wavelength=1.65e-6,  # Bande H
        n_steps=100,
        demo=False
    )
    
    delays_k, _ = get_delays(
        wavelength=2.2e-6,  # Bande K
        n_steps=100,
        demo=False
    )
    
    rms_h = np.std(delays_h)
    rms_k = np.std(delays_k)
    
    print(f"  ✓ RMS en bande H (1.65 µm): {rms_h:.2f} nm")
    print(f"  ✓ RMS en bande K (2.2 µm): {rms_k:.2f} nm")
    print(f"  ✓ Ratio RMS_K/RMS_H: {rms_k/rms_h:.3f}")
    print(f"    (attendu: ~{(2.2/1.65):.3f} car OPD ∝ λ)")
    
    # Le ratio devrait être proche de λ_K/λ_H
    expected_ratio = 2.2 / 1.65
    assert 0.8 * expected_ratio < rms_k/rms_h < 1.2 * expected_ratio, \
        "Mise à l'échelle incorrecte avec la longueur d'onde"
    print("  ✓ Test réussi!\n")


def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "=" * 70)
    print("TESTS DU MODULE ATMOSPHERE")
    print("=" * 70 + "\n")
    
    try:
        test_kolmogorov_screen()
        test_get_delays_basic()
        test_custom_positions()
        test_different_n_telescopes()
        test_wavelength_scaling()
        
        print("=" * 70)
        print("✓ TOUS LES TESTS RÉUSSIS!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ ÉCHEC DES TESTS")
        print("=" * 70)
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
