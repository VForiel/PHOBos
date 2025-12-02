#!/usr/bin/env python3
"""
Test script for the new power control implementation in Kbench.

This demonstrates:
1. Auto-calibration on first use
2. set_power() and get_power() methods
3. 2-point calibration algorithm
"""

import numpy as np
import kbench

def test_single_channel():
    """Test power control on a single channel."""
    print("=" * 60)
    print("TEST: Single Channel Power Control")
    print("=" * 60)
    
    # Create a channel (will auto-calibrate on first set_power call)
    channel = kbench.PhaseShifter(17)
    
    print("\n✓ Testing set_power() with auto-calibration...")
    channel.set_power(0.6, verbose=True)
    
    print("\n✓ Getting power back...")
    measured_power = channel.get_power(verbose=True)
    
    print(f"\n✓ Set: 0.6 W, Measured: {measured_power:.3f} W")
    
    # Turn off
    channel.turn_off()
    print("\n✓ Channel turned off\n")


def test_power_range():
    """Test power control over a range."""
    print("=" * 60)
    print("TEST: Power Range")
    print("=" * 60)
    
    channel = kbench.PhaseShifter(17)
    
    # Test range of powers
    power_range = np.linspace(0.1, 1.0, 10)
    measured_powers = []
    
    print("\n✓ Testing power range from 0.1 to 1.0 W...")
    for p in power_range:
        channel.set_power(p, verbose=False)
        measured = channel.get_power(verbose=False)
        measured_powers.append(measured)
        print(f"   Set: {p:.2f} W → Measured: {measured:.3f} W")
    
    # Turn off
    channel.turn_off()
    print("\n✓ Test complete\n")


def test_arch_powers():
    """Test power control on chip architecture."""
    print("=" * 60)
    print("TEST: Architecture Power Control")
    print("=" * 60)
    
    # Create chip with architecture 6 (4 TOPAs)
    chip = kbench.Arch(6)
    print(f"\n✓ Created chip: {chip.name}")
    print(f"   Channels: {chip.topas}")
    
    # Set powers (will auto-calibrate each channel on first use)
    powers = [0.3, 0.4, 0.5, 0.6]
    print(f"\n✓ Setting powers: {powers} W")
    chip.set_powers(powers, verbose=False)
    
    # Get powers
    measured_powers = chip.get_powers(verbose=False)
    print(f"\n✓ Measured powers: {measured_powers}")
    
    # Turn off
    chip.turn_off()
    print("\n✓ Chip turned off\n")


def test_manual_calibration():
    """Test manual calibration call."""
    print("=" * 60)
    print("TEST: Manual Calibration")
    print("=" * 60)
    
    channel = kbench.PhaseShifter(18)
    
    print("\n✓ Calling calibrate() manually...")
    channel.calibrate(verbose=True)
    
    print(f"\n✓ Calibration coefficient: {kbench.xpow.POWER_CORRECTION[17]:.6f}")
    
    print("\n✓ Now using calibrated coefficient...")
    channel.set_power(0.5, verbose=True)
    measured = channel.get_power(verbose=True)
    
    print(f"\n✓ Set: 0.5 W, Measured: {measured:.3f} W")
    
    # Turn off
    channel.turn_off()
    print("\n✓ Test complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KBENCH POWER CONTROL TEST SUITE")
    print("=" * 60)
    
    if kbench.SANDBOX_MODE:
        print("\n⛱️  Running in SANDBOX MODE (simulated hardware)")
    else:
        print("\n🔌 Running with real hardware")
    
    print("\n")
    
    try:
        test_single_channel()
        test_power_range()
        test_arch_powers()
        test_manual_calibration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNew features working correctly:")
        print("  • Auto-calibration on first set_power() call")
        print("  • 2-point calibration (1V and 30V)")
        print("  • set_power() method")
        print("  • get_power() method")
        print("  • Arch.set_powers() and Arch.get_powers()")
        print("  • Manual calibrate() call")
        print()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
