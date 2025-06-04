#!/usr/bin/env python3
"""Test script to verify CI fixes work locally."""

print("Testing imports...")

try:
    import docker
    print("✅ docker import successful")
except ImportError as e:
    print(f"❌ docker import failed: {e}")

try:
    from advanced_modules.inverse_time_echoes import InverseTimeEchoes
    print("✅ InverseTimeEchoes import successful")
except ImportError as e:
    print(f"❌ InverseTimeEchoes import failed: {e}")

try:
    from predictive_overlay.neural_forecaster import NeuralForecaster
    print("✅ NeuralForecaster import successful")
except ImportError as e:
    print(f"❌ NeuralForecaster import failed: {e}")

print("✅ All critical imports tested")
