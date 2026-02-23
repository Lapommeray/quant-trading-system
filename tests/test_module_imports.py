#!/usr/bin/env python3
"""Test module imports for restored modules."""

import importlib

import pytest

MODULES_TO_TEST = [
    ("core.qol_engine", "QOLEngine"),
    ("signals.veve_triggers", "VeveTriggers"),
    ("signals.legba_crossroads", "LegbaCrossroads"),
    ("quant.entropy_shield", "EntropyShield"),
    ("quant.liquidity_mirror", "LiquidityMirror"),
    ("advanced_modules.dna_breath", "DNABreath"),
    ("advanced_modules.dna_overlord", "DNAOverlord"),
    ("advanced_modules.spectral_signal_fusion", "SpectralSignalFusion"),
    ("advanced_modules.quantum_tremor_scanner", "QuantumTremorScanner"),
    ("advanced_modules.time_fractal_fft", "TimeFractalFFT"),
]


@pytest.mark.parametrize("module_path,class_name", MODULES_TO_TEST)
def test_module_import(module_path, class_name):
    module = importlib.import_module(module_path)
    assert hasattr(module, class_name)
