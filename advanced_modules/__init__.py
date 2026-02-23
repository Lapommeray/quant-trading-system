"""Advanced modules package.

This package intentionally avoids eager imports so optional dependencies
(e.g. ccxt, mplfinance) do not break unrelated module imports.
Import submodules directly, e.g. `advanced_modules.port_activity_analyzer`.
Lazy attribute access is provided so ``from advanced_modules import X`` still
works for the commonly used public symbols.
"""

import importlib as _importlib

_CLASS_MODULE_MAP = {
    "QuantumTremorScanner": "quantum_tremor_scanner",
    "SpectralSignalFusion": "spectral_signal_fusion",
    "DNABreath": "dna_breath",
    "DNAOverlord": "dna_overlord",
    "VoidTraderChartRenderer": "void_trader_chart_renderer",
    "TimeResonantNeuralLattice": "time_resonant_neural_lattice",
    "SelfRewritingDNAAI": "self_rewriting_dna_ai",
    "CausalQuantumReasoning": ("causal_quantum_reasoning", "CausalQuantumReasoningEngine"),
    "LatencyCancellationField": "latency_cancellation_field",
    "EmotionHarvestAI": "emotion_harvest_ai",
    "QuantumLiquiditySignatureReader": "quantum_liquidity_signature_reader",
    "CausalFlowSplitter": "causal_flow_splitter",
    "InverseTimeEchoes": "inverse_time_echoes",
    "LiquidityEventHorizonMapper": "liquidity_event_horizon_mapper",
    "ShadowSpreadResonator": "shadow_spread_resonator",
    "ArbitrageSynapseChain": "arbitrage_synapse_chain",
    "SentimentEnergyCouplingEngine": "sentiment_energy_coupling_engine",
    "MultiTimelineProbabilityMesh": "multi_timeline_probability_mesh",
    "SovereignQuantumOracle": "sovereign_quantum_oracle",
    "SyntheticConsciousness": "synthetic_consciousness",
    "LanguageUniverseDecoder": "language_universe_decoder",
    "ZeroEnergyRecursiveIntelligence": "zero_energy_recursive_intelligence",
    "TruthVerificationCore": "truth_verification_core",
    "HumanLagExploit": "human_lag_exploit",
}

__all__ = list(_CLASS_MODULE_MAP)


def __getattr__(name):
    if name in _CLASS_MODULE_MAP:
        entry = _CLASS_MODULE_MAP[name]
        if isinstance(entry, tuple):
            module_name, class_name = entry
        else:
            module_name, class_name = entry, name
        mod = _importlib.import_module(f".{module_name}", __name__)
        attr = getattr(mod, class_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
