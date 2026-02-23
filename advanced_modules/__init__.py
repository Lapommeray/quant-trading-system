"""Advanced modules package.

This package uses lazy imports so optional dependencies
(e.g. ccxt, mplfinance) do not break unrelated module imports.
Submodules are loaded on first access via ``__getattr__``.
"""

import importlib as _importlib

_LAZY_IMPORTS = {
    "HumanLagExploit": ".human_lag_exploit",
    "InvisibleDataMiner": ".invisible_data_miner",
    "MetaAdaptiveAI": ".meta_adaptive_ai",
    "SelfDestructProtocol": ".self_destruct_protocol",
    "QuantumSentimentDecoder": ".quantum_sentiment_decoder",
    "QuantumTremorScanner": ".quantum_tremor_scanner",
    "SpectralSignalFusion": ".spectral_signal_fusion",
    "DNABreath": ".dna_breath",
    "DNAOverlord": ".dna_overlord",
    "VoidTraderChartRenderer": ".void_trader_chart_renderer",
    "MetaConsciousRoutingLayer": ".meta_conscious_routing_layer",
    "QuantumCodeGenerator": ".quantum_code_generator",
    "AntiStuck": ".anti_stuck",
    "TimeResonantNeuralLattice": ".time_resonant_neural_lattice",
    "SelfRewritingDNAAI": ".self_rewriting_dna_ai",
    "CausalQuantumReasoning": ".causal_quantum_reasoning:CausalQuantumReasoningEngine",
    "LatencyCancellationField": ".latency_cancellation_field",
    "EmotionHarvestAI": ".emotion_harvest_ai",
    "QuantumLiquiditySignatureReader": ".quantum_liquidity_signature_reader",
    "CausalFlowSplitter": ".causal_flow_splitter",
    "InverseTimeEchoes": ".inverse_time_echoes",
    "LiquidityEventHorizonMapper": ".liquidity_event_horizon_mapper",
    "ShadowSpreadResonator": ".shadow_spread_resonator",
    "ArbitrageSynapseChain": ".arbitrage_synapse_chain",
    "SentimentEnergyCouplingEngine": ".sentiment_energy_coupling_engine",
    "MultiTimelineProbabilityMesh": ".multi_timeline_probability_mesh",
    "SovereignQuantumOracle": ".sovereign_quantum_oracle",
    "SyntheticConsciousness": ".synthetic_consciousness",
    "LanguageUniverseDecoder": ".language_universe_decoder",
    "ZeroEnergyRecursiveIntelligence": ".zero_energy_recursive_intelligence",
    "TruthVerificationCore": ".truth_verification_core",
    "PureMathFoundation": ".pure_math_foundation",
    "MathComputationInterface": ".math_computation_interface",
    "AdvancedStochasticCalculus": ".advanced_stochastic_calculus",
    "QuantumProbability": ".quantum_probability",
    "TopologicalDataAnalysis": ".topological_data_analysis",
    "MeasureTheory": ".measure_theory",
    "RoughPathTheory": ".rough_path_theory",
    "MathematicalIntegrationLayer": ".mathematical_integration_layer",
    "QuantumTopologyAnalyzer": ".quantum_topology_analyzer",
    "NonErgodicCalculus": ".nonergodic_calculus",
    "MetaLearningEngine": ".meta_learning_engine",
    "NeuralPDEMarketAnalyzer": ".neural_pde_market_analyzer",
    "ExecutionAlphaOptimizer": ".execution_alpha_optimizer",
    "HyperTopologyAnalyzer": ".hyper_topology_analyzer",
    "PathSignatureTransformer": ".path_signature_transformer",
    "DarkPoolGAN": ".dark_pool_gan",
    "NeuromorphicPDE": ".neuromorphic_pde",
    "QuantumExecutionOptimizer": ".quantum_execution_optimizer",
    "DarkPoolDNADecoder": ".dark_pool_dna_decoder",
    "NeuralMarketHolography": ".neural_market_holography",
    "QuantumLiquidityWarper": ".quantum_liquidity_warper",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    entry = _LAZY_IMPORTS.get(name)
    if entry is not None:
        try:
            if ":" in entry:
                module_path, real_name = entry.split(":", 1)
            else:
                module_path, real_name = entry, name
            mod = _importlib.import_module(module_path, __name__)
            attr = getattr(mod, real_name)
            globals()[name] = attr
            return attr
        except Exception:
            raise ImportError(f"cannot import name {name!r} from 'advanced_modules'")
    raise AttributeError(f"module 'advanced_modules' has no attribute {name!r}")
