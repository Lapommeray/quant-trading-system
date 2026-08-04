"""Autonomous trading runtime.

The package exposes the canonical organism, shared event bus, durable learning
memory, market-regime detector and bounded self-coding engine.  Generated code
is restricted to validated artifacts; live execution and risk controls remain
fail-closed.
"""

from .organism import (
    BaseModule,
    BaseTradingModule,
    ModuleAutoDiscovery,
    ModuleHealth,
    ModuleResult,
    Organism,
    OrganismConfig,
    get_event_bus,
    get_organism,
    get_registered_modules,
    register_module,
    reset_organism,
)
from .audit import AuditTrail
from .consensus import ConsensusEngine, ConsensusResult
from .execution import AutonomousExecutor, ExecutorConfig
from .events import AutonomyEvent, EventTypes
from core.event_bus import (
    Event,
    EventBus,
    EventPriority,
    PriorityEventBus,
    reset_event_bus,
)
from .gold_set import GoldSet, GoldSetStressTester, GoldStressReport
from .guardrails import AutonomousGuardrails, CapitalTracker, GuardrailLimits
from .learning import LearningStore, MistakeMemory
from .market import MarketRegime, MarketRegimeDetector, extract_closes
from .monitor import AutonomousMonitor
from .protected_executor import ProtectedTradeExecutor
from .sandbox import SandboxExecutor, SandboxPolicy, SandboxResult
from .sentinel import (
    MultiTimeframeSentinel,
    PanicSentinel,
    SentinelConfig,
    SentinelDecision,
)
from .shadow import ShadowDeployment, ShadowManager, ShadowMetrics, ShadowPolicy
from .self_coding import (
    ApprovalPolicy,
    AutoCodingEngine,
    ChangeProposal,
    ChangeRisk,
    CodeProposal,
    ProposalStatus,
    SafeCodeValidator,
    SelfCodingEngine,
    TestSuiteGenerator,
    ValidationReport,
)

__all__ = [
    "ApprovalPolicy",
    "AuditTrail",
    "AutonomousGuardrails",
    "AutonomyEvent",
    "AutoCodingEngine",
    "BaseModule",
    "BaseTradingModule",
    "ChangeProposal",
    "ChangeRisk",
    "CodeProposal",
    "ConsensusEngine",
    "ConsensusResult",
    "CapitalTracker",
    "ExecutorConfig",
    "AutonomousExecutor",
    "AutonomousMonitor",
    "Event",
    "EventBus",
    "EventPriority",
    "EventTypes",
    "PriorityEventBus",
    "GoldSet",
    "GoldSetStressTester",
    "GoldStressReport",
    "GuardrailLimits",
    "LearningStore",
    "MarketRegime",
    "MarketRegimeDetector",
    "MistakeMemory",
    "ModuleAutoDiscovery",
    "ModuleHealth",
    "ModuleResult",
    "MultiTimeframeSentinel",
    "Organism",
    "OrganismConfig",
    "PanicSentinel",
    "ProposalStatus",
    "ProtectedTradeExecutor",
    "SafeCodeValidator",
    "SandboxExecutor",
    "SandboxPolicy",
    "SandboxResult",
    "SelfCodingEngine",
    "TestSuiteGenerator",
    "SentinelConfig",
    "SentinelDecision",
    "ShadowDeployment",
    "ShadowManager",
    "ShadowMetrics",
    "ShadowPolicy",
    "ValidationReport",
    "extract_closes",
    "get_event_bus",
    "get_organism",
    "get_registered_modules",
    "register_module",
    "reset_event_bus",
    "reset_organism",
]
