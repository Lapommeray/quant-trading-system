"""Bounded self-coding and recovery for trading modules.

This module implements the useful part of autonomous coding without allowing a
model to rewrite the live execution path.  A cycle can diagnose a module,
create a deterministic improvement artifact, validate it, and approve/apply it
when the artifact is low risk.  Protected source files (execution, risk,
credentials, safety and the organism itself) always remain pending for human
review.

Generated artifacts are policy/configuration candidates.  They are not
imported or executed as live code.  Runtime tuning is restricted to an
allow-listed adaptive namespace on the module.  This distinction is important
for a system that can place real orders: a passing syntax check is not proof
that a strategy is profitable or safe.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .audit import AuditTrail
from .sandbox import SandboxExecutor, SandboxPolicy


class ChangeRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProposalStatus(str, Enum):
    GENERATED = "generated"
    VALIDATED = "validated"
    APPROVED = "approved"
    PENDING_APPROVAL = "pending_approval"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass
class ValidationReport:
    passed: bool
    syntax_valid: bool
    policy_valid: bool
    tests_passed: Optional[bool] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestExecutionReport:
    passed: bool
    executed: bool
    skipped: bool = False
    return_code: Optional[int] = None
    output: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedTestSuite:
    code: str
    path: Optional[str] = None
    validation: Optional[ValidationReport] = None
    execution: Optional[TestExecutionReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "validation": self.validation.to_dict() if self.validation else None,
            "execution": self.execution.to_dict() if self.execution else None,
        }


@dataclass
class CodeProposal:
    proposal_id: str
    module_name: str
    description: str
    risk: ChangeRisk
    status: ProposalStatus
    created_at: float
    source_path: Optional[str]
    artifact_path: str
    code: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[ValidationReport] = None
    approval_reason: str = ""
    applied_at: Optional[float] = None
    test_code: str = ""
    test_path: Optional[str] = None
    test_execution: Optional[TestExecutionReport] = None
    regression_execution: Optional[TestExecutionReport] = None
    shadow_id: Optional[str] = None
    deployment: str = "not_deployed"

    @property
    def test_suite(self) -> tuple[str, str]:
        """Return the candidate/test pair required by the approval pipeline."""
        return self.code, self.test_code

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["risk"] = self.risk.value
        result["status"] = self.status.value
        if self.validation:
            result["validation"] = self.validation.to_dict()
        return result


# A semantic alias makes integrations read naturally.
ChangeProposal = CodeProposal


class PenaltyBox:
    """Temporarily disable the self-coder after a policy violation."""

    def __init__(self, penalty_hours: float = 1.0, clock=time.time) -> None:
        self.penalty_seconds = max(1.0, float(penalty_hours) * 3600.0)
        self.clock = clock
        self.until = 0.0
        self.reasons: list[str] = []

    @property
    def active(self) -> bool:
        return float(self.clock()) < self.until

    def trigger(self, reason: str) -> None:
        self.until = max(self.until, float(self.clock()) + self.penalty_seconds)
        self.reasons.append(str(reason))
        self.reasons = self.reasons[-20:]

    def remaining_seconds(self) -> float:
        return max(0.0, self.until - float(self.clock()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "remaining_seconds": self.remaining_seconds(),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ApprovalPolicy:
    """Policy for autonomous approval.

    Only generated artifacts and allow-listed adaptive parameters can be
    auto-approved.  A deployment may disable even that behavior, but no flag
    can make protected source mutation auto-approved.
    """

    auto_approve_low_risk: bool = True
    auto_apply_low_risk: bool = True
    # Every proposal gets a generated test suite.  Baseline regression runs
    # are enabled by the live Organism configuration, and can be injected in
    # isolated unit tests without spawning a nested pytest process.
    require_tests: bool = True
    run_baseline_tests: bool = False
    baseline_command: tuple[str, ...] = ()
    test_timeout_seconds: float = 5.0
    penalty_hours: float = 1.0
    max_artifact_bytes: int = 64_000
    max_code_lines: int = 200
    max_complexity: int = 10
    max_nested_depth: int = 3
    allowed_imports: tuple[str, ...] = (
        "numpy",
        "pandas",
        "ta",
        "scipy",
        "datetime",
        "math",
        "statistics",
        "typing",
        "dataclasses",
        "unittest",
    )
    protected_path_tokens: tuple[str, ...] = (
        "okx_live",
        "execution",
        "risk",
        "safety",
        "credential",
        "secret",
        "organism",
        "event_bus",
        "main.py",
    )


class SafeCodeValidator:
    """Strict AST allow-list validator; generated code is never executed here."""

    _forbidden_imports = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "importlib",
        "ctypes",
        "pickle",
        "shelve",
        "marshal",
        "builtins",
        "requests",
        "ccxt",
        "multiprocessing",
        "threading",
    }
    _forbidden_calls = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "skip",
        "skipIf",
        "skipUnless",
        "skipTest",
    }
    _forbidden_names = {
        "__builtins__",
        "__globals__",
        "__code__",
        "environ",
        "getenv",
        "env",
    }

    def __init__(
        self,
        *,
        max_bytes: int = 64_000,
        max_lines: int = 200,
        max_complexity: int = 10,
        max_nested_depth: int = 3,
        allowed_imports: Optional[Iterable[str]] = None,
    ) -> None:
        self.max_bytes = max(1_024, int(max_bytes))
        self.max_lines = max(1, int(max_lines))
        self.max_complexity = max(1, int(max_complexity))
        self.max_nested_depth = max(1, int(max_nested_depth))
        self.allowed_imports = set(
            allowed_imports
            or {
                "numpy",
                "pandas",
                "ta",
                "scipy",
                "datetime",
                "math",
                "statistics",
                "typing",
                "dataclasses",
                "unittest",
            }
        )

    def validate(
        self,
        code: str,
        *,
        filename: str = "generated_module.py",
        tests: Optional[Callable[[str], Any]] = None,
        require_tests: bool = False,
        allowed_imports: Optional[Iterable[str]] = None,
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        syntax_valid = False
        policy_valid = True
        tests_passed: Optional[bool] = None
        tree: Optional[ast.AST] = None
        permitted = self.allowed_imports | set(allowed_imports or ())

        if not isinstance(code, str) or not code.strip():
            errors.append("generated code is empty")
        elif len(code.encode("utf-8")) > self.max_bytes:
            errors.append(f"generated code exceeds {self.max_bytes} bytes")
        else:
            try:
                tree = ast.parse(code, filename=filename, mode="exec")
                syntax_valid = True
            except SyntaxError as exc:
                errors.append(f"syntax error: {exc}")

        if tree is not None:
            line_count = max(
                (getattr(node, "lineno", 0) for node in ast.walk(tree)), default=0
            )
            if line_count > self.max_lines:
                policy_valid = False
                errors.append(f"code exceeds {self.max_lines} lines")
            complexity = sum(
                isinstance(
                    node,
                    (
                        ast.If,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.Try,
                        ast.IfExp,
                        ast.ExceptHandler,
                    ),
                )
                or (isinstance(node, ast.BoolOp) and len(node.values) > 1)
                for node in ast.walk(tree)
            )
            if complexity > self.max_complexity:
                policy_valid = False
                errors.append(f"complexity {complexity} exceeds {self.max_complexity}")
            depth = self._max_nested_depth(tree)
            if depth > self.max_nested_depth:
                policy_valid = False
                errors.append(f"nesting depth {depth} exceeds {self.max_nested_depth}")

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".", 1)[0]
                        if root in self._forbidden_imports or root not in permitted:
                            policy_valid = False
                            errors.append(f"forbidden import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".", 1)[0]
                    if root in self._forbidden_imports or root not in permitted:
                        policy_valid = False
                        errors.append(f"forbidden import: {node.module}")
                elif isinstance(node, ast.Call):
                    function_name = self._call_name(node)
                    if function_name in self._forbidden_calls:
                        policy_valid = False
                        errors.append(f"forbidden call: {function_name}")
                elif isinstance(node, ast.Attribute):
                    if node.attr.startswith("__") or node.attr in self._forbidden_names:
                        policy_valid = False
                        errors.append(f"forbidden attribute: {node.attr}")
                elif isinstance(node, ast.Name) and node.id in self._forbidden_names:
                    policy_valid = False
                    errors.append(f"forbidden name: {node.id}")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    lowered = node.value.lower()
                    if any(
                        token in lowered
                        for token in (
                            "okx_api_key",
                            "okx_api_secret",
                            "environ",
                            "__import__",
                        )
                    ):
                        policy_valid = False
                        errors.append("credential/environment access marker detected")

            if re.search(r"[A-Za-z0-9+/=]{80,}", code) or code.count("chr(") > 5:
                policy_valid = False
                errors.append("possible encoded/obfuscated payload detected")

        if tests is not None and syntax_valid and policy_valid:
            try:
                tests_passed = bool(tests(code))
            except Exception as exc:
                tests_passed = False
                errors.append(f"candidate test failed: {exc}")
            if tests_passed is False:
                errors.append("candidate tests did not pass")
        elif require_tests:
            tests_passed = False
            errors.append(
                "tests are required but no candidate test runner was supplied"
            )
        else:
            warnings.append("no runtime/backtest was executed; syntax and policy only")

        return ValidationReport(
            passed=not errors
            and syntax_valid
            and policy_valid
            and tests_passed is not False,
            syntax_valid=syntax_valid,
            policy_valid=policy_valid,
            tests_passed=tests_passed,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _max_nested_depth(self, tree: ast.AST) -> int:
        controls = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Try,
        )

        def walk(node: ast.AST, depth: int) -> int:
            next_depth = depth + 1 if isinstance(node, controls) else depth
            return max(
                (walk(child, next_depth) for child in ast.iter_child_nodes(node)),
                default=next_depth,
            )

        return walk(tree, 0)

    # Friendly aliases for callers that use a verb.
    validate_code = validate
    validate_ast = validate


class RegressionTestRunner:
    """Run a configured baseline test command in a bounded subprocess."""

    def __init__(
        self,
        project_root: str | os.PathLike[str],
        *,
        command: Optional[Iterable[str]] = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.command = tuple(command or ())
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def run(self) -> TestExecutionReport:
        if not self.command:
            return TestExecutionReport(
                passed=True,
                executed=False,
                skipped=True,
                error="baseline command not configured",
            )
        if os.environ.get("QTS_BASELINE_RUN") == "1":
            return TestExecutionReport(
                passed=True,
                executed=False,
                skipped=True,
                error="nested baseline run suppressed",
            )
        env = dict(os.environ)
        env["QTS_BASELINE_RUN"] = "1"
        try:
            result = subprocess.run(
                list(self.command),
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return TestExecutionReport(
                passed=result.returncode == 0,
                executed=True,
                return_code=result.returncode,
                output=output[-20_000:],
                error=(
                    "" if result.returncode == 0 else "baseline regression tests failed"
                ),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return TestExecutionReport(
                passed=False,
                executed=True,
                error=f"baseline test runner error: {exc}",
            )


class GeneratedTestRunner:
    """Validate and run a proposal's generated unit test in isolation."""

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.sandbox = SandboxExecutor(
            SandboxPolicy(timeout_seconds=self.timeout_seconds)
        )

    def run(
        self, proposal: CodeProposal, validator: SafeCodeValidator
    ) -> TestExecutionReport:
        if not proposal.test_code or not proposal.test_path:
            return TestExecutionReport(
                False, False, error="generated test suite missing"
            )
        module_name = Path(proposal.artifact_path).stem
        test_name = Path(proposal.test_path).stem
        test_validation = validator.validate(
            proposal.test_code,
            filename=proposal.test_path,
            allowed_imports={"unittest", module_name},
        )
        if not test_validation.passed:
            proposal.diagnosis["generated_test_validation"] = test_validation.to_dict()
            return TestExecutionReport(
                passed=False,
                executed=False,
                error="generated test suite failed AST validation",
            )
        result = self.sandbox.run(
            [
                sys.executable,
                "-I",
                "-m",
                "unittest",
                "discover",
                "-s",
                str(Path(proposal.test_path).parent),
                "-p",
                f"{Path(proposal.test_path).name}",
            ],
            cwd=Path(proposal.test_path).parent,
        )
        output = result.stdout + result.stderr
        skipped = "skipped" in output.lower() or "skiptest" in output.lower()
        passed = result.success and not skipped
        return TestExecutionReport(
            passed=passed,
            executed=True,
            skipped=skipped,
            return_code=result.return_code,
            output=output[-20_000:],
            error=(
                "generated suite contained skipped tests"
                if skipped
                else result.error
                or ("" if passed else f"generated suite {test_name} failed")
            ),
        )


class TestSuiteGenerator:
    """Deterministically produce a non-skippable unit-test companion."""

    @staticmethod
    def generate(module_stem: str) -> str:
        return f'''"""Generated tests for {module_stem}."""
import unittest
import {module_stem} as candidate


class GeneratedCandidateTests(unittest.TestCase):
    def test_parameters_are_mapping(self):
        parameters = candidate.get_parameters()
        self.assertIsInstance(parameters, dict)

    def test_metadata_is_present(self):
        self.assertEqual(candidate.METADATA.get("generated_by"), "local-bounded-self-coding-v1")


if __name__ == "__main__":
    unittest.main()
'''


class SelfCodingEngine:
    """Generate, validate and govern safe module-improvement artifacts."""

    _safe_parameter_names = {
        "confidence_floor",
        "weight_multiplier",
        "lookback",
        "cooldown_seconds",
        "volatility_multiplier",
        "regime_affinity_multiplier",
    }

    def __init__(
        self,
        *,
        project_root: Optional[str | os.PathLike[str]] = None,
        artifact_dir: Optional[str | os.PathLike[str]] = None,
        policy: Optional[ApprovalPolicy] = None,
        validator: Optional[SafeCodeValidator] = None,
        event_bus: Any = None,
        clock=time.time,
    ) -> None:
        root = (
            Path(project_root or Path(__file__).resolve().parents[1])
            .expanduser()
            .resolve()
        )
        self.project_root = root
        configured_artifacts = (
            artifact_dir
            or os.environ.get("QTS_SELF_CODING_DIR")
            or root / "strategies" / "evolved"
        )
        self.artifact_dir = Path(configured_artifacts).expanduser().resolve()
        self.policy = policy or ApprovalPolicy()
        self.validator = validator or SafeCodeValidator(
            max_bytes=self.policy.max_artifact_bytes,
            max_lines=self.policy.max_code_lines,
            max_complexity=self.policy.max_complexity,
            max_nested_depth=self.policy.max_nested_depth,
            allowed_imports=self.policy.allowed_imports,
        )
        self.event_bus = event_bus
        self.clock = clock
        self.audit_trail = AuditTrail("audit_logs/code_generation.jsonl")
        self.penalty_box = PenaltyBox(self.policy.penalty_hours, clock)
        self.generated_test_runner = GeneratedTestRunner(
            self.policy.test_timeout_seconds
        )
        baseline_command = self.policy.baseline_command
        if self.policy.run_baseline_tests and not baseline_command:
            baseline_command = (sys.executable, "-m", "pytest", "-q")
        self.baseline_runner = RegressionTestRunner(
            root,
            command=baseline_command if self.policy.run_baseline_tests else (),
            timeout_seconds=max(30.0, self.policy.test_timeout_seconds * 8),
        )
        self.proposals: Dict[str, CodeProposal] = {}
        self._applied_count = 0
        self._manifest_path = self.artifact_dir / "approval_manifest.jsonl"

    # ------------------------------------------------------------- diagnosis
    def diagnose(
        self, module: Any, context: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        context = dict(context or {})
        stats = context.get("stats", {}) or {}
        mistakes = context.get("mistakes", []) or []
        health = getattr(module, "health", None)
        diagnosis = {
            "module": getattr(module, "module_name", module.__class__.__name__),
            "status": getattr(health, "status", "unknown"),
            "error_count": int(getattr(health, "error_count", 0) or 0),
            "last_error": getattr(health, "last_error", None),
            "sample_count": int(stats.get("sample_count", 0) or 0),
            "avg_reward": float(stats.get("avg_reward", 0.5) or 0.5),
            "mistake_rate": float(stats.get("mistake_rate", 0.0) or 0.0),
            "mistakes": list(mistakes)[-5:],
            "regime": context.get("regime", "unknown"),
        }
        custom = getattr(module, "diagnose", None)
        if callable(custom):
            try:
                extra = custom(context)
                if isinstance(extra, Mapping):
                    diagnosis.update({str(k): extra[k] for k in extra})
            except Exception as exc:
                diagnosis["diagnostic_error"] = str(exc)
        return diagnosis

    def _classify_risk(self, module: Any, artifact_path: Path, code: str) -> ChangeRisk:
        # Artifact output is low-risk by construction.  A module or caller can
        # still mark itself protected; this is defense in depth.
        # The artifact directory is intentionally non-production; arbitrary
        # temporary directory names must not change risk classification.  Use
        # the module identity/source namespace instead.
        module_name = str(getattr(module, "module_name", "")).lower()
        module_path = str(getattr(module.__class__, "__module__", "")).lower()
        source_path = ""
        try:
            source_path = str(
                Path(__import__("inspect").getfile(module.__class__)).resolve()
            ).lower()
        except Exception:
            pass
        for token in self.policy.protected_path_tokens:
            if (
                token.lower() in module_name
                or token.lower() in module_path
                or token.lower() in source_path
            ):
                return ChangeRisk.CRITICAL
        dangerous_markers = (
            "place_order",
            "leverage",
            "max_position",
            "kill_switch",
            "credential",
            "secret",
        )
        if any(marker in code.lower() for marker in dangerous_markers):
            return ChangeRisk.HIGH
        return ChangeRisk.LOW

    def _safe_parameters(
        self,
        module: Any,
        diagnosis: Mapping[str, Any],
        regime_parameters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = (
            getattr(module, "config", {}).get("adaptive", {})
            if isinstance(getattr(module, "config", {}), Mapping)
            else {}
        )
        parameters: Dict[str, Any] = {
            str(key): value
            for key, value in dict(current or {}).items()
            if str(key) in self._safe_parameter_names
        }
        parameters.setdefault("weight_multiplier", 1.0)
        parameters.setdefault("confidence_floor", 0.60)

        avg_reward = float(diagnosis.get("avg_reward", 0.5) or 0.5)
        mistake_rate = float(diagnosis.get("mistake_rate", 0.0) or 0.0)
        if avg_reward < 0.45 or mistake_rate > 0.30:
            parameters["weight_multiplier"] = max(
                0.50, float(parameters["weight_multiplier"]) * 0.95
            )
            parameters["confidence_floor"] = min(
                0.90, float(parameters["confidence_floor"]) + 0.02
            )
        elif avg_reward > 0.65 and mistake_rate < 0.15:
            parameters["weight_multiplier"] = min(
                1.20, float(parameters["weight_multiplier"]) * 1.02
            )

        for key, value in (regime_parameters or {}).items():
            if key in self._safe_parameter_names:
                parameters[key] = value
        # Never let a generated candidate carry an execution-risk parameter.
        return {
            key: parameters[key]
            for key in sorted(parameters)
            if key in self._safe_parameter_names
        }

    def _render_code(
        self,
        *,
        module_name: str,
        proposal_id: str,
        regime: str,
        parameters: Mapping[str, Any],
        lessons: Iterable[Any],
    ) -> str:
        safe_module = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
        metadata = {
            "module_name": module_name,
            "proposal_id": proposal_id,
            "regime": regime,
            "generated_by": "local-bounded-self-coding-v1",
        }
        lessons_list = [
            str(item.get("lesson", item)) if isinstance(item, Mapping) else str(item)
            for item in list(lessons)[:5]
        ]
        return (
            '"""Generated adaptive policy artifact. Not imported into the live trader."""\n\n'
            f"MODULE_NAME = {module_name!r}\n"
            f"ARTIFACT_NAME = {safe_module!r}\n"
            f"METADATA = {json.dumps(metadata, sort_keys=True)}\n"
            f"PARAMETERS = {dict(parameters)!r}\n"
            f"LESSONS = {lessons_list!r}\n\n"
            "def get_parameters():\n"
            '    """Return a copy of the allow-listed adaptive parameters."""\n'
            "    return dict(PARAMETERS)\n"
        )

    def _render_test_code(self, module_stem: str) -> str:
        return TestSuiteGenerator.generate(module_stem)

    # ------------------------------------------------------------- lifecycle
    def generate_proposal(
        self,
        module: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
        regime_parameters: Optional[Mapping[str, Any]] = None,
    ) -> CodeProposal:
        if self.penalty_box.active:
            raise RuntimeError(
                f"self-coder penalty box active for {self.penalty_box.remaining_seconds():.0f}s"
            )
        context = dict(context or {})
        module_name = str(getattr(module, "module_name", module.__class__.__name__))
        proposal_id = uuid.uuid4().hex
        diagnosis = self.diagnose(module, context)
        parameters = self._safe_parameters(module, diagnosis, regime_parameters)
        code = self._render_code(
            module_name=module_name,
            proposal_id=proposal_id,
            regime=str(context.get("regime", "unknown")),
            parameters=parameters,
            lessons=context.get("mistakes", []),
        )
        module_dir = self.artifact_dir / re.sub(r"[^a-zA-Z0-9_.-]", "_", module_name)
        artifact_path = module_dir / f"proposal_{proposal_id}.py"
        test_path = module_dir / f"test_proposal_{proposal_id}.py"
        test_code = self._render_test_code(artifact_path.stem)
        risk = self._classify_risk(module, artifact_path, code)
        source_path: Optional[str]
        try:
            source_path = str(
                Path(__import__("inspect").getfile(module.__class__)).resolve()
            )
        except Exception:
            source_path = None
        proposal = CodeProposal(
            proposal_id=proposal_id,
            module_name=module_name,
            description="Tune allow-listed adaptive parameters from observed outcomes and market regime.",
            risk=risk,
            status=ProposalStatus.GENERATED,
            created_at=float(self.clock()),
            source_path=source_path,
            artifact_path=str(artifact_path),
            code=code,
            parameters=parameters,
            diagnosis=diagnosis,
            test_code=test_code,
            test_path=str(test_path),
        )
        self.proposals[proposal_id] = proposal
        # Persist the candidate immediately for review/recovery.  It is an
        # inert artifact and is never imported as live code.
        try:
            self._write_artifact(proposal)
            self._write_test_artifact(proposal)
        except OSError as exc:
            proposal.diagnosis["artifact_write_error"] = str(exc)
        self._publish("CODE_PROPOSED", proposal.to_dict())
        return proposal

    def validate_proposal(
        self,
        proposal: CodeProposal | str,
        *,
        tests: Optional[Callable[[str], Any]] = None,
    ) -> ValidationReport:
        proposal = self._get(proposal)
        if self.penalty_box.active:
            report = ValidationReport(
                passed=False,
                syntax_valid=False,
                policy_valid=False,
                errors=["self-coder penalty box active"],
            )
            proposal.validation = report
            proposal.status = ProposalStatus.REJECTED
            return report

        code_report = self.validator.validate(
            proposal.code,
            filename=proposal.artifact_path,
            tests=tests,
            require_tests=False,
        )
        module_stem = Path(proposal.artifact_path).stem
        test_report = self.validator.validate(
            proposal.test_code,
            filename=proposal.test_path or "generated_test.py",
            require_tests=False,
            allowed_imports={"unittest", module_stem},
        )
        proposal.diagnosis["generated_test_validation"] = test_report.to_dict()
        if test_report.passed:
            proposal.test_execution = self.generated_test_runner.run(
                proposal, self.validator
            )
        else:
            proposal.test_execution = TestExecutionReport(
                passed=False,
                executed=False,
                error="generated test suite failed AST validation",
            )

        baseline = (
            self.baseline_runner.run()
            if self.policy.run_baseline_tests
            else TestExecutionReport(
                passed=True,
                executed=False,
                skipped=True,
                error="baseline regression command not enabled",
            )
        )
        proposal.regression_execution = baseline
        tests_passed = bool(
            test_report.passed
            and proposal.test_execution
            and proposal.test_execution.passed
            and baseline.passed
            and (code_report.tests_passed is not False)
        )
        errors = list(code_report.errors)
        if not test_report.passed:
            errors.extend(f"generated test: {item}" for item in test_report.errors)
        if proposal.test_execution and not proposal.test_execution.passed:
            errors.append(proposal.test_execution.error or "generated tests failed")
        if not baseline.passed:
            errors.append(baseline.error or "baseline regression tests failed")
        warnings = list(code_report.warnings)
        if baseline.skipped:
            warnings.append(baseline.error or "baseline regression tests skipped")
        report = ValidationReport(
            passed=bool(
                code_report.syntax_valid
                and code_report.policy_valid
                and tests_passed
                and not errors
            ),
            syntax_valid=code_report.syntax_valid,
            policy_valid=code_report.policy_valid and test_report.policy_valid,
            tests_passed=tests_passed,
            errors=errors,
            warnings=warnings,
        )
        proposal.validation = report
        if not report.policy_valid:
            self.penalty_box.trigger("strict AST policy violation")
            self._publish("CODE_PENALTY_BOX", self.penalty_box.to_dict())
        if report.passed:
            proposal.status = ProposalStatus.VALIDATED
        else:
            proposal.status = ProposalStatus.REJECTED
            proposal.diagnosis["validation_mistake"] = errors[-10:]
        self._publish("CODE_VALIDATED", {"proposal": proposal.to_dict()})
        return report

    def approve_proposal(
        self,
        proposal: CodeProposal | str,
        *,
        reason: str = "",
        manual: bool = False,
    ) -> CodeProposal:
        proposal = self._get(proposal)
        if proposal.status is not ProposalStatus.VALIDATED and not (
            manual and proposal.status is ProposalStatus.PENDING_APPROVAL
        ):
            proposal.status = ProposalStatus.REJECTED
            proposal.approval_reason = "proposal must pass validation before approval"
            return proposal
        if (
            proposal.risk is ChangeRisk.LOW
            and self.policy.auto_approve_low_risk
            and not manual
        ):
            proposal.status = ProposalStatus.APPROVED
            proposal.approval_reason = (
                reason or "auto-approved: validated low-risk artifact"
            )
            self._publish(
                "CODE_APPROVED", {"proposal": proposal.to_dict(), "automatic": True}
            )
            return proposal
        if manual:
            proposal.status = ProposalStatus.APPROVED
            proposal.approval_reason = (
                reason
                or "manually approved after review; live source remains protected"
            )
            self._publish(
                "CODE_APPROVED", {"proposal": proposal.to_dict(), "automatic": False}
            )
            return proposal
        proposal.status = ProposalStatus.PENDING_APPROVAL
        proposal.approval_reason = (
            reason or "manual approval required for protected or elevated-risk change"
        )
        self._publish("CODE_PENDING_APPROVAL", {"proposal": proposal.to_dict()})
        return proposal

    def apply_proposal(
        self,
        proposal: CodeProposal | str,
        *,
        module: Any = None,
        manual: bool = False,
    ) -> CodeProposal:
        proposal = self._get(proposal)
        if proposal.status is not ProposalStatus.APPROVED:
            return proposal
        if proposal.risk is not ChangeRisk.LOW and not manual:
            proposal.status = ProposalStatus.PENDING_APPROVAL
            proposal.approval_reason = (
                "source/execution-affecting changes cannot be auto-applied"
            )
            return proposal
        if (
            proposal.risk is ChangeRisk.LOW
            and not manual
            and not self.policy.auto_apply_low_risk
        ):
            proposal.status = ProposalStatus.PENDING_APPROVAL
            proposal.approval_reason = "automatic application disabled by policy"
            return proposal
        if self._applied_count >= 100:
            proposal.status = ProposalStatus.PENDING_APPROVAL
            proposal.approval_reason = "process application limit reached"
            return proposal

        try:
            self._write_artifact(proposal)
            # The manifest is the approval boundary.  A deployment can review
            # or promote artifacts separately; live code is never overwritten.
            self._append_manifest(proposal)
            # Approval persists the inert artifact only.  Runtime parameters
            # are applied to a shadow clone and can reach the active module
            # only after shadow and gold-set promotion.
            proposal.status = ProposalStatus.APPLIED
            proposal.applied_at = float(self.clock())
            self._applied_count += 1
            self._publish(
                "CODE_APPLIED",
                {"proposal": proposal.to_dict(), "live_source_modified": False},
            )
        except OSError as exc:
            proposal.status = ProposalStatus.REJECTED
            proposal.approval_reason = f"artifact write failed: {exc}"
        return proposal

    # Intuitive aliases used by maintenance integrations.
    create_proposal = generate_proposal

    def validate_and_approve(
        self,
        proposal: CodeProposal | str,
        *,
        tests: Optional[Callable[[str], Any]] = None,
    ) -> CodeProposal:
        self.validate_proposal(proposal, tests=tests)
        return self.approve_proposal(proposal)

    apply_change = apply_proposal

    def promote_proposal(
        self, proposal: CodeProposal | str, *, reviewer: str = "human"
    ) -> CodeProposal:
        """Manually approve and store a protected artifact for review.

        Promotion still never overwrites live source or imports the result.
        """
        proposal = self._get(proposal)
        self.approve_proposal(
            proposal, reason=f"manual reviewer: {reviewer}", manual=True
        )
        return self.apply_proposal(proposal, manual=True)

    def run_for_module(
        self,
        module: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
        regime_parameters: Optional[Mapping[str, Any]] = None,
        tests: Optional[Callable[[str], Any]] = None,
        apply: bool = True,
    ) -> Dict[str, Any]:
        proposal = self.generate_proposal(
            module, context=context, regime_parameters=regime_parameters
        )
        report = self.validate_proposal(proposal, tests=tests)
        if report.passed:
            self.approve_proposal(proposal)
            if apply:
                self.apply_proposal(proposal, module=module)
        return proposal.to_dict()

    # NEW: Full autonomous self-fix + approve loop for any caller
    def autonomous_self_code_cycle(
        self,
        module: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
        apply: bool = True,
        auto_learn: bool = True,
    ) -> Dict[str, Any]:
        """Complete autonomous cycle: diagnose → generate → validate → auto-approve → apply → learn mistakes."""
        context = dict(context or {})
        context["autonomous_cycle"] = True
        
        proposal = self.generate_proposal(module, context=context)
        validation = self.validate_proposal(proposal)
        
        approved = False
        if validation.passed:
            approved_prop = self.approve_proposal(proposal)
            approved = approved_prop.status == ProposalStatus.APPROVED
        
        applied = False
        if approved and apply and proposal.risk == ChangeRisk.LOW:
            applied_prop = self.apply_proposal(proposal, module=module)
            applied = applied_prop.status == ProposalStatus.APPLIED
        
        result = proposal.to_dict()
        result.update({
            "autonomous": True,
            "validated": validation.passed,
            "approved": approved,
            "applied": applied,
            "organism_unity": True,
        })
        
        if auto_learn and hasattr(module, "learn_from_mistakes"):
            try:
                module.learn_from_mistakes(result.get("diagnosis", {}))
            except Exception:
                pass
        
        return result

    def auto_fix(
        self,
        module: Any,
        *,
        reason: str = "",
        context: Optional[Mapping[str, Any]] = None,
        apply: bool = True,
    ) -> Dict[str, Any]:
        """Attempt in-memory repair, then create a governed fix artifact."""
        repair = self.attempt_auto_repair(module, reason=reason)
        if repair.get("repaired"):
            return {"repair": repair, "artifact": None, "fixed": True}
        artifact = self.run_for_module(
            module, context=context or {"reason": reason}, apply=apply
        )
        return {
            "repair": repair,
            "artifact": artifact,
            # An artifact is a candidate, not proof that the running object is
            # fixed.  Keep the failed module isolated until a reviewed
            # promotion/restart occurs.
            "fixed": False,
            "artifact_applied": artifact.get("status") == ProposalStatus.APPLIED.value,
        }

    def attempt_auto_repair(self, module: Any, reason: str = "") -> Dict[str, Any]:
        """Try a module-provided repair/reinitialize hook, never rewrite source."""

        result: Dict[str, Any] = {
            "module": getattr(module, "module_name", module.__class__.__name__),
            "repaired": False,
            "reason": reason,
        }
        repair = getattr(module, "repair", None)
        module_type = type(module)
        has_custom_repair = "repair" in getattr(module_type, "__dict__", {})
        has_custom_initialize = "initialize" in getattr(module_type, "__dict__", {})
        try:
            if callable(repair) and has_custom_repair:
                value = repair(reason=reason)
                result["repaired"] = bool(value is not False)
                result["method"] = "repair"
            else:
                initialize = getattr(module, "initialize", None)
                if callable(initialize) and has_custom_initialize:
                    result["repaired"] = bool(initialize())
                    result["method"] = "initialize"
                else:
                    result["method"] = "no_custom_repair_hook"
            if result["repaired"]:
                health = getattr(module, "health", None)
                if health is not None:
                    health.status = "ok"
                    health.last_error = None
                self._publish("MODULE_REPAIRED", result)
        except Exception as exc:
            result["error"] = str(exc)
            self._publish("MODULE_REPAIR_FAILED", result)
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "proposals": len(self.proposals),
            "applied": self._applied_count,
            "pending": sum(
                1
                for item in self.proposals.values()
                if item.status is ProposalStatus.PENDING_APPROVAL
            ),
            "auto_approve_low_risk": self.policy.auto_approve_low_risk,
            "auto_apply_low_risk": self.policy.auto_apply_low_risk,
            "require_tests": self.policy.require_tests,
            "run_baseline_tests": self.policy.run_baseline_tests,
            "penalty_box": self.penalty_box.to_dict(),
            "audit": self.audit_trail.status(),
        }

    # --------------------------------------------------------------- helpers
    def _get(self, proposal: CodeProposal | str) -> CodeProposal:
        if isinstance(proposal, CodeProposal):
            return proposal
        return self.proposals[str(proposal)]

    def _write_artifact(self, proposal: CodeProposal) -> None:
        destination = Path(proposal.artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(proposal.code, encoding="utf-8")
        temporary.replace(destination)

    def _write_test_artifact(self, proposal: CodeProposal) -> None:
        if not proposal.test_path:
            return
        destination = Path(proposal.test_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(proposal.test_code, encoding="utf-8")
        temporary.replace(destination)

    def _append_manifest(self, proposal: CodeProposal) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(proposal.to_dict(), sort_keys=True) + "\n")
            stream.flush()

    def _apply_runtime_parameters(
        self, module: Any, parameters: Mapping[str, Any]
    ) -> None:
        if module is None or not parameters:
            return
        apply_tuning = getattr(module, "apply_adaptive_parameters", None)
        if callable(apply_tuning):
            apply_tuning(
                {
                    key: value
                    for key, value in parameters.items()
                    if key in self._safe_parameter_names
                }
            )

    def _publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.audit_trail.record(event_type, payload, source="SelfCodingEngine")
        if self.event_bus is not None and hasattr(self.event_bus, "publish"):
            try:
                if hasattr(self.event_bus, "publish_async"):
                    self.event_bus.publish_async(
                        event_type,
                        payload,
                        source="SelfCodingEngine",
                    )
                else:
                    self.event_bus.publish(
                        event_type, payload, source="SelfCodingEngine"
                    )
            except Exception:
                pass


# Names used by callers that prefer an agent/manager vocabulary.
AutoCodingEngine = SelfCodingEngine
ModuleSelfCodingAgent = SelfCodingEngine
CodeValidator = SafeCodeValidator
TestGenerator = GeneratedTestSuite

__all__ = [
    "ApprovalPolicy",
    "AutoCodingEngine",
    "ChangeProposal",
    "ChangeRisk",
    "CodeProposal",
    "CodeValidator",
    "GeneratedTestRunner",
    "GeneratedTestSuite",
    "ModuleSelfCodingAgent",
    "PenaltyBox",
    "ProposalStatus",
    "RegressionTestRunner",
    "SafeCodeValidator",
    "SelfCodingEngine",
    "TestExecutionReport",
    "TestGenerator",
    "TestSuiteGenerator",
    "ValidationReport",
]
