"""
Self-Evolution Agent - Perpetual Ascension Engine

A fully autonomous, perpetually self-improving trading system that:
- Runs as a continuous innovation daemon
- Uses multi-agent debate (researcher → coder → critic → tester)
- Ingests cutting-edge research from arXiv/SSRN
- Generates and tests new alpha strategies
- Maintains a hall of fame baseline for comparison
- Enforces eternal guardrails for safety

Directives 17-21 Implementation:
17. Perpetual Self-Directed Innovation Engine
18. Autonomous Research Ingestion
19. Infinite Architectural Expansion
20. Self-Sustaining Performance Evolution
21. Eternal Safeguards for Infinite Growth
"""

import os
import sys
import json
import time
import logging
import hashlib
import difflib
import traceback
import threading
import subprocess
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque
import ast
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("self_evolution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfEvolutionAgent")

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not available. Daemon mode will use simple loop.")

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("GitPython not available. Using subprocess for git operations.")

try:
    from advanced_modules.bayesian_market_state import BayesianMarketState
    from advanced_modules.trade_scheduler import UtilityFrontierScheduler
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger.warning("Bayesian modules not available. IG fitness disabled.")

try:
    from advanced_modules.rlevolver import RLEvolver, RLAction
    RL_EVOLVER_AVAILABLE = True
except ImportError:
    RL_EVOLVER_AVAILABLE = False
    logger.warning("RL Evolver not available. Policy learning disabled.")


LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROK_API_KEY")
REPO_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(REPO_PATH, "evolution_log.json")
TASK_QUEUE_FILE = os.path.join(REPO_PATH, "task_queue.json")
HALL_OF_FAME_FILE = os.path.join(REPO_PATH, "hall_of_fame.json")
RESEARCH_PAPERS_URL = "http://export.arxiv.org/api/query?search_query=cat:q-fin.*+OR+cat:stat.ML&sortBy=submittedDate&sortOrder=descending&max_results=20"


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class EvolutionTask:
    id: str
    description: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    proposed_changes: List[Dict] = field(default_factory=list)
    test_results: Optional[Dict] = None
    metrics_improvement: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "proposed_changes": self.proposed_changes,
            "test_results": self.test_results,
            "metrics_improvement": self.metrics_improvement
        }


@dataclass
class PerformanceMetrics:
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    p_accept: float = 0.5
    confidence: float = 0.5
    info_gain: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
        
    def beats(self, other: 'PerformanceMetrics', threshold: float = 0.05) -> bool:
        """Check if this metrics beats another by threshold"""
        if other.sharpe_ratio == 0:
            return self.sharpe_ratio > 0
        improvement = (self.sharpe_ratio - other.sharpe_ratio) / abs(other.sharpe_ratio)
        return improvement >= threshold
        
    def compute_fitness(self, lambda_ig: float = 0.2) -> float:
        """
        Compute fitness score with information gain component.
        
        F = E[Sharpe] * E[p_accept] * E[confidence] + lambda_IG * E[IG]
        
        This balances profit-seeking with curiosity-driven exploration,
        rewarding agents that "learn faster" rather than just "earn more."
        
        Args:
            lambda_ig: Weight for information gain term (default 0.2)
            
        Returns:
            Fitness score
        """
        base_fitness = self.sharpe_ratio * self.p_accept * self.confidence
        ig_bonus = lambda_ig * self.info_gain
        return base_fitness + ig_bonus


@dataclass
class CodeChange:
    file_path: str
    original_content: str
    new_content: str
    change_type: str
    description: str
    diff: str = ""
    
    def compute_diff(self):
        original_lines = self.original_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines, 
            new_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}"
        )
        self.diff = ''.join(diff)


class VersionControl:
    """File-based version control for tracking changes"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / ".evolution_versions"
        self.versions_dir.mkdir(exist_ok=True)
        self.history_file = self.versions_dir / "history.json"
        self.history = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
        
    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def create_version(self, file_path: str, content: str, description: str) -> str:
        version_id = hashlib.sha256(
            f"{file_path}{datetime.now().isoformat()}{content}".encode()
        ).hexdigest()[:12]
        
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        backup_path = version_dir / Path(file_path).name
        with open(backup_path, 'w') as f:
            f.write(content)
            
        version_entry = {
            "version_id": version_id,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "backup_path": str(backup_path)
        }
        
        self.history.append(version_entry)
        self._save_history()
        
        logger.info(f"Created version {version_id} for {file_path}")
        return version_id
        
    def rollback(self, version_id: str) -> bool:
        for entry in self.history:
            if entry["version_id"] == version_id:
                backup_path = Path(entry["backup_path"])
                if backup_path.exists():
                    with open(backup_path, 'r') as f:
                        content = f.read()
                    with open(entry["file_path"], 'w') as f:
                        f.write(content)
                    logger.info(f"Rolled back {entry['file_path']} to version {version_id}")
                    return True
        return False


class MultiAgentDebateSystem:
    """
    Multi-agent debate system for rigorous code generation.
    
    Agents:
    - Researcher: Proposes new high-signal ideas
    - Coder: Implements the proposals
    - Critic: Reviews and critiques the code
    - Tester: Validates through automated testing
    """
    
    GROK_BASE_URL = "https://api.x.ai/v1"
    DEFAULT_MODEL = "grok-beta"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    
    def __init__(self, llm_api_key: Optional[str] = None):
        self.api_key = llm_api_key or LLM_API_KEY
        self.available = self.api_key is not None
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_failure_count = 0
        
    def _call_llm_with_retry(self, prompt: str, system_prompt: str = "") -> str:
        """Call LLM API with exponential backoff retry"""
        last_error = None
        backoff = self.INITIAL_BACKOFF
        
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._call_llm_internal(prompt, system_prompt)
                self.llm_success_count += 1
                return result
            except Exception as e:
                last_error = e
                self.llm_failure_count += 1
                logger.warning(f"LLM call attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                
                if attempt < self.MAX_RETRIES - 1:
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    
        logger.error(f"All LLM retry attempts failed. Last error: {last_error}")
        return self._template_response(prompt)
        
    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Call LLM API with retry logic"""
        self.llm_call_count += 1
        
        if not self.available:
            return self._template_response(prompt)
            
        return self._call_llm_with_retry(prompt, system_prompt)
        
    def _call_llm_internal(self, prompt: str, system_prompt: str = "") -> str:
        """Internal LLM call using Grok API (OpenAI compatible)"""
        import openai
        
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.GROK_BASE_URL
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.DEFAULT_MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
            
    def _template_response(self, prompt: str) -> str:
        """Template-based fallback when LLM unavailable"""
        if "order flow" in prompt.lower():
            return self._order_flow_template()
        elif "regime" in prompt.lower():
            return self._regime_template()
        elif "arbitrage" in prompt.lower():
            return self._arbitrage_template()
        elif "absorption" in prompt.lower():
            return self._absorption_template()
        else:
            return f"# Template response for: {prompt[:100]}\npass"
            
    def researcher_agent(self, context: str) -> str:
        """Research agent proposes new high-signal ideas"""
        system_prompt = """You are a quantitative researcher at a top hedge fund. 
        Only propose ideas that are:
        1. Mathematically rigorous and testable
        2. Based on microstructure, statistical arbitrage, or proven quantitative methods
        3. High-signal with clear edge hypothesis
        Never propose pseudoscience or unproven concepts."""
        
        prompt = f"""Based on the following context, propose a rigorous new high-signal trading idea:

Context: {context}

Provide:
1. Clear hypothesis
2. Mathematical foundation
3. Data requirements
4. Expected edge mechanism
5. Risk considerations"""

        return self._call_llm(prompt, system_prompt)
        
    def coder_agent(self, proposal: str) -> str:
        """Coder agent implements the proposal"""
        system_prompt = """You are an expert Python developer specializing in quantitative trading systems.
        Write clean, efficient, well-documented code.
        Follow existing code conventions.
        Never duplicate existing functionality.
        Include proper error handling."""
        
        prompt = f"""Implement the following proposal as a Python module:

Proposal: {proposal}

Requirements:
1. Clean, production-ready Python code
2. Proper type hints and docstrings
3. Integration with existing system architecture
4. Comprehensive error handling
5. Unit test suggestions"""

        return self._call_llm(prompt, system_prompt)
        
    def critic_agent(self, code: str, proposal: str) -> Dict[str, Any]:
        """Critic agent reviews and critiques the code"""
        system_prompt = """You are a senior code reviewer with expertise in quantitative finance.
        Be thorough but constructive.
        Focus on: correctness, efficiency, security, maintainability."""
        
        prompt = f"""Review this code implementation:

Original Proposal: {proposal}

Code:
```python
{code}
```

Provide:
1. Bugs or logical errors found
2. Security concerns
3. Performance issues
4. Suggestions for improvement
5. Overall assessment (APPROVE/REVISE/REJECT)"""

        response = self._call_llm(prompt, system_prompt)
        
        approved = "APPROVE" in response.upper()
        
        return {
            "review": response,
            "approved": approved,
            "needs_revision": "REVISE" in response.upper(),
            "rejected": "REJECT" in response.upper()
        }
        
    def tester_agent(self, base_dir: str) -> Dict[str, Any]:
        """Tester agent runs automated tests"""
        result = {
            "passed": False,
            "metrics": PerformanceMetrics().to_dict(),
            "test_output": "",
            "errors": []
        }
        
        try:
            test_script = os.path.join(base_dir, "run_comprehensive_test.py")
            if os.path.exists(test_script):
                proc = subprocess.run(
                    [sys.executable, test_script],
                    cwd=base_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                result["test_output"] = proc.stdout
                result["passed"] = proc.returncode == 0 and "ALL TESTS PASSED" in proc.stdout
                
                if proc.stderr:
                    result["errors"].append(proc.stderr)
                    
        except subprocess.TimeoutExpired:
            result["errors"].append("Test timeout exceeded")
        except Exception as e:
            result["errors"].append(str(e))
            
        return result
        
    def _order_flow_template(self) -> str:
        return '''
class OrderFlowImbalanceDetector:
    """Detects order flow imbalances for alpha generation"""
    
    def __init__(self, lookback: int = 100, threshold: float = 0.6):
        self.lookback = lookback
        self.threshold = threshold
        
    def calculate_imbalance(self, bids: list, asks: list) -> float:
        total_bids = sum(bids)
        total_asks = sum(asks)
        total = total_bids + total_asks
        if total == 0:
            return 0.0
        return (total_bids - total_asks) / total
        
    def generate_signal(self, imbalance: float) -> str:
        if imbalance > self.threshold:
            return "BUY"
        elif imbalance < -self.threshold:
            return "SELL"
        return "NEUTRAL"
'''

    def _regime_template(self) -> str:
        return '''
class RegimeDetector:
    """Multi-timeframe regime detection"""
    
    def __init__(self, vol_lookback: int = 20):
        self.vol_lookback = vol_lookback
        
    def detect_regime(self, returns: list) -> str:
        import numpy as np
        vol = np.std(returns[-self.vol_lookback:])
        trend = np.mean(returns[-self.vol_lookback:])
        
        if vol > 0.02:
            return "HIGH_VOLATILITY"
        elif trend > 0.001:
            return "TRENDING_UP"
        elif trend < -0.001:
            return "TRENDING_DOWN"
        return "RANGING"
'''

    def _arbitrage_template(self) -> str:
        return '''
class StatisticalArbitrageDetector:
    """Pairs trading with cointegration analysis"""
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        
    def calculate_spread(self, series1: list, series2: list, hedge_ratio: float) -> list:
        import numpy as np
        s1 = np.array(series1)
        s2 = np.array(series2)
        return (s1 - hedge_ratio * s2).tolist()
        
    def generate_signal(self, spread: list, threshold: float = 2.0) -> str:
        import numpy as np
        z_score = (spread[-1] - np.mean(spread)) / np.std(spread)
        if z_score > threshold:
            return "SHORT_SPREAD"
        elif z_score < -threshold:
            return "LONG_SPREAD"
        return "NEUTRAL"
'''

    def _absorption_template(self) -> str:
        return '''
class AbsorptionClusterDetector:
    """Detects absorption and exhaustion patterns"""
    
    def __init__(self, volume_threshold: float = 2.0):
        self.volume_threshold = volume_threshold
        
    def detect_absorption(self, prices: list, volumes: list) -> dict:
        import numpy as np
        avg_vol = np.mean(volumes)
        price_change = abs(prices[-1] - prices[-2]) if len(prices) > 1 else 0
        
        is_absorption = (volumes[-1] > avg_vol * self.volume_threshold and 
                        price_change < np.std(np.diff(prices)) * 0.5)
        
        return {
            "detected": is_absorption,
            "volume_ratio": volumes[-1] / avg_vol if avg_vol > 0 else 0,
            "price_stability": price_change
        }
'''


class ResearchIngestionModule:
    """
    Autonomous research ingestion from arXiv and SSRN.
    
    Fetches latest papers, summarizes them, and proposes implementations
    for promising ideas.
    """
    
    ARXIV_API = "http://export.arxiv.org/api/query"
    CATEGORIES = ["q-fin.TR", "q-fin.PM", "q-fin.ST", "stat.ML", "cs.LG"]
    
    def __init__(self, llm_api_key: Optional[str] = None):
        self.api_key = llm_api_key or LLM_API_KEY
        self.papers_cache: List[Dict] = []
        
    def fetch_recent_papers(self, max_results: int = 20) -> List[Dict]:
        """Fetch recent papers from arXiv"""
        papers = []
        
        try:
            query = "+OR+".join([f"cat:{cat}" for cat in self.CATEGORIES])
            url = f"{self.ARXIV_API}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {
                    "title": entry.find('atom:title', ns).text.strip(),
                    "summary": entry.find('atom:summary', ns).text.strip(),
                    "published": entry.find('atom:published', ns).text,
                    "authors": [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
                    "link": entry.find('atom:id', ns).text
                }
                papers.append(paper)
                
            self.papers_cache = papers
            logger.info(f"Fetched {len(papers)} papers from arXiv")
            
        except Exception as e:
            logger.error(f"Failed to fetch papers: {e}")
            
        return papers
        
    def evaluate_paper(self, paper: Dict) -> Dict[str, Any]:
        """Evaluate if a paper is worth implementing"""
        keywords_positive = [
            "order flow", "market microstructure", "high frequency",
            "statistical arbitrage", "pairs trading", "cointegration",
            "regime detection", "volatility", "momentum", "mean reversion",
            "machine learning", "neural network", "reinforcement learning"
        ]
        
        keywords_negative = [
            "quantum", "sentiment", "social media", "news", "nlp",
            "cryptocurrency prediction", "bitcoin forecast"
        ]
        
        text = (paper["title"] + " " + paper["summary"]).lower()
        
        positive_score = sum(1 for kw in keywords_positive if kw in text)
        negative_score = sum(1 for kw in keywords_negative if kw in text)
        
        score = positive_score - negative_score * 2
        
        return {
            "paper": paper,
            "score": score,
            "recommend_implementation": score >= 2,
            "positive_keywords": [kw for kw in keywords_positive if kw in text],
            "negative_keywords": [kw for kw in keywords_negative if kw in text]
        }
        
    def generate_implementation_task(self, paper: Dict) -> Optional[str]:
        """Generate implementation task from paper"""
        evaluation = self.evaluate_paper(paper)
        
        if not evaluation["recommend_implementation"]:
            return None
            
        task_description = f"""Implement trading strategy based on research paper:
Title: {paper['title']}
Key concepts: {', '.join(evaluation['positive_keywords'])}
Summary: {paper['summary'][:500]}..."""

        return task_description


class HallOfFame:
    """
    Maintains baseline performance metrics for comparison.
    
    New strategies must beat the hall of fame by a threshold
    to be integrated into the system.
    """
    
    def __init__(self, file_path: str = HALL_OF_FAME_FILE):
        self.file_path = file_path
        self.data = self._load()
        
    def _load(self) -> Dict:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {
            "baseline_metrics": PerformanceMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                max_drawdown=0.15,
                win_rate=0.55
            ).to_dict(),
            "strategies": [],
            "last_updated": datetime.now().isoformat()
        }
        
    def _save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)
            
    def get_baseline(self) -> PerformanceMetrics:
        """Get baseline metrics"""
        m = self.data["baseline_metrics"]
        return PerformanceMetrics(**m)
        
    def update_baseline(self, metrics: PerformanceMetrics):
        """Update baseline if metrics are better"""
        current = self.get_baseline()
        if metrics.beats(current, threshold=0.05):
            self.data["baseline_metrics"] = metrics.to_dict()
            self._save()
            logger.info("Hall of Fame baseline updated")
            
    def add_strategy(self, name: str, metrics: PerformanceMetrics, code_hash: str):
        """Add strategy to hall of fame"""
        self.data["strategies"].append({
            "name": name,
            "metrics": metrics.to_dict(),
            "code_hash": code_hash,
            "added_at": datetime.now().isoformat()
        })
        self._save()


class SelfEvolutionAgent:
    """
    Main perpetual self-evolution agent.
    
    Features:
    - Multi-agent debate for code generation
    - Autonomous research ingestion
    - Hall of fame baseline comparison
    - Perpetual innovation daemon
    - Eternal safety guardrails
    - Self/market analysis for autonomous adaptation
    - Loss prevention and proactive risk management
    
    Daemon Self-Improvement Mandate:
    The agent must prioritize:
    1. Deep anomaly/pattern discovery in all raw data streams
    2. Auto-generation of novel proprietary signals only AI-scale iteration can uncover
    3. Real-time loss forecasting and preemptive adaptation
    4. Weekly full-system audit + radical innovation if edge stagnant
    """
    
    SAFETY_GUARDS = {
        "max_changes_per_cycle": 10,
        "require_test_pass": True,
        "require_baseline_improvement": True,
        "improvement_threshold": 0.05,
        "forbidden_patterns": [
            r"os\.system",
            r"subprocess\.call.*shell=True",
            r"eval\(",
            r"exec\(",
            r"__import__\(",
            r"open\(.*/etc/",
            r"rm\s+-rf",
        ],
        "max_single_trade_risk": 0.03,
        "require_human_override_for_live": True
    }
    
    TASK_PRIORITY_ORDER = [
        "loss_prevention",
        "risk_management",
        "signal_generation",
        "research_implementation",
        "innovation",
        "radical_innovation"
    ]
    
    def __init__(self, 
                 base_dir: str = None,
                 llm_api_key: Optional[str] = None,
                 auto_apply: bool = False,
                 enable_rl: bool = True):
        self.base_dir = Path(base_dir or REPO_PATH)
        self.auto_apply = auto_apply
        
        self.version_control = VersionControl(str(self.base_dir))
        self.debate_system = MultiAgentDebateSystem(llm_api_key)
        self.research_module = ResearchIngestionModule(llm_api_key)
        self.hall_of_fame = HallOfFame()
        
        self.task_queue: deque = deque()
        self.completed_tasks: List[EvolutionTask] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._scheduler = None
        
        self.rl_evolver: Optional['RLEvolver'] = None
        if enable_rl and RL_EVOLVER_AVAILABLE:
            try:
                checkpoint_path = str(self.base_dir / "models" / "rl_checkpoint.zip")
                self.rl_evolver = RLEvolver(checkpoint_path=checkpoint_path)
                self.rl_evolver.set_task_queue(self.task_queue)
                logger.info("RL Evolver initialized and connected to task queue")
            except Exception as e:
                logger.warning(f"Failed to initialize RL Evolver: {e}")
                self.rl_evolver = None
        
        self.metrics = {
            "cycles_completed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "changes_applied": 0,
            "baseline_improvements": 0,
            "last_cycle": None,
            "daemon_started": None,
            "rl_steps": 0,
            "rl_proposals": 0
        }
        
        self._load_state()
        
        logger.info(f"SelfEvolutionAgent initialized at {self.base_dir}")
        
    def _load_state(self):
        """Load agent state from file"""
        state_file = self.base_dir / ".evolution_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.metrics = state.get("metrics", self.metrics)
                
                for task_data in state.get("pending_tasks", []):
                    task = EvolutionTask(
                        id=task_data["id"],
                        description=task_data["description"],
                        task_type=task_data["task_type"],
                        priority=TaskPriority[task_data["priority"]],
                        status=TaskStatus(task_data["status"])
                    )
                    self.task_queue.append(task)
                    
    def _save_state(self):
        """Save agent state to file"""
        state_file = self.base_dir / ".evolution_state.json"
        state = {
            "metrics": self.metrics,
            "pending_tasks": [t.to_dict() for t in self.task_queue],
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def _log_evolution(self, entry: Dict):
        """Log evolution activity"""
        log_file = self.base_dir / "evolution_log.json"
        
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
                
        entry["timestamp"] = datetime.now().isoformat()
        logs.append(entry)
        
        logs = logs[-1000:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
    def add_task(self, 
                 description: str, 
                 task_type: str = "improvement",
                 priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Add a new task to the queue"""
        task_id = hashlib.sha256(
            f"{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        task = EvolutionTask(
            id=task_id,
            description=description,
            task_type=task_type,
            priority=priority
        )
        
        self.task_queue.append(task)
        self._save_state()
        
        logger.info(f"Added task {task_id}: {description[:50]}...")
        return task_id
        
    def parse_logs_for_drawdown(self) -> float:
        """Parse recent logs/metrics for drawdown information"""
        try:
            risk_manager_path = self.base_dir / "risk" / "institutional_risk_manager.py"
            metrics_file = self.base_dir / "performance_metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    return metrics.get("max_drawdown", 0.0)
                    
            log_file = self.base_dir / "evolution_log.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    for entry in reversed(logs[-50:]):
                        if "drawdown" in str(entry).lower():
                            return entry.get("drawdown", 0.0)
        except Exception as e:
            logger.warning(f"Could not parse drawdown: {e}")
        return 0.0
        
    def get_current_regime(self) -> str:
        """Get current market regime from regime_detection module"""
        try:
            sys.path.insert(0, str(self.base_dir))
            from advanced_modules.regime_detection import RegimeDetector
            
            detector = RegimeDetector()
            import numpy as np
            sample_returns = np.random.randn(100) * 0.01
            regime = detector.detect_regime(sample_returns.tolist())
            return regime
        except Exception as e:
            logger.warning(f"Could not get regime: {e}")
            return "UNKNOWN"
            
    def detect_anomalies(self) -> List[str]:
        """Detect anomalies in recent data streams"""
        anomalies = []
        try:
            log_file = self.base_dir / "evolution_log.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    
                recent_failures = sum(1 for l in logs[-20:] if l.get("event") == "task_failed")
                if recent_failures > 5:
                    anomalies.append(f"High task failure rate: {recent_failures}/20")
                    
                cycle_times = [l.get("duration_seconds", 0) for l in logs[-10:] if "duration_seconds" in l]
                if cycle_times and max(cycle_times) > 300:
                    anomalies.append(f"Slow cycle detected: {max(cycle_times):.1f}s")
                    
        except Exception as e:
            logger.warning(f"Anomaly detection error: {e}")
            
        return anomalies
        
    def self_market_analysis(self) -> str:
        """Analyze recent performance and market data for autonomous adaptation"""
        recent_drawdown = self.parse_logs_for_drawdown()
        market_regime = self.get_current_regime()
        anomalies = self.detect_anomalies()
        
        analysis = f"Recent drawdown: {recent_drawdown:.2%}. Regime: {market_regime}."
        if anomalies:
            analysis += f" Anomalies: {', '.join(anomalies)}"
        else:
            analysis += " No anomalies detected."
            
        return analysis
        
    def auto_loss_prevention_task(self):
        """Automatically generate loss prevention tasks based on analysis"""
        analysis = self.self_market_analysis()
        
        drawdown = self.parse_logs_for_drawdown()
        regime = self.get_current_regime()
        
        if drawdown > 0.05:
            self.add_task(
                f"Auto-fix loss streak: Evolve new adaptive signal/countermeasure for drawdown {drawdown:.2%}",
                "loss_prevention",
                TaskPriority.CRITICAL
            )
            self.add_task(
                "Tighten circuit breakers and re-optimize position sizing",
                "risk_management",
                TaskPriority.CRITICAL
            )
            
        if "SHIFT" in regime.upper() or "HIGH_VOLATILITY" in regime.upper():
            self.add_task(
                f"Adapt strategy to current regime: {regime}",
                "regime_adaptation",
                TaskPriority.HIGH
            )
            
        if self.metrics.get("tasks_failed", 0) > self.metrics.get("tasks_completed", 1) * 0.5:
            self.add_task(
                "Self-analysis: Review code/logs for systematic weaknesses",
                "self_improvement",
                TaskPriority.HIGH
            )
            
        stagnant_cycles = 0
        if self.metrics.get("baseline_improvements", 0) == 0 and self.metrics.get("cycles_completed", 0) > 3:
            stagnant_cycles = self.metrics["cycles_completed"]
            
        if stagnant_cycles >= 3:
            self.add_task(
                "Radical innovation: Propose entirely new architectural layer or signal class",
                "radical_innovation",
                TaskPriority.HIGH
            )
            
        logger.info(f"Loss prevention analysis: {analysis}")
        
    def generate_autonomous_tasks(self):
        """Generate new tasks autonomously based on system state"""
        analysis = self.self_market_analysis()
        
        autonomous_tasks = [
            ("Discover new microstructural edge from recent order flow anomalies", "innovation"),
            ("Evolve improved absorption cluster detector using genetic programming", "innovation"),
            ("Propose funding rate cross-asset arbitrage strategy", "innovation"),
            ("Optimize regime detection thresholds based on recent performance", "optimization"),
            ("Add new statistical arbitrage pair based on correlation analysis", "innovation"),
            ("Implement adaptive position sizing based on regime", "improvement"),
            ("Create micro-price calculation from order book depth", "innovation"),
            ("Add spoofing detection via order book dynamics", "innovation"),
            (f"Self-analysis: Review my own code/logs for weaknesses and propose upgrades", "self_improvement"),
            (f"Market adaptation: Generate new proprietary data/signal to exploit current {analysis[:100]}", "market_adaptation"),
            ("Invent novel loss-avoidance layer (e.g., predictive drawdown forecaster)", "loss_prevention"),
            ("Cross-validate all strategies and diversify further", "validation"),
        ]
        
        for description, task_type in autonomous_tasks:
            existing = any(t.description == description for t in self.task_queue)
            if not existing:
                self.add_task(description, task_type, TaskPriority.MEDIUM)
                
        logger.info(f"Generated {len(autonomous_tasks)} autonomous tasks")
        
    def ingest_research(self):
        """Ingest and evaluate recent research papers"""
        papers = self.research_module.fetch_recent_papers(max_results=10)
        
        tasks_added = 0
        for paper in papers:
            task_desc = self.research_module.generate_implementation_task(paper)
            if task_desc:
                self.add_task(task_desc, "research_implementation", TaskPriority.LOW)
                tasks_added += 1
                
        logger.info(f"Ingested {len(papers)} papers, added {tasks_added} implementation tasks")
        
    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate generated code against safety rules"""
        for pattern in self.SAFETY_GUARDS["forbidden_patterns"]:
            if re.search(pattern, code):
                return False, f"Forbidden pattern found: {pattern}"
                
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
            
        return True, "Valid"
        
    def process_task_with_debate(self, task: EvolutionTask) -> bool:
        """Process task using multi-agent debate"""
        task.status = TaskStatus.IN_PROGRESS
        logger.info(f"Processing task {task.id} with multi-agent debate")
        
        try:
            context = f"Task: {task.description}\nType: {task.task_type}"
            proposal = self.debate_system.researcher_agent(context)
            logger.info(f"Researcher proposed: {proposal[:100]}...")
            
            code = self.debate_system.coder_agent(proposal)
            logger.info(f"Coder generated {len(code)} chars of code")
            
            is_valid, validation_msg = self._validate_code(code)
            if not is_valid:
                task.status = TaskStatus.REJECTED
                task.error = f"Code validation failed: {validation_msg}"
                self._generate_diagnostic_task(task, "validation", validation_msg)
                return False
                
            review = self.debate_system.critic_agent(code, proposal)
            logger.info(f"Critic review: approved={review['approved']}")
            
            if review["rejected"]:
                task.status = TaskStatus.REJECTED
                rejection_reason = review['review']
                task.error = f"Critic rejected: {rejection_reason[:200]}"
                
                self._log_evolution({
                    "event": "critic_rejection",
                    "task_id": task.id,
                    "reason": rejection_reason,
                    "proposal": proposal[:500],
                    "code_snippet": code[:500],
                    "suggestions": self._extract_suggestions(rejection_reason)
                })
                logger.warning(f"Critic rejection details for {task.id}:")
                logger.warning(f"  Reason: {rejection_reason[:300]}")
                logger.warning(f"  Suggestions: {self._extract_suggestions(rejection_reason)}")
                
                self._generate_diagnostic_task(task, "critic_rejection", rejection_reason)
                return False
                
            test_results = self.debate_system.tester_agent(str(self.base_dir))
            task.test_results = test_results
            
            if self.SAFETY_GUARDS["require_test_pass"] and not test_results["passed"]:
                task.status = TaskStatus.FAILED
                task.error = "Tests did not pass"
                self._generate_diagnostic_task(task, "test_failure", str(test_results.get("errors", [])))
                return False
                
            if self.SAFETY_GUARDS["require_baseline_improvement"]:
                baseline = self.hall_of_fame.get_baseline()
                new_metrics = PerformanceMetrics(**test_results.get("metrics", {}))
                
                if not new_metrics.beats(baseline, self.SAFETY_GUARDS["improvement_threshold"]):
                    task.status = TaskStatus.REJECTED
                    task.error = "Does not beat baseline by required threshold"
                    return False
                    
            task.proposed_changes.append({
                "proposal": proposal,
                "code": code,
                "review": review,
                "test_results": test_results
            })
            
            if self.auto_apply and review["approved"]:
                self._apply_change(task, code)
                
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            self.metrics["tasks_completed"] += 1
            
            self._log_evolution({
                "event": "task_completed",
                "task_id": task.id,
                "description": task.description,
                "approved": review["approved"]
            })
            
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.id} failed: {e}")
            self._generate_diagnostic_task(task, "exception", str(e))
            return False
            
    def _generate_diagnostic_task(self, failed_task: EvolutionTask, failure_type: str, error_details: str):
        """Generate a self-diagnostic task based on failure"""
        diagnostic_descriptions = {
            "validation": f"Fix code validation error: {error_details[:200]}",
            "critic_rejection": f"Address critic feedback: {error_details[:200]}",
            "test_failure": f"Fix test failures: {error_details[:200]}",
            "exception": f"Debug exception in task processing: {error_details[:200]}",
            "import_error": f"Fix import error: {error_details[:200]}"
        }
        
        description = diagnostic_descriptions.get(
            failure_type, 
            f"Diagnose and fix: {failure_type} - {error_details[:150]}"
        )
        
        self.add_task(
            description,
            "self_diagnostic",
            TaskPriority.HIGH
        )
        
        logger.info(f"Generated diagnostic task for {failure_type}: {description[:100]}...")
        
    def _extract_suggestions(self, review_text: str) -> List[str]:
        """Extract actionable suggestions from critic review"""
        suggestions = []
        
        suggestion_patterns = [
            r"suggest(?:ion)?[s]?:?\s*(.+?)(?:\.|$)",
            r"recommend[s]?:?\s*(.+?)(?:\.|$)",
            r"should\s+(.+?)(?:\.|$)",
            r"consider\s+(.+?)(?:\.|$)",
            r"improve[ment]?[s]?:?\s*(.+?)(?:\.|$)"
        ]
        
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, review_text.lower(), re.IGNORECASE)
            suggestions.extend([m.strip() for m in matches if len(m.strip()) > 10])
            
        return suggestions[:5]
            
    def _apply_change(self, task: EvolutionTask, code: str):
        """Apply code change to the system"""
        if "order flow" in task.description.lower():
            file_path = "advanced_modules/evolved_order_flow.py"
        elif "regime" in task.description.lower():
            file_path = "advanced_modules/evolved_regime.py"
        elif "arbitrage" in task.description.lower():
            file_path = "advanced_modules/evolved_arbitrage.py"
        else:
            file_path = f"advanced_modules/evolved_{task.id}.py"
            
        full_path = self.base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if full_path.exists():
            with open(full_path, 'r') as f:
                original = f.read()
            self.version_control.create_version(file_path, original, f"Before task {task.id}")
        
        with open(full_path, 'w') as f:
            f.write(code)
            
        self.metrics["changes_applied"] += 1
        logger.info(f"Applied change to {file_path}")
        
    def evolution_cycle(self):
        """
        Run a single evolution cycle.
        
        The agent must prioritize:
        1. Deep anomaly/pattern discovery in all raw data streams
        2. Auto-generation of novel proprietary signals only AI-scale iteration can uncover
        3. Real-time loss forecasting and preemptive adaptation
        4. Weekly full-system audit + radical innovation if edge stagnant
        5. RL policy learning from belief state and microstructure signals
        """
        cycle_start = datetime.now()
        logger.info("Starting evolution cycle...")
        
        self._log_evolution({
            "event": "cycle_start",
            "pending_tasks": len(self.task_queue)
        })
        
        self.auto_loss_prevention_task()
        
        self.generate_autonomous_tasks()
        
        self.ingest_research()
        
        self._run_rl_step()
        
        tasks_processed = 0
        max_tasks = self.SAFETY_GUARDS["max_changes_per_cycle"]
        
        while self.task_queue and tasks_processed < max_tasks:
            task = self.task_queue.popleft()
            
            success = self.process_task_with_debate(task)
            self.completed_tasks.append(task)
            tasks_processed += 1
            
            if success:
                logger.info(f"Task {task.id} completed successfully")
                self._record_rl_outcome(success=True)
            else:
                logger.warning(f"Task {task.id} failed: {task.error}")
                self._record_rl_outcome(success=False)
                
        self.metrics["cycles_completed"] += 1
        self.metrics["last_cycle"] = datetime.now().isoformat()
        self._save_state()
        
        if self.rl_evolver:
            self.rl_evolver.save_checkpoint()
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        rl_status = None
        if self.rl_evolver:
            rl_status = self.rl_evolver.get_status()
        
        self._log_evolution({
            "event": "cycle_complete",
            "tasks_processed": tasks_processed,
            "duration_seconds": cycle_duration,
            "rl_status": rl_status
        })
        
        logger.info(f"Evolution cycle complete. Processed {tasks_processed} tasks in {cycle_duration:.1f}s")
        
        return {
            "tasks_processed": tasks_processed,
            "duration": cycle_duration,
            "metrics": self.metrics,
            "rl_status": rl_status
        }
    
    def _run_rl_step(self):
        """Run RL evolver step to propose actions based on current belief state"""
        if not self.rl_evolver:
            return
            
        try:
            belief = self._get_current_belief_state()
            
            drawdown = self.parse_logs_for_drawdown()
            
            action = self.rl_evolver.step(
                belief=belief,
                pnl_delta=0.0,
                exposure=0.0,
                volatility=1.0,
                spread=0.0
            )
            
            self.rl_evolver.propose_action(action)
            
            self.metrics["rl_steps"] = self.rl_evolver.total_steps
            self.metrics["rl_proposals"] += 1
            
            logger.info(f"RL step {self.rl_evolver.total_steps}: {action.to_task_description()}")
            
        except Exception as e:
            logger.warning(f"RL step failed: {e}")
            
    def _get_current_belief_state(self) -> Dict[str, Any]:
        """Get current belief state for RL observation"""
        belief = {
            "p_accept": 0.5,
            "confidence": 0.5,
            "expected_ig_bits": 0.0,
            "regime": self.get_current_regime(),
            "mm_flags": {}
        }
        
        if BAYESIAN_AVAILABLE:
            try:
                market_state = BayesianMarketState()
                state = market_state.get_state()
                belief["p_accept"] = state.p_accept
                belief["confidence"] = state.confidence
                belief["expected_ig_bits"] = state.expected_ig_bits
                belief["regime"] = state.regime
                if state.mm_flags:
                    belief["mm_flags"] = state.mm_flags
            except Exception as e:
                logger.debug(f"Could not get Bayesian state: {e}")
                
        return belief
        
    def _record_rl_outcome(self, success: bool):
        """Record task outcome for RL reward computation"""
        if not self.rl_evolver:
            return
            
        try:
            pnl_delta = 10.0 if success else -5.0
            realized_ig = 0.1 if success else 0.02
            drawdown = 0.0 if success else 0.01
            
            self.rl_evolver.record_outcome(pnl_delta, realized_ig, drawdown)
        except Exception as e:
            logger.debug(f"Could not record RL outcome: {e}")
        
    def start_perpetual_daemon(self, interval_hours: float = 24, interval_minutes: float = None):
        """Start the perpetual innovation daemon
        
        Args:
            interval_hours: Interval in hours (default 24, ignored if interval_minutes set)
            interval_minutes: Interval in minutes (takes precedence over hours)
        """
        if self.running:
            logger.warning("Daemon already running")
            return
            
        self.running = True
        self.metrics["daemon_started"] = datetime.now().isoformat()
        
        if interval_minutes is not None:
            interval_seconds = interval_minutes * 60
            interval_display = f"{interval_minutes}m"
        else:
            interval_seconds = interval_hours * 3600
            interval_display = f"{interval_hours}h"
        
        logger.info(f"Starting perpetual innovation daemon (interval: {interval_display})")
        
        self._log_evolution({
            "event": "daemon_started",
            "interval_minutes": interval_minutes,
            "interval_hours": interval_hours,
            "interval_seconds": interval_seconds
        })
        
        if SCHEDULER_AVAILABLE:
            self._scheduler = BackgroundScheduler()
            
            if interval_minutes is not None:
                self._scheduler.add_job(
                    self.evolution_cycle, 
                    'interval', 
                    minutes=interval_minutes,
                    id='evolution_cycle'
                )
            else:
                self._scheduler.add_job(
                    self.evolution_cycle, 
                    'interval', 
                    hours=interval_hours,
                    id='evolution_cycle'
                )
            self._scheduler.start()
            
            logger.info("Perpetual innovation daemon activated with APScheduler")
            
            try:
                while self.running:
                    time.sleep(60)
            except KeyboardInterrupt:
                self.stop_daemon()
        else:
            def daemon_loop():
                while self.running:
                    try:
                        self.evolution_cycle()
                    except Exception as e:
                        logger.error(f"Cycle error: {e}")
                    time.sleep(interval_seconds)
                    
            self._thread = threading.Thread(target=daemon_loop, daemon=True)
            self._thread.start()
            
            logger.info("Perpetual innovation daemon activated with simple loop")
            
    def stop_daemon(self):
        """Stop the perpetual daemon"""
        self.running = False
        
        if self._scheduler:
            self._scheduler.shutdown()
            
        if self._thread:
            self._thread.join(timeout=10)
            
        self._log_evolution({"event": "daemon_stopped"})
        logger.info("Perpetual innovation daemon stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        status = {
            "running": self.running,
            "pending_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "metrics": self.metrics,
            "hall_of_fame_baseline": self.hall_of_fame.get_baseline().to_dict(),
            "auto_apply": self.auto_apply,
            "rl_evolver_enabled": self.rl_evolver is not None
        }
        
        if self.rl_evolver:
            status["rl_status"] = self.rl_evolver.get_status()
            
        return status
        
    def run_single_cycle_demo(self) -> Dict[str, Any]:
        """Run a single cycle for demonstration"""
        logger.info("Running single evolution cycle demo...")
        
        self.add_task(
            "Implement improved order flow imbalance detector",
            "innovation",
            TaskPriority.HIGH
        )
        
        result = self.evolution_cycle()
        
        return {
            "cycle_result": result,
            "status": self.get_status(),
            "recent_tasks": [t.to_dict() for t in self.completed_tasks[-5:]]
        }


def main():
    """Main entry point - starts perpetual daemon or runs demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Evolution Agent")
    parser.add_argument("--daemon", action="store_true", help="Start perpetual daemon")
    parser.add_argument("--interval", type=float, default=24, help="Daemon interval in hours")
    parser.add_argument("--interval-minutes", type=float, default=None, help="Daemon interval in minutes (overrides --interval)")
    parser.add_argument("--demo", action="store_true", help="Run single cycle demo")
    parser.add_argument("--auto-apply", action="store_true", help="Auto-apply approved changes")
    
    args = parser.parse_args()
    
    agent = SelfEvolutionAgent(auto_apply=args.auto_apply)
    
    if args.daemon:
        print("=" * 60)
        print("PERPETUAL ASCENSION ENGINE")
        print("Eternal Innovation Daemon Starting...")
        if args.interval_minutes:
            print(f"Cycle Interval: {args.interval_minutes} minutes")
        else:
            print(f"Cycle Interval: {args.interval} hours")
        print("=" * 60)
        agent.start_perpetual_daemon(
            interval_hours=args.interval,
            interval_minutes=args.interval_minutes
        )
    elif args.demo:
        print("=" * 60)
        print("SELF-EVOLUTION AGENT DEMO")
        print("=" * 60)
        result = agent.run_single_cycle_demo()
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=" * 60)
        print("SELF-EVOLUTION AGENT STATUS")
        print("=" * 60)
        status = agent.get_status()
        print(json.dumps(status, indent=2, default=str))
        print("\nUsage:")
        print("  --daemon            Start perpetual innovation daemon")
        print("  --interval HOURS    Daemon interval in hours (default: 24)")
        print("  --interval-minutes  Daemon interval in minutes (overrides --interval)")
        print("  --demo              Run single cycle demonstration")
        print("  --auto-apply        Auto-apply approved changes")


if __name__ == "__main__":
    main()
