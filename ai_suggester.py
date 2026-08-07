import openai
import os
import subprocess
from pathlib import Path

def get_suggestions():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    openai.api_key = api_key

    repo = Path(__file__).parent
    sources = []

    protected = [
        "safety_governance.py",
        "risk_mitigation_layers/",
        "compliance_check.py",
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        ".env",
        "config.json",
        "mt5_bridge.py",
        "okx_live/"
    ]

    for f in repo.rglob("*.py"):
        if any(part in f.parts for part in [".git", "tests", "testing", "__pycache__", ".venv"]):
            continue
        if any(f.match(prot) or prot.rstrip('/') in str(f) for prot in protected):
            continue
        try:
            sources.append(f"--- {f.relative_to(repo)} ---\n{f.read_text()}")
        except Exception:
            continue

    codebase = "\n\n".join(sources)
    prompt = (
        "You are a world-class quant developer. Analyze this trading system. "
        "Propose exactly one code improvement that increases win rate, reduces max drawdown, "
        "or raises Sharpe ratio without breaking any existing functionality. "
        "Output ONLY a valid git diff patch (unified diff). Do not modify any protected files: "
        "safety_governance.py, risk_mitigation_layers/, compliance_check.py, config, broker connectors, .env. "
        "Reply with the patch and nothing else.\n\n" + codebase
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        patch = response.choices[0].message.content.strip()

        if patch.startswith("diff --git") or patch.startswith("---"):
            return [{"description": "AI-suggested improvement", "patch": patch}]
        else:
            return []
    except Exception:
        return []
