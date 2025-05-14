# Encrypted Bytecode

This directory contains obfuscated and encrypted execution engine code for the Quantum Trading System. The files in this directory are generated using PyArmor for code obfuscation and protection.

## Purpose

The encrypted bytecode provides several security benefits:
- Protection of proprietary trading algorithms
- Prevention of unauthorized access to execution logic
- Obfuscation of sensitive code paths
- Regulatory compliance through code isolation

## Generation Process

The encrypted bytecode is generated from the core execution engine using the following process:

1. The original source code is first processed through PyArmor
2. The resulting bytecode is further encrypted using quantum-resistant encryption
3. Runtime wrappers are generated to interface with the encrypted code
4. Verification hashes are created to ensure code integrity

## Usage

To use the encrypted modules:

```python
from secure.encrypted_bytecode import core_execution

# Initialize the execution engine
engine = core_execution.initialize(
    api_key="YOUR_API_KEY",
    security_level="maximum"
)

# Execute trades through the secure interface
result = engine.execute_trade(
    symbol="BTCUSD",
    direction="buy",
    quantity=1.0,
    price=50000.0
)
```

## Security Notes

- Never attempt to decompile or reverse-engineer the encrypted bytecode
- API keys and credentials are stored separately in secure environment variables
- All access to the encrypted modules is logged and audited
- The encryption keys are rotated regularly for maximum security

## Regeneration

To regenerate the encrypted bytecode (authorized users only):

```bash
cd /scripts
./obfuscate_core.sh --security-level=maximum
```

This will process the core execution engine and place the encrypted files in this directory.
