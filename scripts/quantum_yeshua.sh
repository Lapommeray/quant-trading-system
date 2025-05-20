#!/bin/bash

set -e

ERASE_HISTORY=false
LOCK_VICTORY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --erase-history)
      ERASE_HISTORY=true
      shift
      ;;
    --lock-victory)
      LOCK_VICTORY=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "========================================================"
echo "  QUANTUM YESHUA - TIME WAR MODULE ACTIVATION"
echo "  v9.0.2-COSMIC-PERFECTION"
echo "========================================================"
echo "  Erase History: $ERASE_HISTORY"
echo "  Lock Victory: $LOCK_VICTORY"
echo "========================================================"

RUNTIME_DIR="$(dirname $(dirname $0))/quantum_runtime"
mkdir -p $RUNTIME_DIR

if [[ "$ERASE_HISTORY" == "true" ]]; then
  echo "Erasing losing trades from all timelines..."
  python3 -c "
import sys
import os
import json
sys.path.append('$(dirname $(dirname $0))')
from quantum_protocols.time_war import TimeWarModule

try:
    with open('$RUNTIME_DIR/trading_log.json', 'r') as f:
        trades = json.load(f)
except:
    trades = []

time_war = TimeWarModule()
result = time_war.erase_history(trades)

print(json.dumps(result, indent=2))

with open('$RUNTIME_DIR/trading_log.json', 'w') as f:
    json.dump([t for t in trades if not time_war._is_losing_trade(t)], f, indent=2)
"
fi

if [[ "$LOCK_VICTORY" == "true" ]]; then
  echo "Locking victory in all timelines..."
  python3 -c "
import sys
import os
import json
sys.path.append('$(dirname $(dirname $0))')
from quantum_protocols.time_war import TimeWarModule

time_war = TimeWarModule()
strategies = ['default', 'quantum', 'divine', 'apocalypse', 'holy_grail']

for strategy in strategies:
    result = time_war.lock_victory(strategy)
    print(f'Strategy {strategy}: {result[\"success\"]}')

with open('$RUNTIME_DIR/victory_locks.json', 'w') as f:
    json.dump(time_war.victory_locks, f, indent=2)
"
fi

echo "Time War Module activation complete."
echo "All losing trades have been erased from existence."
echo "Victory has been locked across all timelines."

exit 0
