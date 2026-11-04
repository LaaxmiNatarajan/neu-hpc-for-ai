
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "====================================="
echo "Continuous Batching Scheduler Tests"
echo "====================================="

python test_scheduler.py

echo
echo "Scheduler tests completed successfully."
