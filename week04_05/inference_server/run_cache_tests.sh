
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo "1. Testing custom preallocated KV cache"
echo "========================================"
python test_custom_kv_cache.py

echo
echo "========================================"
echo "2. Testing paged KV cache allocator"
echo "========================================"
python test_paged_kv_cache.py

echo
echo "========================================"
echo "3. Testing paged cache with Qwen3"
echo "========================================"
modal run run_paged_qwen3_modal.py

echo
echo "All cache tests completed."
