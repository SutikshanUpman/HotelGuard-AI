#!/bin/sh
# entrypoint.sh — routes startup based on environment
#
# The OpenEnv validator sets RUN_INFERENCE=1 when it wants to evaluate.
# HuggingFace Spaces does not set this, so uvicorn starts normally.
#
# If the validator instead calls `python inference.py` directly (no env var),
# that still works because inference.py has its own __main__ guard.

set -e

if [ "$RUN_INFERENCE" = "1" ]; then
    echo "[ENTRYPOINT] RUN_INFERENCE=1 detected — running inference.py"
    exec python inference.py
else
    echo "[ENTRYPOINT] Starting uvicorn (HuggingFace Spaces / normal mode)"
    exec uvicorn app:app --host 0.0.0.0 --port 7860
fi
