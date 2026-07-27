#!/usr/bin/env bash
# Alias for dataset 2 (backward compatibility)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sh "$SCRIPT_DIR/train_safe_polamp_dataset_2.sh" "$@"
