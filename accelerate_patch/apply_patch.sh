#!/bin/bash
# This script is meant to be ran from the root of the repo: `bash accelerate_patch/apply_patch.sh`

# Locate the accelerate library directory
ACCELERATE_DIR=$(python -c "import accelerate; import os; print(os.path.dirname(accelerate.__file__))")

# Apply the patch
patch $ACCELERATE_DIR/accelerator.py < accelerate_patch/accelerator.patch
patch $ACCELERATE_DIR/data_loader.py < accelerate_patch/data_loader.patch

