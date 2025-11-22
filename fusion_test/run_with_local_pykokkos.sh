#!/bin/bash
# Helper script to run examples with the local pykokkos version
# This ensures the horizontal fusion implementation is used

# Add the workspace to PYTHONPATH
export PYTHONPATH=/home/sifi/pykokkos-fuse:$PYTHONPATH

# Run the command passed as arguments
exec "$@"


