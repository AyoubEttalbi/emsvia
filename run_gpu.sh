#!/bin/bash

# Get the directory where this script is located
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_LIB="$PROJECT_DIR/venv/lib/python3.12/site-packages"

echo "🚀 EMSVIA GPU Launcher"

# Find all nvidia library directories
NVIDIA_LIBS=$(find "$VENV_LIB/nvidia" -name "lib" -type d | tr '\n' ':')

# Export environment variables for the shell
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$VENV_LIB/nvidia"

echo "✅ Environment configured for GPU."
echo "📦 CUDA Paths: $LD_LIBRARY_PATH"

# Run the application
./venv/bin/python3 main.py "$@"
