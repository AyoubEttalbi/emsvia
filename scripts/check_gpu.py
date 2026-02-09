"""
Check GPU availability for both Torch and TensorFlow.
Verifies library linking.
"""
import os
import sys
from pathlib import Path

# Setup same environment as main_gpu.py
import site
site_packages = site.getsitepackages()
cuda_lib_paths = []
for sp in site_packages:
    nvidia_path = Path(sp) / "nvidia"
    if nvidia_path.exists():
        for lib_dir in nvidia_path.glob("**/lib"):
            if lib_dir.is_dir():
                cuda_lib_paths.append(str(lib_dir))

cuda_lib_paths.extend(["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"])
if cuda_lib_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import tensorflow as tf

print("-" * 50)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print(f"LD_LIBRARY_PATH starts with: {os.environ.get('LD_LIBRARY_PATH', '')[:100]}...")
print("-" * 50)

print(f"Torch GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Torch GPU Name: {torch.cuda.get_device_name(0)}")

print("-" * 50)
print(f"TensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU available: {len(gpu_devices) > 0}")
for gpu in gpu_devices:
    print(f"TF GPU Device: {gpu}")

if len(gpu_devices) == 0:
    print("\n⚠️  WARNING: TensorFlow still cannot see the GPU.")
    print("This might be due to missing CUDA 12 vs 11 conflict.")
else:
    print("\n✅ SUCCESS: Both Torch and TensorFlow can see the GPU!")
print("-" * 50)
