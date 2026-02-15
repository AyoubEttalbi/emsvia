import os
import site
import sys
from pathlib import Path

def check_cuda_libs():
    print(f"Python Version: {sys.version}")
    print(f"Prefix: {sys.prefix}")
    
    site_packages = site.getsitepackages()
    print(f"Site Packages: {site_packages}")
    
    cuda_lib_paths = []
    for sp in site_packages:
        nvidia_path = Path(sp) / "nvidia"
        print(f"Checking {nvidia_path}...")
        if nvidia_path.exists():
            print(f"  FOUND nvidia path at {nvidia_path}")
            for lib_dir in nvidia_path.glob("**/lib"):
                if lib_dir.is_dir():
                    print(f"    FOUND lib dir: {lib_dir}")
                    cuda_lib_paths.append(str(lib_dir))
    
    if not cuda_lib_paths:
        print("❌ NO CUDA LIB PATHS FOUND!")
    else:
        print(f"✅ Found {len(cuda_lib_paths)} unique lib paths")
        
if __name__ == "__main__":
    check_cuda_libs()
