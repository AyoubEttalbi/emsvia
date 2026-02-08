import os
import requests
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# OpenCV's DNN SuperRes models
MODELS = {
    "FSRCNN_x4.pb": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
    "EDSR_x4.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
}

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def main():
    print(f"Checking/Downloading Super-Resolution models to {MODELS_DIR}...")
    
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f" - {filename} already exists.")
        else:
            try:
                download_file(url, dest)
                print(f" - {filename} downloaded successfully.")
            except Exception as e:
                print(f" ! Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
