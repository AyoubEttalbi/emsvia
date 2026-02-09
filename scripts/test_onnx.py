import onnxruntime as ort
import numpy as np

providers = ort.get_available_providers()
print(f"Available providers: {providers}")

try:
    session = ort.InferenceSession(None, providers=['CUDAExecutionProvider'])
    print("✅ ONNX Runtime CUDA provider is working!")
except Exception as e:
    print(f"❌ ONNX Runtime CUDA provider failed: {e}")
