import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class FaceEvaluator:
    """
    Evaluates the performance of different detection phases (MTCNN vs RetinaFace vs Ensemble).
    Calculates FPS, Detection Count, and Average Confidence.
    """
    
    def __init__(self):
        self.stats = {
            "mtcnn": {"count": 0, "total_time": 0, "frames": 0},
            "retinaface": {"count": 0, "total_time": 0, "frames": 0},
            "ensemble": {"count": 0, "total_time": 0, "frames": 0}
        }

    def update_stats(self, model_name: str, duration: float, num_faces: int):
        if model_name not in self.stats:
            self.stats[model_name] = {"count": 0, "total_time": 0, "frames": 0}
            
        self.stats[model_name]["count"] += num_faces
        self.stats[model_name]["total_time"] += duration
        self.stats[model_name]["frames"] += 1

    def get_fps(self, model_name: str) -> float:
        if self.stats[model_name]["total_time"] == 0:
            return 0.0
        return self.stats[model_name]["frames"] / self.stats[model_name]["total_time"]

    def print_report(self):
        print("\n--- Performance Report ---")
        for model, data in self.stats.items():
            fps = self.get_fps(model)
            avg_faces = data["count"] / max(1, data["frames"])
            print(f"Model: {model} | FPS: {fps:.1f} | Avg Faces/Frame: {avg_faces:.1f}")
