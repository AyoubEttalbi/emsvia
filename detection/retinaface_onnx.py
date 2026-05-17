"""
RetinaFace Detection using ONNX Runtime GPU.
Much faster than TensorFlow backend and uses less VRAM.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Model constants (InsightFace buffalo_l / det_10g)
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
VARIANCES = [0.1, 0.2]
NMS_THRESHOLD = 0.4


def generate_anchors(input_size: int = INPUT_SIZE) -> np.ndarray:
    """Generate RetinaFace anchors for all FPN scales."""
    anchors = []
    for stride, min_sizes in zip(STRIDES, MIN_SIZES):
        h = input_size // stride
        w = input_size // stride
        for i in range(h):
            for j in range(w):
                for size in min_sizes:
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    anchors.append([cx, cy, size, size])
    return np.array(anchors, dtype=np.float32)


def decode_boxes(anchors: np.ndarray, bbox_deltas: np.ndarray) -> np.ndarray:
    """Decode bbox deltas to pixel coordinates."""
    aw = anchors[:, 2]
    ah = anchors[:, 3]
    acx = anchors[:, 0]
    acy = anchors[:, 1]

    cx = acx + bbox_deltas[:, 0] * VARIANCES[0] * aw
    cy = acy + bbox_deltas[:, 1] * VARIANCES[0] * ah
    w = aw * np.exp(bbox_deltas[:, 2] * VARIANCES[1])
    h = ah * np.exp(bbox_deltas[:, 3] * VARIANCES[1])

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.column_stack([x1, y1, x2, y2])


def decode_landmarks(anchors: np.ndarray, landmark_deltas: np.ndarray) -> np.ndarray:
    """Decode landmark deltas to pixel coordinates."""
    landmarks = np.zeros_like(landmark_deltas)
    ax = anchors[:, 0]
    ay = anchors[:, 1]
    aw = anchors[:, 2]
    ah = anchors[:, 3]
    for i in range(5):
        landmarks[:, i*2] = ax + landmark_deltas[:, i*2] * VARIANCES[0] * aw
        landmarks[:, i*2+1] = ay + landmark_deltas[:, i*2+1] * VARIANCES[0] * ah
    return landmarks


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = NMS_THRESHOLD) -> np.ndarray:
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


class RetinaFaceONNX:
    """
    High-performance RetinaFace detector using ONNX Runtime.
    """
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path or os.path.expanduser("~/.deepface/weights/retinaface.onnx")

        if not os.path.exists(self.model_path):
            logger.warning(f"RetinaFace ONNX model not found at {self.model_path}")
            self.session = None
            return

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']

        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.log_verbosity_level = 0
            # Suppress GPU timer warnings by redirecting stderr during init and warmup
            import sys as _sys
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull, 2)
            try:
                ort.set_default_logger_severity(3)
                self.session = ort.InferenceSession(self.model_path, providers=providers, sess_options=opts)
                # Warmup inference to consume GPU timing warnings
                warmup = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
                self.session.run(None, {self.session.get_inputs()[0].name: warmup})
            finally:
                os.dup2(saved_stderr, 2)
                os.close(devnull)
                os.close(saved_stderr)
            logger.info(f"RetinaFace ONNX initialized on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize RetinaFace ONNX: {e}")
            self.session = None

        self.anchors = generate_anchors(INPUT_SIZE)
        self.input_name = self.session.get_inputs()[0].name if self.session else None

    def detect(self, img: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.session is None:
            return []

        orig_h, orig_w = img.shape[:2]
        img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_input = img_input.astype(np.float32)
        img_input -= np.array([104, 117, 123], dtype=np.float32)
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)

        inputs = {self.input_name: img_input}
        outputs = self.session.run(None, inputs)

        # outputs: [score1, score2, score3, bbox1, bbox2, bbox3, land1, land2, land3]
        scores = np.vstack([outputs[0], outputs[1], outputs[2]])
        bboxes = np.vstack([outputs[3], outputs[4], outputs[5]])
        landmarks = np.vstack([outputs[6], outputs[7], outputs[8]])

        scores = scores.squeeze(axis=-1)
        bboxes = decode_boxes(self.anchors, bboxes)
        landmarks = decode_landmarks(self.anchors, landmarks)

        # Scale to original image size
        scale_x = orig_w / INPUT_SIZE
        scale_y = orig_h / INPUT_SIZE
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        landmarks[:, 0::2] *= scale_x
        landmarks[:, 1::2] *= scale_y

        # Filter by threshold
        mask = scores >= threshold
        bboxes = bboxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(bboxes) == 0:
            return []

        # NMS
        keep = nms(bboxes, scores, NMS_THRESHOLD)
        bboxes = bboxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        results = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            results.append({
                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(scores[i]),
                "model": "retinaface_onnx",
                "landmarks": [[int(lx), int(ly)] for lx, ly in landmarks[i].reshape(5, 2)],
            })
        return results

    def is_ready(self) -> bool:
        return self.session is not None
