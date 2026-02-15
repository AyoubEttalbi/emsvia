import numpy as np
from typing import List, Dict, Tuple, Any
from collections import deque, OrderedDict
from config.settings import TRACKING_MAX_DISAPPEARED, TRACKER_IOU_THRESHOLD, RECOGNITION_MODELS

class FaceTracker:
    """
    A simple IOU-based tracker to maintain face identity across frames.
    """
    def __init__(self, max_disappeared=TRACKING_MAX_DISAPPEARED, iou_threshold=TRACKER_IOU_THRESHOLD):
        self.next_track_id = 0
        self.tracks = OrderedDict() # track_id -> {'box': [x,y,w,h], 'disappeared': 0, 'history': []}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def _calculate_iou(self, boxA, boxB):
        # box: [x, y, w, h] -> [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections. Returns detections with 'track_id'.
        """
        if len(detections) == 0:
            tracks_to_del = []
            results = []
            for track_id, track in self.tracks.items():
                track['disappeared'] += 1
                if track['disappeared'] > self.max_disappeared:
                    tracks_to_del.append(track_id)
                else:
                    # Return last known position with its ID
                    results.append({
                        'track_id': track_id,
                        'box': track['box']
                    })
            for tid in tracks_to_del:
                del self.tracks[tid]
            return results

        # If no tracks, register all detections
        if len(self.tracks) == 0:
            for det in detections:
                self._register_track(det)
        else:
            track_ids = list(self.tracks.keys())
            track_boxes = [t['box'] for t in self.tracks.values()]
            
            # Compute IOU matrix
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype="float32")
            for i, t_box in enumerate(track_boxes):
                for j, d in enumerate(detections):
                    iou_matrix[i, j] = self._calculate_iou(t_box, d['box'])
            
            # Match based on highest IOU
            matched_indices = []
            if iou_matrix.size > 0:
                for _ in range(min(len(track_ids), len(detections))):
                    idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    if iou_matrix[idx] < self.iou_threshold:
                        break
                    matched_indices.append(idx)
                    iou_matrix[idx[0], :] = -1 # Clear row
                    iou_matrix[:, idx[1]] = -1 # Clear col

            matched_tracks = set()
            matched_dets = set()
            for t_idx, d_idx in matched_indices:
                tid = track_ids[t_idx]
                self.tracks[tid]['box'] = detections[d_idx]['box']
                self.tracks[tid]['disappeared'] = 0
                detections[d_idx]['track_id'] = tid
                matched_tracks.add(tid)
                matched_dets.add(d_idx)

            # Handle disappeared tracks
            for t_idx, tid in enumerate(track_ids):
                if tid not in matched_tracks:
                    self.tracks[tid]['disappeared'] += 1
                    if self.tracks[tid]['disappeared'] > self.max_disappeared:
                        del self.tracks[tid]

            # Register new detections
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_dets:
                    self._register_track(det)

        return [d for d in detections if 'track_id' in d]

    def _register_track(self, detection):
        tid = self.next_track_id
        self.tracks[tid] = {
            'box': detection['box'],
            'disappeared': 0,
            'recognition_history': [], 
            'embeddings_buffer': {model: deque(maxlen=10) for model in RECOGNITION_MODELS},
            'identity_stability_counter': 0,
            'current_identity': None,
            'last_match': None,
            'frame_count': 0
        }
        detection['track_id'] = tid
        self.next_track_id += 1
