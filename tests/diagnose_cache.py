import pickle
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

def diagnose():
    cache_path = Path("data/embeddings/cache.pkl")
    if not cache_path.exists():
        print("Cache file not found.")
        return

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    print(f"Loaded cache with {len(cache)} students.")
    
    student_stats = {}
    for student_id, models in cache.items():
        print(f"\nStudent ID: {student_id}")
        for model_name, vectors in models.items():
            vecs = np.array(vectors)
            norms = np.linalg.norm(vecs, axis=1)
            
            # Internal distance
            internal_dists = []
            if len(vectors) > 1:
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        internal_dists.append(cosine(vectors[i], vectors[j]))
            
            avg_internal = np.mean(internal_dists) if internal_dists else 0
            max_internal = np.max(internal_dists) if internal_dists else 0
            
            print(f"  Model: {model_name}")
            print(f"    Count: {len(vectors)}")
            print(f"    Norms: Min={np.min(norms):.4f}, Max={np.max(norms):.4f}, Mean={np.mean(norms):.4f}")
            print(f"    Internal Dist: Mean={avg_internal:.4f}, Max={max_internal:.4f}")
            
            student_stats[(student_id, model_name)] = {
                "vectors": vectors,
                "avg_internal": avg_internal
            }

    print("\n" + "="*50)
    print("Cross-Student Minimum Distances (ArcFace)")
    print("="*50)
    
    combos = list(student_stats.keys())
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            s1, m1 = combos[i]
            s2, m2 = combos[j]
            if m1 != "ArcFace" or m2 != "ArcFace": continue
            
            v1_list = student_stats[(s1, m1)]["vectors"]
            v2_list = student_stats[(s2, m2)]["vectors"]
            
            min_dist = 1.0
            for v1 in v1_list:
                for v2 in v2_list:
                    d = cosine(v1, v2)
                    if d < min_dist: min_dist = d
            
            print(f"Student {s1} vs {s2}: Min={min_dist:.4f}")

if __name__ == "__main__":
    diagnose()
