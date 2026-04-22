import pickle
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

def run():
    base = Path("data/embeddings")
    files = list(base.glob("*.pkl"))

    embs = {}
    for f in files:
        with open(f, "rb") as fp:
            embs[f.stem] = pickle.load(fp)

    out = []
    out.append(f"Loaded students: {list(embs.keys())}")

    for s1 in embs:
        for s2 in embs:
            if s1 >= s2: continue
            model = "ArcFace"
            m1 = embs[s1].get(model, [])
            m2 = embs[s2].get(model, [])
            if not m1 or not m2: continue
            
            dists = [cosine(v1, v2) for v1 in m1 for v2 in m2]
            if dists:
                out.append(f"{s1} vs {s2}: Min={np.min(dists):.4f}, Mean={np.mean(dists):.4f}")

    for line in out:
        print(line)

if __name__ == "__main__":
    run()
