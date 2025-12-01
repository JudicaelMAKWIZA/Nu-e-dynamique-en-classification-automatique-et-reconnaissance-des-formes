# tools/generate_datasets.py
"""
Générateur simple de datasets 2D pour tests.
Crée data/dataset_A.csv et data/dataset_B.csv avec des clusters aléatoires.
"""
# FICHIER AUXILIAIRE (TEST DE GENERATION DES DATASET)

import numpy as np
import pandas as pd
import os

def generate(n_points=300, k=3, dim=2, spread=1.0, out="data/dataset.csv", seed=None):
    rng = np.random.RandomState(seed)
    pts_per = n_points // k
    centers = rng.uniform(-8, 8, size=(k, dim))
    data = []
    for c in centers:
        cluster = c + rng.randn(pts_per, dim) * spread
        data.append(cluster)
    data = np.vstack(data)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(data).to_csv(out, index=False, header=False)
    return out

if __name__ == "__main__":
    generate(n_points=300, k=3, spread=1.2, out="data/dataset_A.csv", seed=42)
    generate(n_points=300, k=4, spread=0.8, out="data/dataset_B.csv", seed=2023)
    print("Datasets générés : data/dataset_A.csv, data/dataset_B.csv")
