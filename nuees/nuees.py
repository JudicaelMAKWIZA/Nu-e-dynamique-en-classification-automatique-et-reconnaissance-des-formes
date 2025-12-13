"""
Nuées Dynamiques — implémentation.

- multi-étalons par classe (n_etalon)
- L_indices_ : liste de listes d'indices (E1..EK)
- classes_ : dict i -> list(indices des points de la classe)
- Convergence : L^{(n+1)} == L^{(n)} (égalité des ensembles d'indices)
- Distances supportées : 'euclidienne', 'sebestyen', 'chebychev', 'chi2'
"""

import numpy as np
from .distances import euclidienne, sebestyen, chebychev, chi2
from .r_function import (
    R_idx,
    D_point_to_class_idx,
    D_point_to_set_idx,
    D_point_to_kernel  # 🔵 AJOUT
)
from collections import defaultdict


def _safe_cov_inv(*args, **kwargs):
    """
    si besoin futur, mais n'est pas utilisée ici(Pour la métrique de Mahalanobis).
    """
    return None


class NuesDynamiques:
    def __init__(
        self,
        k,
        n_etalon=1,
        max_iter=100,
        distance="euclidienne",
        seed=None,
        kernel_type="discrete" 
    ):
        self.k = k
        self.n_etalon = n_etalon
        self.max_iter = max_iter
        self.distance = distance
        self.seed = seed
        self.kernel_type = kernel_type

        if self.kernel_type != "discrete":
            raise NotImplementedError(
            "Seul le noyau discret (ensemble d'individus) est implémenté actuellement, "
            "conformément à l'article de Diday (1971). "
            "Les autres types de noyaux sont proposés dans l'interface mais "
            "non encore implémentés dans l'algorithme."
        )

        # attributs publics
        self.n_iter_ = None
        self.L_indices_ = None
        self.L_ = None               # 🔵 AJOUT : structure générale des noyaux
        self.classes_ = None
        self.converged_ = False
        self.total_partition_quality_ = None
        self.class_homogeneity_ = None

        # mapping des distances
        self._distance_map = {
            "euclidienne": euclidienne,
            "sebestyen": sebestyen,
            "chebychev": chebychev,
            "chi2": chi2
        }
        if self.distance not in self._distance_map:
            raise ValueError(
                f"Distance inconnue : {self.distance}. "
                f"Choix possibles : {list(self._distance_map.keys())}"
            )

    # -----------------------
    # Initialisation des noyaux L
    # -----------------------
    def _init_L_random(self, X):
        """Initialise L_indices_ : pour chaque classe, choisir n_etalon indices aléatoires."""
        rng = np.random.RandomState(self.seed)
        n_samples = X.shape[0]

        L = []

        total_etalons = min(self.k * self.n_etalon, n_samples)
        all_random_indices = rng.choice(
            n_samples, size=total_etalons, replace=False
        )

        for i in range(self.k):
            start = i * self.n_etalon
            end = (i + 1) * self.n_etalon
            idx = all_random_indices[start:min(end, total_etalons)].tolist()
            L.append(idx)

        # 🔵 AJOUT : construction générique des noyaux
        self.L_indices_ = L
        self.L_ = []

        for idxs in L:
            if self.kernel_type == "discrete":
                self.L_.append({
                    "type": "discrete",
                    "indices": idxs
                })

            elif self.kernel_type == "centroid":
                centroid = X[idxs].mean(axis=0)
                self.L_.append({
                    "type": "centroid",
                    "vector": centroid
                })

            elif self.kernel_type == "gaussian":
                mean = X[idxs].mean(axis=0)
                cov = np.cov(X[idxs].T)
                self.L_.append({
                    "type": "gaussian",
                    "mean": mean,
                    "cov": cov
                })

            elif self.kernel_type == "factorial":
                Xc = X[idxs] - X[idxs].mean(axis=0)
                _, _, vt = np.linalg.svd(Xc, full_matrices=False)
                axis = vt[0]
                origin = X[idxs].mean(axis=0)
                self.L_.append({
                    "type": "factorial",
                    "axis": axis,
                    "origin": origin
                })

    # -----------------------
    # Fit (algorithme)
    # -----------------------
    def fit(self, X):
        """
        Entrée : X (n_samples, n_features)
        Sortie (après fit) :
          - self.L_indices_ : liste des noyaux E1..EK (listes d'indices)
          - self.classes_ : dict i -> list(indices des points de la classe)
          - self.n_iter_ : nombre d'itérations effectuées
          - self.converged_ : bool
          - self.total_partition_quality_ : mesure de la valeur de la partition obtenue.
          - self.class_homogeneity_ : mesure de l’homogénéité de chacune des classes obtenues.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X doit être une matrice 2D")
        if X.shape[0] == 0:
            raise ValueError("X est vide")

        self._init_L_random(X)
        self.classes_ = {i: [] for i in range(self.k)}
        distance_fn = self._distance_map[self.distance]
        self.converged_ = False

        for it in range(self.max_iter):
            old_L = self.L_
            new_classes = {i: [] for i in range(self.k)}

            # --- Étape 2 : Affectation ---
            for idx, x in enumerate(X):
                scores = []
                for i in range(self.k):
                    s = R_idx(
                        x, i, self.L_, self.classes_,
                        X, distance_fn
                    )
                    scores.append(s)
                best_i = int(np.argmin(scores))
                new_classes[best_i].append(idx)

            self.classes_ = new_classes

            # --- Étape 3 : Mise à jour des noyaux ---
            self._init_L_random(X)

            # --- Test de convergence ---
            same = True
            for old, new in zip(old_L, self.L_):
                if old["type"] != new["type"]:
                    same = False
                    break
                if old["type"] == "discrete":
                    if set(old["indices"]) != set(new["indices"]):
                        same = False
                        break

            self.n_iter_ = it + 1
            if same:
                self.converged_ = True
                break

        self.total_partition_quality_, self.class_homogeneity_ = self._score_partition(X)
        return self

    # -----------------------
    # Score et Métriques
    # -----------------------
    def _score_partition(self, X):
        total_quality = 0.0
        class_homogeneity = {}
        distance_fn = self._distance_map[self.distance]

        for i in range(self.k):
            members = self.classes_.get(i, [])
            if not members:
                class_homogeneity[i] = 0.0
                continue

            s = 0.0
            for idx in members:
                x = X[idx]
                d = D_point_to_kernel(x, self.L_[i], X, distance_fn)
                s += d

            class_homogeneity[i] = s / len(members)
            total_quality += s

        return total_quality, class_homogeneity

    # -----------------------
    # Predict
    # -----------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.L_ is None:
            raise ValueError("Appeler fit(X) avant predict()")

        labels = np.empty(X.shape[0], dtype=int)
        distance_fn = self._distance_map[self.distance]

        for idx, x in enumerate(X):
            scores = []
            for i in range(self.k):
                s = R_idx(
                    x, i, self.L_, self.classes_,
                    X, distance_fn
                )
                scores.append(s)
            labels[idx] = int(np.argmin(scores))

        return labels
