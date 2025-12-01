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
from .r_function import R_idx, D_point_to_class_idx, D_point_to_set_idx
from collections import defaultdict

def _safe_cov_inv(*args, **kwargs):
    """
    Placeholder : Mahalanobis retiré (choix A). Cette fonction reste comme stub
    si besoin futur, mais n'est pas utilisée ici.
    """
    return None

class NuesDynamiques:
    def __init__(self, k=3, n_etalon=1, max_iter=200, distance="euclidienne", seed=None):
        self.k = int(k)
        self.n_etalon = int(n_etalon)
        self.max_iter = int(max_iter)
        self.distance = distance.lower()
        self.seed = seed

        # attributs publics
        self.n_iter_ = None
        self.L_indices_ = None
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
            raise ValueError(f"Distance inconnue : {self.distance}. Choix possibles : {list(self._distance_map.keys())}")

    # -----------------------
    # Initialisation des noyaux L
    # -----------------------
    def _init_L_random(self, n_samples):
        """Initialise L_indices_ : pour chaque classe, choisir n_etalon indices aléatoires."""
        rng = np.random.RandomState(self.seed)
        L = []
        
        # On s'assure que le nombre total d'étalons initiaux ne dépasse pas n_samples
        total_etalons = min(self.k * self.n_etalon, n_samples)
        
        # Tirer tous les indices uniques
        all_random_indices = rng.choice(n_samples, size=total_etalons, replace=False)
        
        # Répartir les indices entre les classes (E_i)
        for i in range(self.k):
            start = i * self.n_etalon
            end = (i + 1) * self.n_etalon
            
            # Prendre les indices, ajuster si moins de points sont disponibles
            idx = all_random_indices[start:min(end, total_etalons)].tolist()
            L.append(idx)
            
            # Sortir si on a épuisé tous les points
            if len(L[i]) == 0 and n_samples > 0:
                 # Si n_etalon est trop grand, cela pourrait laisser des noyaux vides. 
                 # C'est une limite pratique. On continue pour les autres classes.
                 pass

        self.L_indices_ = L

    # -----------------------
    # Fit (algorithme)
    # -----------------------
    def fit(self, X):
        """
        Entrée : X (n_samples, n_features)
        Sortie (après fit) :
          - self.L_indices_ : liste des noyaux E1..EK (listes d'indices)
          - self.classes_ : dict i -> list(indices des points de la classe)
          - self.n_iter_ : nombre d'itérations effectuées (mesure de securité deja fixé à 200 et peut être ajusté, pour ne pas faire crashé l'application au cas où ça ne converge pas.)
          - self.converged_ : bool
          - self.total_partition_quality_ : mesure de la valeur de la partition obtenue.
          - self.class_homogeneity_ : mesure de l’homogénéité de chacune des classes obtenues.
          - self.similarity_each_individuals_ : degré de similarité de chaque individus à chaque classe(non implémenté ici).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X doit être une matrice 2D")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("X est vide")

        self._init_L_random(n_samples)
        self.classes_ = {i: [] for i in range(self.k)}
        distance_fn = self._distance_map[self.distance]
        self.n_iter_ = 0
        self.converged_ = False

        for it in range(self.max_iter):
            old_L_indices = [list(k) for k in self.L_indices_] # Copie pour la convergence
            new_classes = {i: [] for i in range(self.k)}

            # --- Étape 2 : Affectation (Partition) ---
            # Affectation : minimiser R(x,i,L). On utilise la formule R_idx.
            # R_idx dépend de self.classes_ (la partition précédente) pour le terme D(x, C_i).
            for idx in range(n_samples):
                x = X[idx]
                best_i = None
                best_score = float('inf')
                for i in range(self.k):
                    # Préparation des arguments spécifiques à la distance (ex: Sebestyen)
                    dist_kwargs = {}
                    if self.distance == "sebestyen":
                        members = self.classes_.get(i, [])
                        if members:
                            variances = np.var(X[np.asarray(members, dtype=int)], axis=0, ddof=1)
                            dist_kwargs = {"variances": variances}
                            
                    score = R_idx(x, i, self.L_indices_, self.classes_, X, distance_fn, distance_kwargs=dist_kwargs)
                    
                    if score < best_score:
                        best_score = score
                        best_i = i
                
                # S'assurer qu'un point est toujours assigné si possible
                if best_i is not None:
                    new_classes[best_i].append(idx)
                else:
                    # Cas théorique si toutes les distances sont 'inf' ou NaN
                    # On assigne à la classe 0 par défaut pour éviter de perdre le point.
                    new_classes[0].append(idx)


            # --- Étape 3 : Mise à jour des noyaux L ---
            # Pour chaque classe, choisir les n_etalon points les plus CENTRAUX
            new_L = []
            for i in range(self.k):
                members = new_classes[i]
                
                # Si pas de membres dans la classe, garder l'ancien noyau
                if len(members) < self.n_etalon:
                    new_L.append(list(old_L_indices[i])) 
                    continue

                # Calculer la centralité (score R) pour chaque membre de la nouvelle classe C_i
                # Le score R est ici D(x, C_i) (distance moyenne à la classe) pour trouver le centre.
                score_list = []
                for idx in members:
                    x = X[idx]
                    
                    # Gestion des arguments spécifiques à la distance (ex: Sebestyen)
                    dist_kwargs = {}
                    if self.distance == "sebestyen":
                        # Pour la centralité, on se base sur la variance de la classe nouvellement formée (members)
                        variances = np.var(X[np.asarray(members, dtype=int)], axis=0, ddof=1)
                        dist_kwargs = {"variances": variances}

                    # Utilisation de D_point_to_class_idx (distance moyenne à la classe) 
                    # pour mesurer la centralité du point 'x' au sein de sa nouvelle classe 'members'.
                    s = D_point_to_class_idx(x, X, members, distance_fn, distance_kwargs=dist_kwargs)
                    score_list.append((idx, s))

                # Trier par score croissant (le plus central = score le plus bas) et choisir les meilleurs n_etalon
                score_list.sort(key=lambda t: t[1])
                chosen = [t[0] for t in score_list[:self.n_etalon]]
                new_L.append(chosen)

            # --- Test de convergence ---
            # Comparer les ensembles d'indices par classe : L^{(n+1)} == L^{(n)}
            same = True
            for old, new in zip(old_L_indices, new_L):
                # Utiliser set() pour ignorer l'ordre
                if set(old) != set(new):
                    same = False
                    break

            # Mise à jour des structures
            self.L_indices_ = new_L
            self.classes_ = new_classes
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
        """
        Calcule les métriques de qualité de la partition L (noyaux) obtenue.
        - total_quality (U(L)) : Somme des D(x, E_i) pour tous x. (Objectif de Diday)
        - class_homogeneity : dict i -> H_i (homogénéité moyenne de la classe)
        """
        total_quality = 0.0
        class_homogeneity = {}
        distance_fn = self._distance_map[self.distance]
        
        for i in range(self.k):
            members_indices = self.classes_.get(i, [])
            Ei_indices = self.L_indices_[i]
            
            if not members_indices or not Ei_indices:
                class_homogeneity[i] = 0.0
                continue

            sum_class_D = 0.0
            
            # Gestion des variances pour Sebestyen (basées sur les classes finales)
            dist_kwargs = {}
            if self.distance == "sebestyen":
                variances = np.var(X[np.asarray(members_indices, dtype=int)], axis=0, ddof=1)
                dist_kwargs = {"variances": variances}
            
            # Recalculer D(x, E_i) pour chaque point dans sa classe assignée
            for idx in members_indices:
                x = X[idx]
                
                # D(x, E_i) est la somme des distances (Diday's definition)
                D_xEi = D_point_to_set_idx(x, X, Ei_indices, distance_fn, dist_kwargs)
                sum_class_D += D_xEi
            
            class_homogeneity[i] = sum_class_D / len(members_indices)
            total_quality += sum_class_D
            
        return total_quality, class_homogeneity

    def compute_similarity_matrix(self, X):
        """
        Calcule la matrice de similarité (optionnelle) : 
        pour chaque point x, distance D(x, E_j) à tous les noyaux E_j.
        (Distance D(x, E_j) est inversement proportionnelle à la similarité)
        """
        n_samples = X.shape[0]
        similarity_matrix = np.zeros((n_samples, self.k))
        distance_fn = self._distance_map[self.distance]

        # Variances for Sebestyen (using final classes_ for consistency)
        variances_by_class = {}
        if self.distance == "sebestyen":
             for i in range(self.k):
                 members = self.classes_.get(i, [])
                 if members:
                    variances_by_class[i] = np.var(X[np.asarray(members, dtype=int)], axis=0, ddof=1)
                 else:
                    variances_by_class[i] = None

        for idx in range(n_samples):
            x = X[idx]
            for j in range(self.k):
                Ej_indices = self.L_indices_[j]
                
                dist_kwargs = {}
                if self.distance == "sebestyen" and variances_by_class[j] is not None:
                     dist_kwargs = {"variances": variances_by_class[j]}
                
                # D(x, E_j) est la somme des distances
                similarity_matrix[idx, j] = D_point_to_set_idx(x, X, Ej_indices, distance_fn, dist_kwargs)

        return similarity_matrix

    # -----------------------
    # Predict
    # -----------------------
    def predict(self, X):
        """
        Retourne pour chaque point l'indice de la classe (0..k-1) qui minimise R.
        """
        X = np.asarray(X, dtype=float)
        if self.L_indices_ is None:
            raise ValueError("Appeler fit(X) avant predict()")
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=int)
        distance_fn = self._distance_map[self.distance]

        for idx in range(n_samples):
            x = X[idx]
            best_i = 0 # Initialiser à 0
            best_score = float('inf')
            for i in range(self.k):
                dist_kwargs = {}
                if self.distance == "sebestyen":
                    # Pour la prédiction, utiliser les variances des classes finales obtenues par fit
                    members_prev = self.classes_.get(i, [])
                    if members_prev:
                        variances = np.var(X[np.asarray(members_prev, dtype=int)], axis=0, ddof=1)
                        dist_kwargs = {"variances": variances}

                # On utilise R_idx, la fonction d'agrégation choisie
                score = R_idx(x, i, self.L_indices_, self.classes_, X, distance_fn, distance_kwargs=dist_kwargs)
                if score < best_score:
                    best_score = score
                    best_i = i
            labels[idx] = best_i

        return labels