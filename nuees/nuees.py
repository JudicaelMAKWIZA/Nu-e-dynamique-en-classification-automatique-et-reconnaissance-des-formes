"""
Nuées Dynamiques — implémentation.

- multi-étalons par classe (n_etalon)
- L_indices_ : liste de listes d'indices (utilisé pour le type 'etalons')
- L_kernels_ : liste des représentations du noyau (indices ou vecteur centroïde)
- classes_ : dict i -> list(indices des points de la classe)
- Convergence : L^{(n+1)} == L^{(n)} (égalité des ensembles d'indices pour 'etalons', ou stabilité du vecteur pour 'centroide')
- Distances supportées : 'euclidienne', 'sebestyen', 'chebychev', 'chi2'
- Type de noyaux supportés : 'etalons' (ensembles d'individus), 'centroide' (centres de gravité pondérés)
- Support des masses (mu_i) : chaque individu peut avoir un poids.
"""

import numpy as np
from .distances import euclidienne, sebestyen, chebychev, chi2
# MODIFICATION: Ajout de D_point_to_vector pour le noyau centroïde
from .r_function import R_idx, D_point_to_class_idx, D_point_to_set_idx, D_point_to_vector
from collections import defaultdict

def _safe_cov_inv(*args, **kwargs):
    """
    si besoin futur, mais n'est pas utilisée ici(Pour la métrique de Mahalanobis).
    """
    return None

class NuesDynamiques:
    # MODIFICATION: Ajout de 'kernel_type'
    def __init__(self, k=3, n_etalon=1, max_iter=200, distance="euclidienne", seed=None, kernel_type="etalons"):
        self.k = int(k)
        self.n_etalon = int(n_etalon)
        self.max_iter = int(max_iter)
        self.distance = distance.lower()
        self.seed = seed
        self.kernel_type = kernel_type.lower() # Ajout du type de noyau

        # attributs publics
        self.n_iter_ = None
        self.L_indices_ = None # Contient les indices (seulement si kernel_type='etalons')
        self.L_kernels_ = None # Liste des noyaux (indices ou vecteurs)
        self.classes_ = None
        self.converged_ = False
        self.total_partition_quality_ = None
        self.class_homogeneity_ = None
        
        # attributs privés (stockent les données de fit)
        self._X = None
        self._mu = None
        self.mu_ = None 

        # mapping des distances
        self._distance_map = {
            "euclidienne": euclidienne,
            "sebestyen": sebestyen,
            "chebychev": chebychev,
            "chi2": chi2
        }
        if self.distance not in self._distance_map:
            raise ValueError(f"Distance inconnue : {self.distance}. Choix possibles : {list(self._distance_map.keys())}")
        
        # mapping des types de noyaux
        self._kernel_type_map = {
            "etalons": self._update_L_etalons,
            "centroide": self._update_L_centroide,
        }
        if self.kernel_type not in self._kernel_type_map:
            raise ValueError(f"Type de noyau inconnu : {self.kernel_type}. Choix possibles : {list(self._kernel_type_map.keys())}")


    # -----------------------
    # Initialisation des noyaux L
    # -----------------------
    def _init_L_random(self, n_samples):
        """Initialise L_indices_ (indices aléatoires) et L_kernels_ (noyaux)."""
        rng = np.random.RandomState(self.seed)
        L = []
        
        total_etalons = min(self.k * self.n_etalon, n_samples)
        
        # Tirer tous les indices uniques
        all_random_indices = rng.choice(n_samples, size=total_etalons, replace=False)
        
        # Répartir les indices entre les classes (E_i)
        for i in range(self.k):
            start = i * self.n_etalon
            end = (i + 1) * self.n_etalon
            
            idx = all_random_indices[start:min(end, total_etalons)].tolist()
            L.append(idx)
            
            if len(L[i]) == 0 and n_samples > 0:
                   pass

        self.L_indices_ = L
        
        # Initialisation de L_kernels_ en fonction du type de noyau
        if self.kernel_type == "centroide":
            # Le noyau initial est le centroïde des points initiaux
            self.L_kernels_ = []
            for indices in self.L_indices_:
                centroid = self._calculate_centroide(self._X, indices, self._mu)
                self.L_kernels_.append(centroid)
        else: # 'etalons'
            # L_kernels_ est la même chose que L_indices_ pour le type étalons
            self.L_kernels_ = self.L_indices_

    # -----------------------
    # Utilitaire pour Sebestyen
    # -----------------------
    def _get_distance_kwargs(self, X, members):
        """Calcule les arguments additionnels pour la fonction de distance (ex: variances pour Sebestyen)."""
        dist_kwargs = {}
        if self.distance == "sebestyen" and members:
            # Calcul de la variance non pondérée
            variances = np.var(X[np.asarray(members, dtype=int)], axis=0, ddof=1)
            dist_kwargs = {"variances": variances}
        return dist_kwargs

    # -----------------------
    # Fonctions de mise à jour du noyau (Étape nu)
    # -----------------------
    def _calculate_centroide(self, X, indices, mu):
        """Calcule le centre de gravité (moyenne PONDÉRÉE) des points d'une classe."""
        if not indices:
            return None
        
        indices_arr = np.asarray(indices, dtype=int)
        
        mu_class = mu[indices_arr]
        X_class = X[indices_arr]
        
        total_mass = np.sum(mu_class)
        if total_mass == 0:
             return np.mean(X_class, axis=0)
        
        # Calcul de la moyenne pondérée (centroïde)
        centroid = np.sum(X_class * mu_class[:, np.newaxis], axis=0) / total_mass
        return centroid

    def _update_L_centroide(self, X, new_classes, old_L_kernels):
        """Mise à jour pour le noyau 'centroide': Calculer le centre de gravité."""
        new_L_kernels = []
        for i in range(self.k):
            members = new_classes[i]
            
            centroid = self._calculate_centroide(X, members, self._mu)
            
            if centroid is None:
                # Si la classe est vide, garder l'ancien centroïde
                new_L_kernels.append(old_L_kernels[i])
            else:
                new_L_kernels.append(centroid)
                
        # Pour le centroïde, L_indices_ est une liste vide (pas d'étalons réels)
        return new_L_kernels, [[] for _ in range(self.k)]

    def _update_L_etalons(self, X, new_classes, old_L_indices):
        """Mise à jour pour le noyau 'etalons': Choisir les n_etalon points les plus centraux."""
        new_L_indices = []
        for i in range(self.k):
            members = new_classes[i]
            
            if len(members) < self.n_etalon:
                new_L_indices.append(list(old_L_indices[i])) 
                continue

            score_list = []
            
            dist_kwargs = self._get_distance_kwargs(X, members)

            for idx in members:
                x = X[idx]
                
                # Le score de centralité est D(x, C_i) (distance MOYENNE PONDÉRÉE à la classe)
                s = D_point_to_class_idx(
                    x, X, members, self._mu, 
                    self._distance_map[self.distance], distance_kwargs=dist_kwargs
                )
                score_list.append((idx, s))

            # Trier par score croissant (le plus central = score le plus bas)
            score_list.sort(key=lambda t: t[1])
            chosen = [t[0] for t in score_list[:self.n_etalon]]
            new_L_indices.append(chosen)

        # Pour les étalons, L_kernels_ est la même chose que L_indices_
        return new_L_indices, new_L_indices


    # -----------------------
    # Fit (algorithme)
    # -----------------------
    def fit(self, X, sample_weights=None):
        """
        Entrée : X (n_samples, n_features), sample_weights (n_samples, optionnel)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X doit être une matrice 2D")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("X est vide")

        # --- GESTION ET STOCKAGE DES MASSES (mu_i) ---
        if sample_weights is None:
            self.mu_ = np.ones(n_samples, dtype=float)
        else:
            self.mu_ = np.asarray(sample_weights, dtype=float)
            if self.mu_.shape != (n_samples,):
                raise ValueError("sample_weights doit avoir la même taille que le nombre d'échantillons.")
        
        self._X = X 
        self._mu = self.mu_
        # --------------------------------------------

        self._init_L_random(n_samples) # <-- DOIT ÊTRE APPELÉ APRÈS self._X et self._mu
        self.classes_ = {i: [] for i in range(self.k)}
        distance_fn = self._distance_map[self.distance]
        self.n_iter_ = 0
        self.converged_ = False

        for it in range(self.max_iter):
            # L_representation est l'objet utilisé pour le test de convergence
            old_L_representation = [list(k) for k in (self.L_indices_ if self.kernel_type == "etalons" else self.L_kernels_)]
            
            new_classes = {i: [] for i in range(self.k)}

            # --- Étape 2 : Affectation (Partition) ---
            for idx in range(n_samples):
                x = X[idx]
                best_i = None
                best_score = float('inf')
                for i in range(self.k):
                    
                    dist_kwargs = self._get_distance_kwargs(X, self.classes_.get(i, []))
                        
                    # MODIFICATION: Affectation en fonction du type de noyau
                    if self.kernel_type == "etalons":
                         # Utilise la fonction R_idx (critère d'agrégation/écartement)
                         score = R_idx(
                            x, i, self.L_indices_, self.classes_, X, self.mu_, 
                            distance_fn, distance_kwargs=dist_kwargs
                        )
                    else: # Centroide
                        # Utilise la distance d(x, G_i) pure (critère k-means)
                        Gi = self.L_kernels_[i]
                        if Gi is not None:
                             score = D_point_to_vector(
                                x, Gi, distance_fn, distance_kwargs=dist_kwargs 
                             )
                        else:
                             score = float('inf')
                        
                    if score < best_score:
                        best_score = score
                        best_i = i
                    
                if best_i is not None:
                    new_classes[best_i].append(idx)
                else:
                    new_classes[0].append(idx)


            # --- Étape 3 : Mise à jour des noyaux L ---
            # Appel à la fonction appropriée (définie dans _kernel_type_map)
            new_L_kernels, new_L_indices = self._kernel_type_map[self.kernel_type](
                X, new_classes, self.L_kernels_ # old_L_kernels pour le centroïde / old_L_indices pour les étalons (via L_kernels_)
            )


            # --- Test de convergence ---
            same = True
            new_L_representation = new_L_indices if self.kernel_type == "etalons" else new_L_kernels
            
            for old, new in zip(old_L_representation, new_L_representation):
                if self.kernel_type == "etalons":
                    # Pour les étalons (indices), on compare l'ensemble des indices
                    if set(old) != set(new):
                        same = False
                        break
                else:
                    # Pour les centroïdes (vecteurs), on compare la stabilité du vecteur
                    # Gérer les cas où le noyau est None
                    if (old is None and new is not None) or (new is None and old is not None):
                        same = False
                        break
                    if old is not None and new is not None:
                         # Utiliser la norme euclidienne pour la comparaison
                         if np.linalg.norm(np.asarray(old) - np.asarray(new)) > 1e-6:
                             same = False
                             break
                        
            # Mise à jour des structures
            self.L_indices_ = new_L_indices # Indices mis à jour (vides si centroide)
            self.L_kernels_ = new_L_kernels # Noyaux (indices ou vecteurs) mis à jour
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
        - total_quality (U(L)) : Somme PONDÉRÉE des D(x, A_i) pour tous x.
        - class_homogeneity : H_i (homogénéité MOYENNE PONDÉRÉE de la classe)
        """
        total_quality = 0.0
        class_homogeneity = {}
        distance_fn = self._distance_map[self.distance]
        
        for i in range(self.k):
            members_indices = self.classes_.get(i, [])
            kernel_i = self.L_kernels_[i] # Noyau (indices ou vecteur)

            # Vérification si la classe/noyau est valide
            if not members_indices or kernel_i is None or (self.kernel_type == "etalons" and not kernel_i):
                class_homogeneity[i] = 0.0
                continue

            sum_class_D_weighted = 0.0
            dist_kwargs = self._get_distance_kwargs(X, members_indices)
            
            for idx in members_indices:
                x = X[idx]
                mu_x = self.mu_[idx]

                if self.kernel_type == "etalons":
                    # D(x, E_i) est la somme des distances (Diday's definition)
                    D_xEi = D_point_to_set_idx(x, X, kernel_i, distance_fn, dist_kwargs)
                else: # Centroide
                    # D(x, G_i) est la distance au centroïde
                    D_xEi = D_point_to_vector(x, kernel_i, distance_fn, dist_kwargs)

                # Le critère total est la somme PONDÉRÉE par la masse du point
                sum_class_D_weighted += D_xEi * mu_x
                
            total_mass_class = np.sum(self.mu_[members_indices])
            class_homogeneity[i] = sum_class_D_weighted / total_mass_class if total_mass_class > 0 else 0.0
            total_quality += sum_class_D_weighted
            
        return total_quality, class_homogeneity

    def compute_similarity_matrix(self, X):
        """
        Calcule la matrice de distance D(x, A_j)
        """
        n_samples = X.shape[0]
        similarity_matrix = np.zeros((n_samples, self.k))
        distance_fn = self._distance_map[self.distance]

        # Précalcul des variances (pour Sebestyen)
        variances_by_class = {}
        for i in range(self.k):
            members = self.classes_.get(i, [])
            dist_kwargs = self._get_distance_kwargs(self._X, members)
            if "variances" in dist_kwargs:
                variances_by_class[i] = dist_kwargs["variances"]
            else:
                variances_by_class[i] = None

        for idx in range(n_samples):
            x = X[idx]
            for j in range(self.k):
                kernel_j = self.L_kernels_[j] # Noyau (indices ou vecteur)
                
                dist_kwargs = {}
                if self.distance == "sebestyen" and variances_by_class[j] is not None:
                    dist_kwargs = {"variances": variances_by_class[j]}
                
                if self.kernel_type == "etalons":
                    # D(x, E_j) est la somme des distances
                    similarity_matrix[idx, j] = D_point_to_set_idx(x, self._X, kernel_j, distance_fn, dist_kwargs)
                else: # Centroide
                    # D(x, G_j) est la distance au vecteur
                    similarity_matrix[idx, j] = D_point_to_vector(x, kernel_j, distance_fn, dist_kwargs)

        return similarity_matrix

    # -----------------------
    # Predict
    # -----------------------
    def predict(self, X):
        """
        Retourne pour chaque point l'indice de la classe (0..k-1) qui minimise le critère d'affectation R ou D.
        """
        X = np.asarray(X, dtype=float)
        if self.L_indices_ is None or self._X is None or self._mu is None:
            raise ValueError("Appeler fit(X) avant predict()")
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=int)
        distance_fn = self._distance_map[self.distance]

        for idx in range(n_samples):
            x = X[idx]
            best_i = 0 
            best_score = float('inf')
            for i in range(self.k):
                
                members_prev = self.classes_.get(i, [])
                dist_kwargs = self._get_distance_kwargs(self._X, members_prev)

                if self.kernel_type == "etalons":
                    # Utiliser R_idx (critère d'agrégation/écartement)
                    score = R_idx(
                        x, i, self.L_indices_, self.classes_, self._X, self._mu, 
                        distance_fn, distance_kwargs=dist_kwargs
                    )
                else: # Centroide
                    # Utiliser D(x, G_i) pure (critère k-means)
                    Gi = self.L_kernels_[i]
                    if Gi is not None:
                         score = D_point_to_vector(
                            x, Gi, distance_fn, distance_kwargs=dist_kwargs
                         )
                    else:
                         score = float('inf')
                         
                if score < best_score:
                    best_score = score
                    best_i = i
            labels[idx] = best_i

        return labels