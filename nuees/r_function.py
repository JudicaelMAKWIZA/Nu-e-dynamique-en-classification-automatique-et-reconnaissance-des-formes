import numpy as np

def D_point_to_set_idx(x_vec, X, set_indices, distance_fn, distance_kwargs=None):
    """
    Distance d'un vecteur x_vec au noyau (liste d'indices) set_indices.
    SELON DIDAY (Définition 5) : On prend la SOMME des distances entre x
    et chaque élément du noyau E_i.
    (Utilisé pour le type de noyau 'etalons')
    """
    if not set_indices:
        return 1e9  # valeur neutre élevée si noyau vide

    total_distance = 0.0
    for idx in set_indices:
        y = X[int(idx)]
        if distance_kwargs:
            d = distance_fn(x_vec, y, **distance_kwargs)
        else:
            d = distance_fn(x_vec, y)
        total_distance += d
        
    return float(total_distance) # Retourne la SOMME


def D_point_to_vector(x_vec, vector, distance_fn, distance_kwargs=None):
    """
    Distance d'un vecteur x_vec à un autre vecteur (ex: centroïde).
    (Utilisé pour le type de noyau 'centroide' et pour le calcul de score)
    """
    if vector is None:
        return 1e9
        
    y = np.asarray(vector, dtype=float)
    x = np.asarray(x_vec, dtype=float)

    if distance_kwargs:
        d = distance_fn(x, y, **distance_kwargs)
    else:
        d = distance_fn(x, y)
        
    return float(d)


def D_point_to_class_idx(x_vec, X, class_indices, mu_, distance_fn, distance_kwargs=None):
    """
    Distance moyenne PONDÉRÉE de x_vec à tous les éléments de la classe (class_indices).
    (Utilisé pour la centralité dans l'étape de mise à jour des étalons)
    """
    if not class_indices:
        return 1e9

    total_dist_weighted = 0.0
    total_mass = 0.0
    
    for idx in class_indices:
        mu_y = mu_[int(idx)] # Récupération de la masse
        y = X[int(idx)]
        
        if distance_kwargs:
            d = distance_fn(x_vec, y, **distance_kwargs)
        else:
            d = distance_fn(x_vec, y)
            
        total_dist_weighted += mu_y * d # Distance PONDÉRÉE par la masse
        total_mass += mu_y

    return float(total_dist_weighted / total_mass) if total_mass > 0 else 1e9


def R_idx(x_vec, i, L_indices, classes_dict, X, mu_, distance_fn, distance_kwargs=None):
    """
    Calcul de R(x,i,L) selon Diday (Exemple 1):
        R(x,i,L) = D(x, E_i) * D(x, C_i) / ( sum_j D(x, E_j) )^2
    (Utilisé seulement pour le type de noyau 'etalons')
    """
    if distance_kwargs is None:
        distance_kwargs = {}

    Ei_idx = L_indices[i]
    
    # D(x, E_i) : distance à l'ensemble des étalons (SOMME)
    de = D_point_to_set_idx(x_vec, X, Ei_idx, distance_fn, distance_kwargs)
    
    # D(x, C_i) : distance moyenne PONDÉRÉE à la classe
    Ci_idx = classes_dict.get(i, [])
    dc = D_point_to_class_idx(x_vec, X, Ci_idx, mu_, distance_fn, distance_kwargs)
    
    if dc >= 1e9:
        dc = 1.0

    numerator = de * dc

    sum_dist = 0.0
    for Ej in L_indices:
        # La somme des distances D(x, E_j) pour le dénominateur
        sum_dist += D_point_to_set_idx(x_vec, X, Ej, distance_fn, distance_kwargs)

    denom = (sum_dist ** 2) if sum_dist != 0 else 1e-12

    return float(numerator / denom)