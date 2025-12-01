import numpy as np

def D_point_to_set_idx(x_vec, X, set_indices, distance_fn, distance_kwargs=None):
    """
    Distance d'un vecteur x_vec au noyau (liste d'indices) set_indices.
    SELON DIDAY (Définition 5) : On prend la SOMME des distances entre x
    et chaque élément du noyau E_i.
    """
    #if distance_kwargs is None:
    #    distance_kwargs = {}

    if not set_indices:
        return 1e9  # valeur neutre élevée si noyau vide

    # --- CORRECTION : Utiliser la SOMME (total) au lieu du MINIMUM (dmin) ---
    total_distance = 0.0
    for idx in set_indices:
        y = X[int(idx)]
        if distance_kwargs:
            d = distance_fn(x_vec, y, **distance_kwargs)
        else:
            d = distance_fn(x_vec, y)
        total_distance += d
        
    return float(total_distance) # Retourne la SOMME


def D_point_to_class_idx(x_vec, X, class_indices, distance_fn, distance_kwargs=None):
    """
    Distance moyenne de x_vec à tous les éléments de la classe (class_indices).
    (Utile pour l'étape de mise à jour R(x, i, L))
    """
    # [Le reste de cette fonction reste correct, car tu calcules déjà la somme / taille]
    # ...
    #if distance_kwargs is None:
     #   distance_kwargs = {}

    if not class_indices:
        return 1e9

    total = 0.0
    for idx in class_indices:
        y = X[int(idx)]
        if distance_kwargs:
            total += distance_fn(x_vec, y, **distance_kwargs)
        else:
            total += distance_fn(x_vec, y)
    return float(total / len(class_indices))


def R_idx(x_vec, i, L_indices, classes_dict, X, distance_fn, distance_kwargs=None):
    """
    Calcul de R(x,i,L) selon Diday (Exemple 1):
        R(x,i,L) = D(x, E_i) * D(x, C_i) / ( sum_j D(x, E_j) )^2
    """
    if distance_kwargs is None:
        distance_kwargs = {}

    Ei_idx = L_indices[i]
    # Note : Ci_idx n'est pas utilisé directement dans l'Exemple 1 de R(x,i,L)
    # pour le calcul de l'affectation, c'est l'étape 3 (maj des noyaux) qui utilise
    # R(x, i, L) ~ D(x, C_i), mais tu utilises la forme générale D(x, Ei) dans l'affectation.
    # Dans le contexte de l'Exemple 1 que tu as choisi pour R_idx, la dépendance
    # à self.classes_ dans cette fonction est correcte.

    # D(x, E_i) : distance à l'ensemble des étalons (SOMME)
    de = D_point_to_set_idx(x_vec, X, Ei_idx, distance_fn, distance_kwargs)
    
    # La fonction D_point_to_class_idx est utilisée ici pour le terme D(x, C_i) 
    # dans le numérateur R = D(x, E_i) * D(x, C_i) / [...]
    Ci_idx = classes_dict.get(i, [])
    # D(x, C_i) : distance moyenne à la classe (MOYENNE)
    dc = D_point_to_class_idx(x_vec, X, Ci_idx, distance_fn, distance_kwargs)
    
    # ... (le reste du code pour R_idx est laissé tel quel)
    if dc >= 1e9:
        # si la classe est vide, on neutralise l'impact de D(x, C_i)
        dc = 1.0

    numerator = de * dc

    sum_dist = 0.0
    for Ej in L_indices:
        # La somme des distances D(x, E_j) pour le dénominateur
        sum_dist += D_point_to_set_idx(x_vec, X, Ej, distance_fn, distance_kwargs)

    denom = (sum_dist ** 2) if sum_dist != 0 else 1e-12

    return float(numerator / denom)