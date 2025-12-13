import numpy as np


def D_point_to_set_idx(x_vec, X, set_indices, distance_fn, distance_kwargs=None):
    """
    Distance d'un vecteur x_vec au noyau (liste d'indices) set_indices.
    SELON DIDAY (Définition 5) : SOMME des distances entre x et chaque élément du noyau.
    """
    if not set_indices:
        return 1e9

    total = 0.0
    for idx in set_indices:
        y = X[int(idx)]
        if distance_kwargs:
            total += distance_fn(x_vec, y, **distance_kwargs)
        else:
            total += distance_fn(x_vec, y)
    return float(total)


def D_point_to_class_idx(x_vec, X, class_indices, distance_fn, distance_kwargs=None):
    """
    Distance moyenne de x_vec à tous les éléments de la classe.
    """
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


# 🔵 AJOUT : distance générique à un noyau
def D_point_to_kernel(x, kernel, X, distance_fn):
    ktype = kernel["type"]

    if ktype == "discrete":
        return D_point_to_set_idx(x, X, kernel["indices"], distance_fn)

    elif ktype == "centroid":
        return distance_fn(x, kernel["vector"])

    elif ktype == "gaussian":
        diff = x - kernel["mean"]
        inv_cov = np.linalg.pinv(kernel["cov"])
        return float(diff.T @ inv_cov @ diff)

    elif ktype == "factorial":
        v = kernel["axis"]
        g = kernel["origin"]
        return float(np.linalg.norm(np.cross(x - g, v)))

    else:
        raise ValueError("Type de noyau inconnu")


def R_idx(x_vec, i, L, classes_dict, X, distance_fn, distance_kwargs=None):
    """
    Calcul de R(x,i,L) selon Diday (Exemple 1):
        R(x,i,L) = D(x, E_i) * D(x, C_i) / ( sum_j D(x, E_j) )^2
    """
    if distance_kwargs is None:
        distance_kwargs = {}

    de = D_point_to_kernel(x_vec, L[i], X, distance_fn)
    dc = D_point_to_class_idx(
        x_vec, X, classes_dict.get(i, []),
        distance_fn, distance_kwargs
    )

    if dc >= 1e9:
        dc = 1.0

    numerator = de * dc

    denom_sum = 0.0
    for Ej in L:
        denom_sum += D_point_to_kernel(x_vec, Ej, X, distance_fn)

    denom = denom_sum ** 2 if denom_sum != 0 else 1e-12
    return float(numerator / denom)