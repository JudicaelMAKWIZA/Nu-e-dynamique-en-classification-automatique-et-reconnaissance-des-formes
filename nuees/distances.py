"""
Distances utilisées pour Nuées Dynamiques.
Contient :
- euclidienne
- sebestyen (distance standardisée)
- chebychev
- chi2 (chi-carré)
"""

import numpy as np

def euclidienne(x, y):
    """Distance euclidienne classique entre deux vecteurs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.linalg.norm(x - y))


def sebestyen(x, y, variances=None, eps=1e-12):
    """
    Distance 'Sébéstyen'  :
    d = sqrt( sum_i ( (x_i - y_i)^2 / s_i^2 ) )
    où s_i^2 est la variance de la dimension i (si fournie).
    Si variances is None -> comportement similaire à l'euclidienne.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if variances is None:
        return euclidienne(x, y)
    variances = np.asarray(variances, dtype=float)
    variances = np.where(variances <= 0, eps, variances)
    return float(np.sqrt(np.sum(((x - y) ** 2) / variances)))


def chebychev(x, y):
    """Distance de Chebychev (norme infinie) : max |x_i - y_i|"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.max(np.abs(x - y)))


def chi2(x, y, eps=1e-12):
    """
    Distance chi-carré (chi²) entre deux vecteurs.
    Formule utilisée (classique pour histogrammes) :
        d = sum_i ( (x_i - y_i)^2 / (x_i + y_i + eps) )
    - Contrainte pratique : x and y devraient être >= 0 (histogrammes).
    - eps évite division par zéro.
    - Retourne une valeur >= 0 (non racine, on garde la somme directe, c'est courant pour chi2).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = x + y + eps
    return float(np.sum((x - y) ** 2 / denom))
