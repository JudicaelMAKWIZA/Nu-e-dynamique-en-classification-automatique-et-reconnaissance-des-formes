import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from nuees.nuees import NuesDynamiques


# ----------------------------------------------------------------------
# PAGE
# ----------------------------------------------------------------------
st.set_page_config(page_title="Nuées Dynamiques", page_icon="🎯")
st.title("Nuées Dynamiques — Implémentation")


st.markdown("""
Basé sur l'article de DIDAY (1971) : LA MÉTHODE DES NUÉES DYNAMIQUES  
""")

# ----------------------------------------------------------------------
# 1) GENERATION D’UN DATASET 2D
# ----------------------------------------------------------------------
st.header("1. Génération d’un dataset 2D")

gen_total = st.number_input("Nombre total de points", 10, 3000, 500)
gen_k_guess = st.number_input("Nombre de centres initiaux", 1, 10, 3)
gen_spread = st.slider("Dispersion", 0.5, 3.0, 1.25)

if st.button("Générer le dataset"):
    rng = np.random.RandomState(0)

    # Grille large = pas de doublons garanti
    grid_size = 500
    all_points = [(x, y) for x in range(grid_size) for y in range(grid_size)]

    # Centres tirés aléatoirement
    centers_idx = rng.choice(len(all_points), size=gen_k_guess, replace=False)
    centers = np.array([all_points[i] for i in centers_idx], dtype=float)

    # Répartition multinomiale
    counts = rng.multinomial(gen_total, [1.0 / gen_k_guess] * gen_k_guess)

    selected_points = []

    for i in range(gen_k_guess):
        cx, cy = centers[i]

        # Points candidats autour du centre
        candidate_points = []
        for dx in range(-50, 51):
            for dy in range(-50, 51):
                x = int(cx + dx * gen_spread)
                y = int(cy + dy * gen_spread)
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    candidate_points.append((x, y))

        chosen = rng.choice(len(candidate_points), size=counts[i], replace=False)
        selected_points += [candidate_points[j] for j in chosen]

    X_gen = np.array(selected_points)
    df_gen = pd.DataFrame(X_gen, columns=["x", "y"])
    st.dataframe(df_gen.head())

    # Téléchargement CSV
    csv_buf = BytesIO()
    df_gen.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    st.download_button(
        "Télécharger le CSV généré",
        csv_buf,
        "dataset_nuees.csv",
        "text/csv"
    )

st.markdown("---")

# ----------------------------------------------------------------------
# 2) UPLOAD CSV + CLUSTERING
# ----------------------------------------------------------------------
st.header("2. Téléverser un dataset")

file = st.file_uploader("Téléverser le dataset", type=["csv"])

k = st.number_input("Nombre de classes k", 2, 10, 3)
max_iter = st.number_input("Maximum d’itérations", 10, 500, 200)

# MODIFICATION: Choix du type de noyau
kernel_type = st.selectbox(
    "Type de noyau ($A_i$)",
    ["etalons", "centroide"]
)

# Gestion de n_etalon en fonction du type de noyau
if kernel_type == "etalons":
    n_etalon = st.number_input("Nombre d’étalons par classe", 1, 10, 1)
else:
    # Pour le centroïde, n_etalon est toujours 1 (conceptuellement), mais on le force à 1
    # juste pour satisfaire NuesDynamiques.__init__.
    n_etalon = 1

# choix de la distance
distance_name = st.selectbox(
    "Fonction de distance",
    ["euclidienne", "sebestyen", "chebychev", "chi2"]
)

if file is not None:

    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, header=None)

    df = df.select_dtypes(include=[np.number])
    if df.shape[1] != 2:
        st.error("Votre dataset doit contenir exactement 2 colonnes numériques (x,y).")
        st.stop()

    df.columns = ["x", "y"]
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[["x", "y"]].values
    st.dataframe(df.head())

    # ---------------------------
    # Lancement Nuées Dynamiques
    # ---------------------------
    if st.button("Lancer l'algorithme"):

        model = NuesDynamiques(
            k=int(k),
            n_etalon=int(n_etalon), # N'est pris en compte que si kernel_type='etalons'
            max_iter=int(max_iter),
            distance=distance_name,
            seed=0,
            kernel_type=kernel_type # PASSAGE DU PARAMÈTRE
        )

        # On utilise model.fit(X) sans sample_weights, donc les masses mu sont = 1
        model.fit(X)

        # ---------- CONVERGENCE ----------
        if model.converged_:
            st.success(f"✔ L'agorithme converge en {model.n_iter_} itérations.")
        else:
            st.warning(f"⚠ Pas de convergence après {model.n_iter_} itérations")

        st.subheader("Répartition des classes")
        for c, members in model.classes_.items():
            st.write(f"- Classe {c} : {len(members)} points")

        # ---------- HOMOGENEITE ----------
        st.subheader(" Homogénéité des classes")
        for c, h in model.class_homogeneity_.items():
            st.write(f"- Classe {c} : homogénéité moyenne = **{h:.4f}**")

        # ---------- QUALITÉ DE PARTITION ----------
        st.subheader("Valeur de la partition U(L)")
        st.write(f"U(L) = **{model.total_partition_quality_:.4f}**")

        # ---------- NOYAUX ----------
        st.subheader(f"Noyaux ($A_i$ - Type: {model.kernel_type})")
        # MODIFICATION: Utilisation de L_kernels_ pour l'affichage
        L = model.L_kernels_ 
        
        fig, ax = plt.subplots(figsize=(6, 5))

        # 1. Calculer les labels pour la couleur des points
        labels = model.predict(X)
        classes_ids = np.unique(labels)
        cmap = plt.get_cmap("tab10")
        color_map = {cls: cmap(cls % 10) for cls in classes_ids}

        # Scatter cohérent : chaque point reçoit LA bonne couleur
        colors = [color_map[cls] for cls in labels]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, alpha=0.8)

        # 2. Afficher les noyaux (Étalons ou Centroïdes)
        for i, kernel in enumerate(L):
            if kernel is None:
                st.write(f"Classe {i} → Noyau vide")
                continue
                
            if model.kernel_type == "etalons":
                # Le noyau est une liste d'indices de points réels
                idx_list = kernel
                coords = [tuple(int(c) for c in X[j]) for j in idx_list]
                st.write(f"Classe {i} → indices {idx_list} → coords {coords}")
                
                # Visualisation des étalons (marcheurs X)
                pts = np.array([X[j] for j in idx_list])
                ax.scatter(pts[:, 0], pts[:, 1], marker="X", s=120, edgecolor="black", color="red")
                
            else: # Centroide
                # Le noyau est un vecteur
                st.write(f"Classe {i} → centroïde {np.round(kernel, 4).tolist()}")
                
                # Visualisation des centroïdes (marcheurs X)
                ax.scatter(
                    kernel[0], kernel[1],
                    marker="X", s=120,
                    edgecolor="black", color="red"
                )

        ax.set_title(f"Partition — Nuées Dynamiques ({kernel_type})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # ------------------------------------------------------------------
        # AJOUT DE LA LÉGENDE IDENTIFIANT LES CLASSES
        # ------------------------------------------------------------------
        handles = []
        for cls in classes_ids:
            h = plt.Line2D(
                [], [],
                marker="o", linestyle="",
                markersize=8,
                markerfacecolor=color_map[cls],
                label=f"Classe {cls}"
            )
            handles.append(h)

        ax.legend(handles=handles, title="Classes", loc="upper right", frameon=True)


        st.pyplot(fig)

# FOOTER PERMANENT DANS LA SIDEBAR
# ----------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 20px; padding-top: 10px;'>
        © 2025 — Copyright <strong>Judicaël Makwiza</strong>
    </div>
    """,
    unsafe_allow_html=True
)