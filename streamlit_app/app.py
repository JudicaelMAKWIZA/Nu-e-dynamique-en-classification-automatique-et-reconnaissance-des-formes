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
n_etalon = st.number_input("Nombre d’étalons par classe", 1, 10, 1)
max_iter = st.number_input("Maximum d’itérations", 10, 500, 200)

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
            n_etalon=int(n_etalon),
            max_iter=int(max_iter),
            distance=distance_name,
            seed=0
        )

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
        st.subheader("Étalons")
        L = model.L_indices_
        for i, idx_list in enumerate(L):
            coords = [tuple(X[j]) for j in idx_list]
            st.write(f"Classe {i} → indices {idx_list} → coords {coords}")

        # ---------- VISUALISATION ----------
        labels = model.predict(X)

        fig, ax = plt.subplots(figsize=(6, 5))

        classes_ids = np.unique(labels)
        cmap = plt.get_cmap("tab10")

        # Dictionnaire : classe → couleur
        color_map = {cls: cmap(cls % 10) for cls in classes_ids}

        # Scatter cohérent : chaque point reçoit LA bonne couleur
        colors = [color_map[cls] for cls in labels]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, alpha=0.8)


        # Étendons
        for idx_list in L:
            if idx_list:
                pts = np.array([X[j] for j in idx_list])
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    marker="X", s=120,
                    edgecolor="black", color="red"
                )

        ax.set_title("Partition — Nuées Dynamiques")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # ------------------------------------------------------------------
        #           🔥 AJOUT DE LA LÉGENDE IDENTIFIANT LES CLASSES
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

        # ----------------------------------------------------------------------
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


