# Nuee-dynamique-en-classification-automatique-et-reconnaissance-des-formes

Cette application propose une implémentation de la **méthode des Nuées Dynamiques**, introduite par **Edmond Diday en 1971**.

## Structure du projet
```
.
├── nuees/ # Package Python : logique des Nuées Dynamiques
│ ├── distances.py
│ ├── nuees.py
│ ├── r_function.py
│ └── init.py
│
├── streamlit_app/ # Interface Streamlit
│ └── app.py
│
├── tools/ # Scripts utilitaires (dataset)
│ └── generate_datasets.py
│
├── tests/ # Tests simples
│ └── test_basic.py
│
├── pyproject.toml
├── setup.cfg
├── requirements.txt
├── README.md
└── .gitignore
```
---

## Installation locale

### 1. Créer un environnement virtuel (optionnel)

```bash
python -m venv venv
```

---

### 2. Activer l'environnement virtuel

Windows :
```bash
venv\Scripts\activate
```

Linux/MacOs :

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

### 4. Lancer l'application Streamlit

```bash
streamlit run streamlit_app/app.py
