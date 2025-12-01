# Nuee-dynamique-en-classification-automatique-et-reconnaissance-des-formes
Implémentation de la méthode des Nuées Dynamiques 

Cette application propose une implémentation fidèle de la **méthode des Nuées Dynamiques**, introduite par **Edmond Diday en 1971**.

## 📁 Structure du projet
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

Windows :
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/MacOs :

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

### 3. Lancer l'application Streamlit

```bash
streamlit run streamlit_app/app.py
