# Nuee-dynamique-en-classification-automatique-et-reconnaissance-des-formes
Implémentation de la méthode des Nuées Dynamiques 


Ce projet propose une implémentation **entièrement from-scratch** de la méthode des **Nuées Dynamiques (Diday)**, un algorithme de classification non supervisée proche du K-means mais basé sur une fonction de pertinence **R(x, i, L)**.

L’implémentation est pédagogique, documentée en français, et permet :
- de choisir **4 distances** :  
  - Euclidienne  
  - Sébestyen (distance standardisée)  
  - Mahalanobis  
  - Chebychev  

- d’effectuer du clustering sur n’importe quel dataset numérique  
- de tester l'algorithme via une interface **Streamlit** incluse

---

## 📌 Installation locale

```bash
pip install -e .
