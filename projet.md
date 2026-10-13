# Classification de la qualité du vin

**Auteurs :** Ahmad TAY, Ahmed-Ghazi BLAIECH, Hassan SHRAIM
**Cours :** Machine Learning 1
**Année universitaire :** 2026-2027

---

## Sommaire

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Modalités du projet](#2-modalités-du-projet)
   - 2.1 [Prétraitement des données](#21-prétraitement-des-données)
   - 2.2 [Modélisation](#22-modélisation)
   - 2.3 [Évaluation des modèles](#23-évaluation-des-modèles)
   - 2.4 [Optimisation des hyperparamètres](#24-optimisation-des-hyperparamètres)
   - 2.5 [Bonus](#25-bonus)
3. [Livrables attendus](#3-livrables-attendus)

---

## 1. Contexte et objectif

La qualité d'un vin dépend de nombreux paramètres physico-chimiques : acidité, pH, taux d'alcool, teneur en sucre résiduel, etc.

**Objectif :** développer un modèle de classification supervisée capable de prédire la qualité d'un vin à partir de ces caractéristiques mesurables, soit sur l'échelle de notes (0 à 10), soit sous forme binaire (bon / mauvais).

**Données :** [Wine Quality – UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). Les fichiers sont également disponibles dans le dossier [datasets/](datasets/) :

| Fichier | Contenu |
|---|---|
| `winequality-red.csv` | Vins rouges |
| `winequality-white.csv` | Vins blancs |
| `winequality.names` | Description des variables |

---

## 2. Modalités du projet

### 2.1 Prétraitement des données

- Charger les données et calculer les statistiques descriptives.
- Gérer les valeurs manquantes et détecter les valeurs aberrantes.
- Encoder correctement les variables catégorielles (si présentes).
- Transformer le problème en classification binaire : qualité ≥ 7 ⇒ « bon vin », sinon « mauvais vin ».
- Analyser la distribution des classes afin de vérifier l'équilibre du jeu de données (et, le cas échéant, le traiter : pondération des classes, rééchantillonnage).
- Analyser les corrélations entre les variables.
- Produire des visualisations : matrice de corrélation, histogrammes, boxplots.
- Appliquer des méthodes de sélection de variables parmi celles vues en cours.

> **Bonnes pratiques :** fixer une graine aléatoire pour la reproductibilité, utiliser un découpage train/test stratifié, et effectuer la normalisation et la sélection de variables **à l'intérieur** de la validation croisée (ou d'un `Pipeline`) afin d'éviter toute fuite de données.

### 2.2 Modélisation

**Modèles envisagés :** régression logistique, k-NN, arbre de décision, Naive Bayes, SVM.

**Justification du choix des modèles** selon :

- leurs propriétés théoriques (modèles linéaires vs non linéaires) ;
- leur adaptation au type de données (variables continues, distributions, bruit, etc.).

| Modèle | Type | Remarque |
|---|---|---|
| Régression logistique | Linéaire | Référence simple et interprétable |
| k-NN | Non linéaire | Sensible à l'échelle des variables |
| Arbre de décision | Non linéaire | Interprétable, risque de surapprentissage |
| Naive Bayes | Linéaire / probabiliste | Hypothèse d'indépendance des variables |
| SVM | Linéaire ou non linéaire (selon le noyau) | Sensible à l'échelle, efficace en grande dimension |

### 2.3 Évaluation des modèles

- Évaluer les modèles à l'aide de métriques pertinentes :
  - accuracy, précision, rappel, F1-score ;
  - matrice de confusion ;
  - courbe ROC et AUC.
- Comparer les performances et interpréter les résultats.
- Identifier le modèle le plus performant selon un compromis biais/variance.

### 2.4 Optimisation des hyperparamètres

- Utiliser des techniques d'optimisation telles que :
  - **Grid Search** : recherche exhaustive sur une grille d'hyperparamètres ;
  - **Random Search** : recherche aléatoire, plus rapide.
- Évaluer les modèles optimisés par validation croisée (k-fold).

### 2.5 Bonus

- [ ] Publier la solution sur GitHub ou GitLab.
- [ ] Ajouter des tests unitaires.
- [ ] Conteneuriser le code (Docker) afin qu'il puisse être exécuté facilement par un tiers.
- [ ] Mettre en place une application web qui renvoie la prédiction pour une observation donnée en entrée.

---

## 3. Livrables attendus

- [ ] **Rapport technique** détaillant les étapes de traitement, d'analyse et de modélisation.
- [ ] **Notebook Python** (Jupyter / Colab) contenant le code complet.
