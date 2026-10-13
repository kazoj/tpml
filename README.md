# Classification de la qualité du vin

**Auteurs :** Ahmad TAY, Ahmed-Ghazi BLAIECH, Hassan SHRAIM
**Cours :** Machine Learning 1, 2026-2027

Prédire si un vin est « bon » (note ≥ 7) ou non à partir de 11 mesures physico-chimiques (alcool, acidité, sucre résiduel, SO₂…). Les vins rouges et blancs sont traités séparément. Le sujet complet est dans [projet.md](projet.md).

## Résultats

Modèle final : **régression logistique** (pénalité L2, `class_weight="balanced"`, log sur les variables asymétriques puis `StandardScaler`), avec un seuil de décision réglé par validation croisée. Évaluation unique sur le jeu de test (20 %, stratifié) :

| | AUC | F1 | Précision | Rappel |
|---|---|---|---|---|
| Vin rouge | 0,892 | 0,565 | 0,500 | 0,649 |
| Vin blanc | 0,835 | 0,577 | 0,457 | 0,782 |

Après réglage des hyperparamètres, les autres modèles (SVM RBF, k-NN, SVM linéaire, Naive Bayes) restent à moins d'un écart-type de la régression logistique : on garde la plus simple.

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| [projet.ipynb](projet.ipynb) | Notebook complet : prétraitement (2.1), modélisation (2.2), évaluation (2.3), optimisation des hyperparamètres et évaluation finale (2.4) |
| [rapport_projet_ml.pdf](rapport_projet_ml.pdf) | Rapport technique |
| [soutenance_projet_ml.pdf](soutenance_projet_ml.pdf) | Support de soutenance |
| `datasets/` | Données [Wine Quality (UCI)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) |
| `figures/` | Figures produites par le notebook |
| `requirements.txt`, `Dockerfile` | Environnement d'exécution |

## Lancer le projet

### Avec Docker (recommandé)

```bash
docker build -t wine-quality .
docker run --rm -p 127.0.0.1:8888:8888 wine-quality
```

Puis ouvrir <http://localhost:8888> et le notebook `projet.ipynb`. JupyterLab est lancé sans jeton d'accès : le port n'est donc publié que sur `127.0.0.1`.

Pour réexécuter tout le notebook sans interface (environ 3 minutes ; le notebook et le dossier `figures/` du dossier courant sont mis à jour) :

```bash
docker run --rm -v "$PWD":/app wine-quality \
  jupyter nbconvert --to notebook --execute --inplace projet.ipynb
```

### En local (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Toutes les étapes aléatoires utilisent une graine fixe (`SEED = 42`) : les résultats sont reproductibles.
