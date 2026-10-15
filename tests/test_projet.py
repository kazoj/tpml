"""Tests unitaires des fonctions du notebook (chargement, nettoyage, découpage, prétraitement, métriques)."""
import numpy as np
import pandas as pd
import pytest


# --- Chargement et nettoyage -------------------------------------------------

@pytest.mark.parametrize("kind, n_rows", [("red", 1599), ("white", 4898)])
def test_load_raw(nb, kind, n_rows):
    df = nb.load_raw(kind)
    assert df.shape == (n_rows, 12)
    assert "quality" in df.columns
    assert df.isna().sum().sum() == 0


def test_clean_toy_example(nb):
    df = pd.DataFrame({"alcohol": [9.0, 9.0, 12.0, 11.0], "quality": [5, 5, 7, 6]})
    out = nb.clean(df)
    assert len(out) == 3  # le doublon exact est retiré
    assert out["good"].tolist() == [0, 1, 0]  # bon vin si quality >= 7


@pytest.mark.parametrize("kind, n_rows", [("red", 1359), ("white", 3961)])
def test_load_clean_removes_duplicates(nb, kind, n_rows):
    df = nb.load_clean(kind)
    assert len(df) == n_rows
    assert not df.duplicated().any()
    assert set(df["good"].unique()) == {0, 1}


def test_load_clean_both_adds_wine_type(nb):
    df = nb.load_clean("both")
    assert len(df) == 1359 + 3961
    assert df["wine_type"].sum() == 1359  # 1 = rouge


# --- Découpage train/test ------------------------------------------------------

@pytest.mark.parametrize("kind", ["red", "white"])
def test_split_is_stratified_and_without_leak(nb, kind):
    X_tr, X_te, y_tr, y_te = nb.load_split(kind)
    n = len(X_tr) + len(X_te)
    assert len(X_te) == pytest.approx(0.2 * n, abs=1)
    assert not {"quality", "good"} & set(X_tr.columns)  # la cible n'est pas dans X
    assert X_tr.index.intersection(X_te.index).empty  # aucun vin à la fois en train et en test
    assert abs(y_tr.mean() - y_te.mean()) < 0.01  # même proportion de bons vins


def test_split_is_reproducible(nb):
    first, second = nb.load_split("red"), nb.load_split("red")
    assert first[1].index.equals(second[1].index)


# --- Prétraitement --------------------------------------------------------------

def test_make_preprocessor_log_and_drop(nb):
    X_tr = nb.load_split("white")[0]
    log_cols = ["chlorides", "residual sugar"]
    out = nb.make_preprocessor(log_cols, drop_cols=["density"], scaler=None).fit_transform(X_tr)
    assert "density" not in out.columns
    assert len(out.columns) == X_tr.shape[1] - 1
    for col in log_cols:
        np.testing.assert_allclose(out[col], np.log(X_tr[col]))
    np.testing.assert_allclose(out["alcohol"], X_tr["alcohol"])  # colonne non transformée


def test_make_preprocessor_standardizes(nb):
    X_tr = nb.load_split("red")[0]
    out = nb.make_preprocessor(["chlorides"]).fit_transform(X_tr)
    np.testing.assert_allclose(out.mean(), 0, atol=1e-9)
    np.testing.assert_allclose(out.std(ddof=0), 1, atol=1e-9)


def test_iqr_outliers(nb):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})
    assert nb.iqr_outliers(X)["a"].tolist() == [False, False, False, False, True]


# --- Métriques et seuil -----------------------------------------------------------

def test_evaluate_known_values(nb):
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    score = np.array([0.1, 0.6, 0.7, 0.9])
    m = nb.evaluate(y, pred, score)
    assert (m["VP"], m["FP"], m["FN"], m["VN"]) == (2, 1, 0, 1)
    assert m["précision"] == pytest.approx(2 / 3)
    assert m["rappel"] == 1.0
    assert m["spécificité"] == 0.5
    assert m["F1"] == pytest.approx(0.8)
    assert m["AUC"] == 1.0  # chaque bon vin a un score plus élevé que chaque mauvais vin


def test_best_f1_threshold_separable(nb):
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.4, 0.8, 0.9])
    threshold, *_, f1, _ = nb.best_f1_threshold(y, score)
    assert 0.4 < threshold <= 0.8
    assert f1.max() == pytest.approx(1.0)


# --- Test de fumée : le modèle final apprend quelque chose ------------------------------

def test_logistic_regression_beats_chance(nb):
    X_tr, _, y_tr, _ = nb.load_split("red")
    pipe = nb.Pipeline([("prep", nb.make_preprocessor(["chlorides", "residual sugar", "sulphates"])),
                        ("model", nb.LogisticRegression(class_weight="balanced", max_iter=2000))])
    scores = nb.cv_scores(pipe, X_tr, y_tr)
    assert scores["roc_auc"] > 0.8
