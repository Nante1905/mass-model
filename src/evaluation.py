"""Métriques par nuit, choix du seuil et intervalles par bootstrap par blocs de nuits."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, cohen_kappa_score, log_loss, roc_auc_score

import config

METRIQUES_PROBA = ["log_loss", "brier", "roc_auc"]
METRIQUES_DECISION = ["accuracy", "kappa"]
# Sens de l'amélioration, pour lire les différences appariées.
PLUS_BAS_MEILLEUR = {"log_loss", "brier"}


def metriques(y, p=None, decision=None):
    """p : probabilité (None pour une règle déterministe) ; decision : 0/1."""
    y = np.asarray(y, dtype=int)
    res = {}
    if p is not None:
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        res["log_loss"] = log_loss(y, p, labels=[0, 1])
        res["brier"] = brier_score_loss(y, p)
        res["roc_auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    res["accuracy"] = accuracy_score(y, decision)
    res["kappa"] = cohen_kappa_score(y, decision)
    res["taux_predit"] = float(np.mean(decision))
    res["taux_observe"] = float(np.mean(y))
    return res


def choisir_seuil(y, p):
    debut, fin, pas = config.GRILLE_SEUIL
    grille = np.round(np.arange(debut, fin + pas / 2, pas), 6)
    critere = {"kappa": cohen_kappa_score, "accuracy": accuracy_score}[config.CRITERE_SEUIL]
    scores = [critere(y, (p >= s).astype(int)) for s in grille]
    return float(grille[int(np.argmax(scores))])


def _indices_par_bloc(dates, longueur):
    """Blocs de `longueur` dates consécutives ; tous les sites d'une date vont ensemble."""
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    uniques = np.unique(dates)
    bloc = (np.arange(len(uniques)) // longueur)[np.searchsorted(uniques, dates)]
    return [np.flatnonzero(bloc == b) for b in range(bloc.max() + 1)]


def bootstrap_blocs(dates, y, modeles, reference, n_tirages):
    """Intervalles par bootstrap par blocs, avec les mêmes tirages pour tous les modèles.

    modeles : {nom: (p ou None, decision)}. Renvoie les intervalles de chaque modèle
    et ceux de la différence appariée reference − modèle, qui seule dit si l'écart
    entre deux modèles dépasse le bruit d'échantillonnage.
    """
    blocs = _indices_par_bloc(dates, config.BOOTSTRAP_LONGUEUR_BLOC)
    rng = np.random.default_rng(config.BOOTSTRAP_GRAINE)
    y = np.asarray(y, dtype=int)
    tirages = {nom: [] for nom in modeles}
    for _ in range(n_tirages):
        idx = np.concatenate([blocs[b] for b in rng.integers(0, len(blocs), len(blocs))])
        for nom, (p, d) in modeles.items():
            tirages[nom].append(metriques(y[idx], None if p is None else p[idx], d[idx]))
    tirages = {nom: pd.DataFrame(t) for nom, t in tirages.items()}

    a = (1 - config.BOOTSTRAP_NIVEAU) / 2
    colonnes = METRIQUES_PROBA + METRIQUES_DECISION

    def intervalle(serie):
        return (float(serie.quantile(a)), float(serie.quantile(1 - a)))

    ic = {nom: {c: intervalle(t[c]) for c in colonnes if c in t} for nom, t in tirages.items()}
    ref = tirages[reference]
    diff = {
        nom: {c: intervalle(ref[c] - t[c]) for c in colonnes if c in t and c in ref}
        for nom, t in tirages.items()
        if nom != reference
    }
    return ic, diff


def metriques_par_groupe(df, colonne, modeles):
    """Métriques de chaque modèle par valeur de `colonne` (site, saison)."""
    lignes = []
    for valeur, sous in df.groupby(colonne, sort=True):
        y = sous[config.CIBLE].to_numpy()
        for nom, (col_p, col_d) in modeles.items():
            p = None if col_p is None else sous[col_p].to_numpy()
            m = metriques(y, p, sous[col_d].to_numpy())
            lignes.append({colonne: valeur, "modele": nom, "n_nuits": len(sous), **m})
    return pd.DataFrame(lignes)
