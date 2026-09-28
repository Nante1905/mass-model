"""Chargement, agrégation par nuit et découpages.

L'unité d'évaluation est la nuit : le label est constant sur la nuit, et compter
chaque heure gonflerait l'effectif en surpondérant les longues nuits d'hiver.
"""

import numpy as np
import pandas as pd

import config

CLES = ["site_id", "nuit"]


def charger_horaire():
    """Features horaires, heures sans label exclues (602 heures, non imputées)."""
    df = pd.read_parquet(config.CHEMIN_FEATURES_HORAIRES)
    df["nuit"] = pd.to_datetime(df["nuit"])
    return df.dropna(subset=[config.CIBLE]).reset_index(drop=True)


def agreger_par_nuit(df):
    """Une ligne par (site, nuit) : statistiques, valeurs ponctuelles, contexte."""
    df = df.sort_values(CLES + ["instant_utc"]).reset_index(drop=True)
    for v in config.VARIABLES_LOG1P:
        df[v] = np.log1p(df[v])
    variables = config.VARIABLES_METEO
    g = df.groupby(CLES, sort=True)

    assert (g[config.CIBLE].nunique() == 1).all(), "label non constant sur une nuit"

    stats = g[variables].agg(list(config.AGREGATS))
    stats.columns = [f"{v}__{config.AGREGATS[s]}" for v, s in stats.columns]

    rang = g.cumcount()
    n_heures = g["instant_utc"].transform("size")
    masques = {"debut": rang == 0, "milieu": rang == n_heures // 2, "fin": rang == n_heures - 1}
    ponctuels = []
    for nom in config.POSITIONS_NUIT:
        p = df.loc[masques[nom], CLES + variables].set_index(CLES)
        p.columns = [f"{v}__{nom}" for v in variables]
        ponctuels.append(p)

    precip = (df["met_weather_code"] >= config.WEATHER_CODE_PRECIP_MIN).astype(float)
    code = df.assign(_precip=precip).groupby(CLES, sort=True).agg(
        met_weather_code__max=("met_weather_code", "max"),
        met_weather_code__frac_precip=("_precip", "mean"),
    )

    meta = g[["annee", "split", "split_spatial", config.CIBLE] + config.FEATURES_CONTEXTE].first()

    nuits = pd.concat([meta, stats, *ponctuels, code], axis=1).reset_index()
    nuits[config.CIBLE] = nuits[config.CIBLE].astype(int)
    nuits["mois"] = nuits["nuit"].dt.month
    nuits["saison"] = np.where(nuits["mois"].isin(config.MOIS_SAISON_HUMIDE), "humide", "seche")
    return nuits


def construire_table_nuits():
    """Table par nuit, recalculée depuis les données horaires et mise en cache."""
    nuits = agreger_par_nuit(charger_horaire())
    config.CHEMIN_FEATURES_NUITS.parent.mkdir(parents=True, exist_ok=True)
    nuits.to_parquet(config.CHEMIN_FEATURES_NUITS, index=False)
    return nuits


def verifier_entrees(colonnes):
    """Assert anti-circularité : seules des features met_ et le contexte autorisé entrent."""
    interdites = [c for c in colonnes if c.startswith(config.PREFIXES_INTERDITS)]
    assert not interdites, f"colonnes interdites en entrée : {interdites}"
    exclues = [c for c in colonnes if c.split("__")[0] in config.VARIABLES_EXCLUES]
    assert not exclues, f"variables exclues (CAMS absent avant 2022) en entrée : {exclues}"
    inconnues = [c for c in colonnes if not c.startswith("met_") and c not in config.FEATURES_CONTEXTE]
    assert not inconnues, f"colonnes hors met_ et hors contexte autorisé : {inconnues}"


def colonnes_entree(nuits):
    colonnes = [c for c in nuits.columns if c.startswith("met_")] + config.FEATURES_CONTEXTE
    verifier_entrees(colonnes)
    return colonnes


def decouper(nuits, mode):
    """Parties entrainement / validation / test selon le découpage demandé.

    En spatial, la validation reste l'année 2022 des sept autres sites, pour que
    l'arrêt précoce et le choix du seuil ne voient jamais Antananarivo. Le test
    Tana est rapporté sur toutes les années et sur 2023-2024 seules (test à la
    fois spatial et temporel).
    """
    split = nuits["split"]
    if mode == "temporel":
        masques = {
            "entrainement": split == "entrainement",
            "validation": split == "validation",
            "test": split == "test",
        }
    elif mode == "spatial":
        autres = nuits["split_spatial"] == "entrainement"
        tana = nuits["split_spatial"] == "test"
        masques = {
            "entrainement": autres & (split == "entrainement"),
            "validation": autres & (split == "validation"),
            "test": tana,
            "test_2023_2024": tana & (split == "test"),
        }
    else:
        raise ValueError(f"mode de découpage inconnu : {mode}")
    return {nom: nuits[m].reset_index(drop=True) for nom, m in masques.items()}
