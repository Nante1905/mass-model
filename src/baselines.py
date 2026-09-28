"""Les trois références à battre : classe majoritaire, climatologie site × mois, seuil ERA5."""

import numpy as np

import config


def proba_majoritaire(entrainement, n):
    """Probabilité constante : fréquence d'exploitabilité sur l'entraînement."""
    return np.full(n, entrainement[config.CIBLE].mean())


def proba_climatologie(entrainement, df):
    """Fréquence d'exploitabilité par (site, mois) apprise sur l'entraînement.

    Repli sur le mois seul pour un site absent de l'entraînement (Tana en spatial),
    puis sur la fréquence globale.
    """
    cible = config.CIBLE
    par_site_mois = entrainement.groupby(["site_id", "mois"])[cible].mean().rename("p")
    par_mois = entrainement.groupby("mois")[cible].mean().rename("p_mois")
    out = df[["site_id", "mois"]].join(par_site_mois, on=["site_id", "mois"]).join(par_mois, on="mois")
    return out["p"].fillna(out["p_mois"]).fillna(entrainement[cible].mean()).to_numpy()


def decision_era5(df):
    """Nébulosité ERA5 moyenne de la nuit < 30 % -> exploitable (règle du label)."""
    return (df["met_cloud_cover__moy"] < config.SEUIL_ERA5_EXPLOITABLE).astype(int).to_numpy()
