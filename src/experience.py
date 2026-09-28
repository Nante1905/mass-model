"""Socle commun des scripts d'entraînement : préparation, baselines, seuils, évaluation, sauvegarde.

Chaque script (entrainer_mlp.py, entrainer_lightgbm.py) n'implémente que l'entraînement
de son modèle ; tout le reste passe par ici. Les modèles sont ainsi comparés sur les
mêmes nuits, contre les mêmes baselines, avec les mêmes métriques.
"""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
from src.baselines import decision_era5, proba_climatologie, proba_majoritaire
from src.donnees import colonnes_entree, construire_table_nuits, decouper
from src.evaluation import (
    METRIQUES_DECISION,
    METRIQUES_PROBA,
    PLUS_BAS_MEILLEUR,
    bootstrap_blocs,
    choisir_seuil,
    metriques,
    metriques_par_groupe,
)

# nom du modèle -> (colonne de probabilité ou None, colonne de décision)
BASELINES = {
    "majoritaire": ("p_majoritaire", "d_majoritaire"),
    "climatologie": ("p_climatologie", "d_climatologie"),
    "era5_seuil": (None, "d_era5_seuil"),
}


def lire_arguments(doc):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["temporel", "spatial"], default=config.MODE_DECOUPAGE)
    parser.add_argument("--evaluer-test", action="store_true", help="évaluer aussi sur le test (une seule fois, à la fin)")
    parser.add_argument("--essai", action="store_true", help="version réduite : vérifie le pipeline, pas un résultat")
    return parser.parse_args()


def _decrire(parts, noms):
    lignes = [
        {"partie": n, "nuits": len(parts[n]), "sites": parts[n]["site_id"].nunique(),
         "annees": f"{parts[n]['annee'].min()}-{parts[n]['annee'].max()}",
         "taux_exploitable": round(parts[n][config.CIBLE].mean(), 3)}
        for n in noms
    ]
    print(pd.DataFrame(lignes).to_string(index=False))


def preparer_experience(nom, args):
    """Dossiers de sortie, table par nuit, garde-fou, découpage et matrices x / y."""
    suffixe = f"essai_{nom}_{args.mode}" if args.essai else f"{nom}_{args.mode}"
    dossier = config.DOSSIER_RESULTATS / suffixe
    dossier.mkdir(parents=True, exist_ok=True)
    dossier_modeles = config.DOSSIER_MODELES / suffixe
    dossier_modeles.mkdir(parents=True, exist_ok=True)

    nuits = construire_table_nuits()
    colonnes = colonnes_entree(nuits)
    parts = decouper(nuits, args.mode)
    evaluees = ["validation"] + ([p for p in parts if p.startswith("test")] if args.evaluer_test else [])

    print(f"{nom} — découpage {args.mode} — {len(colonnes)} features par nuit")
    _decrire(parts, ["entrainement"] + evaluees)

    x = {n: parts[n][colonnes].to_numpy(np.float32) for n in parts}
    y = {n: parts[n][config.CIBLE].to_numpy(np.float32) for n in parts}
    for n in ["entrainement"] + evaluees:
        assert np.isfinite(x[n]).all(), f"valeurs non finies dans {n}"

    return SimpleNamespace(
        nom=nom, args=args, dossier=dossier, dossier_modeles=dossier_modeles,
        colonnes=colonnes, parts=parts, evaluees=evaluees, x=x, y=y,
        n_bootstrap=config.ESSAI_BOOTSTRAP_N if args.essai else config.BOOTSTRAP_N,
    )


def resume_graine(exp, p_val, **infos):
    """Scores sans seuil d'une graine sur 2022, pour mesurer la dispersion entre graines."""
    m = metriques(exp.y["validation"], p_val, (p_val >= 0.5).astype(int))
    return {**infos, **{k: m[k] for k in METRIQUES_PROBA}}


def _formater(points, ic):
    colonnes = METRIQUES_PROBA + METRIQUES_DECISION
    lignes = {}
    for nom, m in points.items():
        lignes[nom] = {
            c: (f"{m[c]:.3f} [{ic[nom][c][0]:.3f}, {ic[nom][c][1]:.3f}]" if c in m else "—") for c in colonnes
        }
        lignes[nom]["taux_predit"] = f"{m['taux_predit']:.3f}"
    return pd.DataFrame(lignes).T


def _formater_diff(diff, reference):
    lignes = {}
    for nom, d in diff.items():
        lignes[f"{reference} − {nom}"] = {
            c: f"[{lo:+.3f}, {hi:+.3f}]" + (" *" if (hi < 0 if c in PLUS_BAS_MEILLEUR else lo > 0) else "")
            for c, (lo, hi) in d.items()
        }
    return pd.DataFrame(lignes).T


def _en_json(objet):
    if isinstance(objet, dict):
        return {str(k): _en_json(v) for k, v in objet.items()}
    if isinstance(objet, (list, tuple)):
        return [_en_json(v) for v in objet]
    if isinstance(objet, (np.floating, np.integer)):
        return objet.item()
    return objet


def evaluer_et_sauvegarder(exp, probas, resumes_graines, hyperparametres):
    """Moyenne des graines, seuils sur 2022, scores contre les baselines, sauvegardes.

    probas : {partie: [probabilités de chaque graine]}.
    """
    nom = exp.nom
    modeles = {**BASELINES, nom: (f"p_{nom}", f"d_{nom}")}
    resumes_graines = pd.DataFrame(resumes_graines)

    # Seuils choisis sur 2022 uniquement, puis figés pour toutes les parties.
    entrainement = exp.parts["entrainement"]
    seuils = {
        nom: choisir_seuil(exp.y["validation"], np.mean(probas["validation"], axis=0)),
        "climatologie": choisir_seuil(exp.y["validation"], proba_climatologie(entrainement, exp.parts["validation"])),
    }

    rapport = {
        "modele": nom,
        "mode": exp.args.mode,
        "essai": exp.args.essai,
        "hyperparametres": hyperparametres,
        "n_features": len(exp.colonnes),
        "features": exp.colonnes,
        "seuils": seuils,
        "critere_seuil": config.CRITERE_SEUIL,
        "dispersion_graines_validation": resumes_graines.to_dict(orient="list"),
        "parties": {},
    }

    for n in exp.evaluees:
        df = exp.parts[n][["site_id", "nuit", "annee", "mois", "saison", config.CIBLE, "met_cloud_cover__moy"]].copy()
        df[f"p_{nom}"] = np.mean(probas[n], axis=0)
        df["p_climatologie"] = proba_climatologie(entrainement, df)
        df["p_majoritaire"] = proba_majoritaire(entrainement, len(df))
        df[f"d_{nom}"] = (df[f"p_{nom}"] >= seuils[nom]).astype(int)
        df["d_climatologie"] = (df["p_climatologie"] >= seuils["climatologie"]).astype(int)
        df["d_majoritaire"] = (df["p_majoritaire"] >= 0.5).astype(int)
        df["d_era5_seuil"] = decision_era5(df)
        df.to_parquet(exp.dossier / f"predictions_{n}.parquet", index=False)

        y = df[config.CIBLE].to_numpy()
        tableaux = {m: (None if cp is None else df[cp].to_numpy(), df[cd].to_numpy()) for m, (cp, cd) in modeles.items()}
        points = {m: metriques(y, p, d) for m, (p, d) in tableaux.items()}
        ic, diff = bootstrap_blocs(df["nuit"], y, tableaux, reference=nom, n_tirages=exp.n_bootstrap)
        par_site = metriques_par_groupe(df, "site_id", modeles)
        par_saison = metriques_par_groupe(df, "saison", modeles)
        par_site.to_csv(exp.dossier / f"par_site_{n}.csv", index=False)
        par_saison.to_csv(exp.dossier / f"par_saison_{n}.csv", index=False)
        rapport["parties"][n] = {"points": points, "ic": ic, f"diff_{nom}_moins_modele": diff}

        print(f"\n=== {n} — {len(df)} nuits-site, IC {config.BOOTSTRAP_NIVEAU:.0%} bootstrap par blocs "
              f"de {config.BOOTSTRAP_LONGUEUR_BLOC} nuits ({exp.n_bootstrap} tirages) ===")
        print(_formater(points, ic).to_string())
        print(f"\nDifférences appariées (* : IC exclut 0 en faveur de {nom})")
        print(_formater_diff(diff, nom).to_string())
        colonnes_groupe = ["n_nuits", "log_loss", "brier", "roc_auc", "accuracy", "kappa"]
        for libelle, table in [("site", par_site), ("saison", par_saison)]:
            cle = table.columns[0]
            print(f"\nPar {libelle} :")
            du_modele = table[table["modele"] == nom].set_index(cle)[colonnes_groupe].add_suffix(f"_{nom}")
            kappas = (table[table["modele"] != nom]
                      .pivot(index=cle, columns="modele", values="kappa").add_prefix("kappa_"))
            print(du_modele.join(kappas).round(3).to_string())

    print(f"\nSeuils (choisis sur 2022, critère {config.CRITERE_SEUIL}) : {seuils}")
    print("Dispersion entre graines (validation, seuil 0,5) :")
    print(resumes_graines[METRIQUES_PROBA].agg(["mean", "std"]).round(4).to_string())

    with open(exp.dossier / "rapport.json", "w", encoding="utf-8") as f:
        json.dump(_en_json(rapport), f, ensure_ascii=False, indent=2)
    print(f"\nRésultats : {exp.dossier}\nModèles : {exp.dossier_modeles}")
