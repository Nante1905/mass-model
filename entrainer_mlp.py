"""Premier modèle de l'étage stochastique : MLP TensorFlow / Keras sur les agrégats par nuit.

Usage (depuis la racine du projet, venv activé) :
    python entrainer_mlp.py --essai              # vérification rapide du pipeline, pas un résultat
    python entrainer_mlp.py                      # découpage temporel, évaluation sur la validation 2022
    python entrainer_mlp.py --mode spatial       # Tana jamais vu
    python entrainer_mlp.py --evaluer-test       # à ne lancer qu'une fois, à la fin

Les métriques de validation 2022 sont optimistes par construction : 2022 sert à
l'arrêt précoce et au choix du seuil. Le chiffre honnête est celui du test.
"""

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

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
from src.mlp import entrainer_mlp

# nom du modèle -> (colonne de probabilité ou None, colonne de décision)
MODELES = {
    "majoritaire": ("p_majoritaire", "d_majoritaire"),
    "climatologie": ("p_climatologie", "d_climatologie"),
    "era5_seuil": (None, "d_era5_seuil"),
    "mlp": ("p_mlp", "d_mlp"),
}


def lire_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["temporel", "spatial"], default=config.MODE_DECOUPAGE)
    parser.add_argument("--evaluer-test", action="store_true", help="évaluer aussi sur le test (une seule fois, à la fin)")
    parser.add_argument("--essai", action="store_true", help="1 graine, quelques époques : vérifie le pipeline")
    return parser.parse_args()


def decrire(parts, noms):
    lignes = [
        {"partie": n, "nuits": len(parts[n]), "sites": parts[n]["site_id"].nunique(),
         "annees": f"{parts[n]['annee'].min()}-{parts[n]['annee'].max()}",
         "taux_exploitable": round(parts[n][config.CIBLE].mean(), 3)}
        for n in noms
    ]
    print(pd.DataFrame(lignes).to_string(index=False))


def tableau_global(y, df, dates, n_bootstrap):
    """Point + intervalle bootstrap par modèle, et différences appariées mlp − baseline."""
    modeles = {
        nom: (None if cp is None else df[cp].to_numpy(), df[cd].to_numpy()) for nom, (cp, cd) in MODELES.items()
    }
    points = {nom: metriques(y, p, d) for nom, (p, d) in modeles.items()}
    ic, diff = bootstrap_blocs(dates, y, modeles, reference="mlp", n_tirages=n_bootstrap)
    return points, ic, diff


def formater(points, ic):
    colonnes = METRIQUES_PROBA + METRIQUES_DECISION
    lignes = {}
    for nom, m in points.items():
        lignes[nom] = {
            c: (f"{m[c]:.3f} [{ic[nom][c][0]:.3f}, {ic[nom][c][1]:.3f}]" if c in m else "—") for c in colonnes
        }
        lignes[nom]["taux_predit"] = f"{m['taux_predit']:.3f}"
    return pd.DataFrame(lignes).T


def formater_diff(diff):
    lignes = {}
    for nom, d in diff.items():
        lignes[f"mlp − {nom}"] = {
            c: f"[{lo:+.3f}, {hi:+.3f}]" + (" *" if (hi < 0 if c in PLUS_BAS_MEILLEUR else lo > 0) else "")
            for c, (lo, hi) in d.items()
        }
    return pd.DataFrame(lignes).T


def en_json(objet):
    if isinstance(objet, dict):
        return {str(k): en_json(v) for k, v in objet.items()}
    if isinstance(objet, (list, tuple)):
        return [en_json(v) for v in objet]
    if isinstance(objet, (np.floating, np.integer)):
        return objet.item()
    return objet


def main():
    args = lire_arguments()
    tf.config.experimental.enable_op_determinism()

    suffixe = f"essai_mlp_{args.mode}" if args.essai else f"mlp_{args.mode}"
    dossier = config.DOSSIER_RESULTATS / suffixe
    dossier.mkdir(parents=True, exist_ok=True)
    dossier_modeles = config.DOSSIER_MODELES / suffixe
    dossier_modeles.mkdir(parents=True, exist_ok=True)

    nuits = construire_table_nuits()
    colonnes = colonnes_entree(nuits)
    parts = decouper(nuits, args.mode)
    evaluees = ["validation"] + ([p for p in parts if p.startswith("test")] if args.evaluer_test else [])

    print(f"Découpage {args.mode} — {len(colonnes)} features par nuit")
    decrire(parts, ["entrainement"] + evaluees)

    x = {n: parts[n][colonnes].to_numpy(np.float32) for n in parts}
    y = {n: parts[n][config.CIBLE].to_numpy(np.float32) for n in parts}
    for n in ["entrainement"] + evaluees:
        assert np.isfinite(x[n]).all(), f"valeurs non finies dans {n}"

    graines = config.MLP_GRAINES[:1] if args.essai else config.MLP_GRAINES
    epoques_max = config.ESSAI_EPOQUES if args.essai else config.MLP_EPOQUES_MAX
    n_bootstrap = config.ESSAI_BOOTSTRAP_N if args.essai else config.BOOTSTRAP_N

    probas = {n: [] for n in evaluees}
    resume_graines = []
    for graine in graines:
        modele, historique = entrainer_mlp(
            x["entrainement"], y["entrainement"], x["validation"], y["validation"], graine, epoques_max
        )
        historique.to_csv(dossier / f"historique_graine{graine}.csv", index_label="epoque")
        modele.save(dossier_modeles / f"graine{graine}.keras")
        for n in evaluees:
            probas[n].append(modele.predict(x[n], verbose=0).ravel())
        p_val = probas["validation"][-1]
        resume = {
            "graine": graine,
            "epoques": len(historique),
            "meilleure_epoque": int(historique["val_loss"].idxmin()) + 1,
            **{k: v for k, v in metriques(y["validation"], p_val, (p_val >= 0.5).astype(int)).items()
               if k in METRIQUES_PROBA},
        }
        resume_graines.append(resume)
        print(f"graine {graine} : {resume['epoques']} époques (meilleure {resume['meilleure_epoque']}), "
              f"log-loss val {resume['log_loss']:.4f}, AUC val {resume['roc_auc']:.4f}")
    resume_graines = pd.DataFrame(resume_graines)

    # Seuils choisis sur 2022 uniquement, puis figés pour toutes les parties.
    entrainement = parts["entrainement"]
    p_mlp_val = np.mean(probas["validation"], axis=0)
    seuils = {
        "mlp": choisir_seuil(y["validation"], p_mlp_val),
        "climatologie": choisir_seuil(y["validation"], proba_climatologie(entrainement, parts["validation"])),
    }

    rapport = {
        "mode": args.mode,
        "essai": args.essai,
        "n_features": len(colonnes),
        "features": colonnes,
        "graines": list(graines),
        "seuils": seuils,
        "critere_seuil": config.CRITERE_SEUIL,
        "dispersion_graines_validation": resume_graines.to_dict(orient="list"),
        "parties": {},
    }

    for n in evaluees:
        df = parts[n][["site_id", "nuit", "annee", "mois", "saison", config.CIBLE, "met_cloud_cover__moy"]].copy()
        df["p_mlp"] = np.mean(probas[n], axis=0)
        df["p_climatologie"] = proba_climatologie(entrainement, df)
        df["p_majoritaire"] = proba_majoritaire(entrainement, len(df))
        df["d_mlp"] = (df["p_mlp"] >= seuils["mlp"]).astype(int)
        df["d_climatologie"] = (df["p_climatologie"] >= seuils["climatologie"]).astype(int)
        df["d_majoritaire"] = (df["p_majoritaire"] >= 0.5).astype(int)
        df["d_era5_seuil"] = decision_era5(df)
        df.to_parquet(dossier / f"predictions_{n}.parquet", index=False)

        points, ic, diff = tableau_global(df[config.CIBLE].to_numpy(), df, df["nuit"], n_bootstrap)
        par_site = metriques_par_groupe(df, "site_id", MODELES)
        par_saison = metriques_par_groupe(df, "saison", MODELES)
        par_site.to_csv(dossier / f"par_site_{n}.csv", index=False)
        par_saison.to_csv(dossier / f"par_saison_{n}.csv", index=False)
        rapport["parties"][n] = {"points": points, "ic": ic, "diff_mlp_moins_modele": diff}

        print(f"\n=== {n} — {len(df)} nuits-site, IC {config.BOOTSTRAP_NIVEAU:.0%} bootstrap par blocs "
              f"de {config.BOOTSTRAP_LONGUEUR_BLOC} nuits ({n_bootstrap} tirages) ===")
        print(formater(points, ic).to_string())
        print("\nDifférences appariées (* : IC exclut 0 en faveur du MLP)")
        print(formater_diff(diff).to_string())
        colonnes_groupe = ["n_nuits", "log_loss", "brier", "roc_auc", "accuracy", "kappa"]
        for nom, table in [("site", par_site), ("saison", par_saison)]:
            cle = table.columns[0]
            print(f"\nPar {nom} :")
            mlp = table[table["modele"] == "mlp"].set_index(cle)[colonnes_groupe].add_suffix("_mlp")
            kappas = (table[table["modele"] != "mlp"]
                      .pivot(index=cle, columns="modele", values="kappa").add_prefix("kappa_"))
            print(mlp.join(kappas).round(3).to_string())

    print(f"\nSeuils (choisis sur 2022, critère {config.CRITERE_SEUIL}) : {seuils}")
    print("Dispersion entre graines (validation, seuil 0,5) :")
    print(resume_graines[METRIQUES_PROBA].agg(["mean", "std"]).round(4).to_string())

    with open(dossier / "rapport.json", "w", encoding="utf-8") as f:
        json.dump(en_json(rapport), f, ensure_ascii=False, indent=2)
    print(f"\nRésultats : {dossier}\nModèles : {dossier_modeles}")


if __name__ == "__main__":
    main()
