"""Gradient boosting LightGBM sur les agrégats par nuit, comparé aux mêmes baselines que le MLP.

Usage (depuis la racine du projet, venv activé) :
    python entrainer_lightgbm.py --essai              # vérification rapide du pipeline, pas un résultat
    python entrainer_lightgbm.py                      # découpage temporel, évaluation sur la validation 2022
    python entrainer_lightgbm.py --mode spatial       # Tana jamais vu
    python entrainer_lightgbm.py --evaluer-test       # à ne lancer qu'une fois, à la fin

Les métriques de validation 2022 sont optimistes par construction : 2022 sert à
l'arrêt précoce et au choix du seuil. Le chiffre honnête est celui du test.
"""

import pandas as pd

import config
from src.experience import evaluer_et_sauvegarder, lire_arguments, preparer_experience, resume_graine
from src.gbm import entrainer_lgbm, importances, predire


def hyperparametres():
    return {
        **config.LGBM_PARAMS,
        "arbres_max": config.LGBM_ARBRES_MAX,
        "arret_precoce": config.LGBM_ARRET_PRECOCE,
    }


def main():
    args = lire_arguments(__doc__)
    exp = preparer_experience("lightgbm", args)

    graines = config.LGBM_GRAINES[:1] if args.essai else config.LGBM_GRAINES
    arbres_max = config.ESSAI_LGBM_ARBRES if args.essai else config.LGBM_ARBRES_MAX

    probas = {n: [] for n in exp.evaluees}
    resumes = []
    gains = {}
    for graine in graines:
        booster, historique = entrainer_lgbm(
            exp.x["entrainement"], exp.y["entrainement"], exp.x["validation"], exp.y["validation"],
            exp.colonnes, graine, arbres_max,
        )
        historique.to_csv(exp.dossier / f"historique_graine{graine}.csv", index_label="arbre")
        booster.save_model(str(exp.dossier_modeles / f"graine{graine}.txt"), num_iteration=booster.best_iteration)
        for n in exp.evaluees:
            probas[n].append(predire(booster, exp.x[n]))
        gains[f"graine{graine}"] = importances(booster)
        resume = resume_graine(
            exp, probas["validation"][-1], graine=graine, arbres=booster.best_iteration,
        )
        resumes.append(resume)
        print(f"graine {graine} : {resume['arbres']} arbres retenus, "
              f"log-loss val {resume['log_loss']:.4f}, AUC val {resume['roc_auc']:.4f}")

    gains = pd.DataFrame(gains)
    gains["moyenne"] = gains.mean(axis=1)
    gains = gains.sort_values("moyenne", ascending=False)
    gains.to_csv(exp.dossier / "importance_features.csv", index_label="feature")
    print("\nImportance des features (part du gain total, moyenne des graines) — 15 premières :")
    print(gains["moyenne"].head(15).round(3).to_string())

    evaluer_et_sauvegarder(exp, probas, resumes, hyperparametres())


if __name__ == "__main__":
    main()
