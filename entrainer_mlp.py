"""Premier modèle de l'étage stochastique : MLP TensorFlow / Keras sur les agrégats par nuit.

Usage (depuis la racine du projet, venv activé) :
    python entrainer_mlp.py --essai              # vérification rapide du pipeline, pas un résultat
    python entrainer_mlp.py                      # découpage temporel, évaluation sur la validation 2022
    python entrainer_mlp.py --mode spatial       # Tana jamais vu
    python entrainer_mlp.py --evaluer-test       # à ne lancer qu'une fois, à la fin

Les métriques de validation 2022 sont optimistes par construction : 2022 sert à
l'arrêt précoce et au choix du seuil. Le chiffre honnête est celui du test.
"""

import tensorflow as tf

import config
from src.experience import evaluer_et_sauvegarder, lire_arguments, preparer_experience, resume_graine
from src.mlp import entrainer_mlp


def hyperparametres():
    return {
        "couches": config.MLP_COUCHES,
        "activation": config.MLP_ACTIVATION,
        "dropout": config.MLP_DROPOUT,
        "l2": config.MLP_L2,
        "taux_apprentissage": config.MLP_TAUX_APPRENTISSAGE,
        "taille_lot": config.MLP_TAILLE_LOT,
        "epoques_max": config.MLP_EPOQUES_MAX,
        "patience": config.MLP_PATIENCE,
    }


def main():
    args = lire_arguments(__doc__)
    tf.config.experimental.enable_op_determinism()
    exp = preparer_experience("mlp", args)

    graines = config.MLP_GRAINES[:1] if args.essai else config.MLP_GRAINES
    epoques_max = config.ESSAI_EPOQUES if args.essai else config.MLP_EPOQUES_MAX

    probas = {n: [] for n in exp.evaluees}
    resumes = []
    for graine in graines:
        modele, historique = entrainer_mlp(
            exp.x["entrainement"], exp.y["entrainement"], exp.x["validation"], exp.y["validation"], graine, epoques_max
        )
        historique.to_csv(exp.dossier / f"historique_graine{graine}.csv", index_label="epoque")
        modele.save(exp.dossier_modeles / f"graine{graine}.keras")
        for n in exp.evaluees:
            probas[n].append(modele.predict(exp.x[n], verbose=0).ravel())
        resume = resume_graine(
            exp, probas["validation"][-1],
            graine=graine, epoques=len(historique), meilleure_epoque=int(historique["val_loss"].idxmin()) + 1,
        )
        resumes.append(resume)
        print(f"graine {graine} : {resume['epoques']} époques (meilleure {resume['meilleure_epoque']}), "
              f"log-loss val {resume['log_loss']:.4f}, AUC val {resume['roc_auc']:.4f}")

    evaluer_et_sauvegarder(exp, probas, resumes, hyperparametres())


if __name__ == "__main__":
    main()
