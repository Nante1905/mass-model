"""Gradient boosting LightGBM sur les agrégats par nuit."""

import lightgbm as lgb
import pandas as pd

import config


def entrainer_lgbm(x_tr, y_tr, x_val, y_val, noms_features, graine, arbres_max):
    """Entraîne une graine ; arrêt précoce sur la log-loss de validation 2022."""
    params = {**config.LGBM_PARAMS, "seed": graine, "num_threads": config.LGBM_NUM_THREADS}
    d_tr = lgb.Dataset(x_tr, y_tr, feature_name=noms_features)
    d_val = lgb.Dataset(x_val, y_val, reference=d_tr)
    evaluations = {}
    booster = lgb.train(
        params,
        d_tr,
        num_boost_round=arbres_max,
        valid_sets=[d_tr, d_val],
        valid_names=["entrainement", "validation"],
        callbacks=[
            lgb.early_stopping(config.LGBM_ARRET_PRECOCE, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evaluations),
        ],
    )
    historique = pd.DataFrame(
        {f"{metrique}_{partie}": valeurs for partie, d in evaluations.items() for metrique, valeurs in d.items()}
    )
    return booster, historique


def predire(booster, x):
    return booster.predict(x, num_iteration=booster.best_iteration)


def importances(booster):
    """Gain total apporté par chaque feature, en part du gain de l'ensemble du modèle."""
    gain = pd.Series(
        booster.feature_importance("gain", iteration=booster.best_iteration), index=booster.feature_name()
    )
    return gain / gain.sum()
