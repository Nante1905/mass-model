"""MLP TensorFlow / Keras sur les agrégats par nuit."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402


def construire_mlp(x_entrainement):
    """Normalisation apprise sur l'entraînement seul, intégrée au modèle sauvegardé."""
    normalisation = keras.layers.Normalization(name="normalisation")
    normalisation.adapt(x_entrainement)
    regularisation = keras.regularizers.L2(config.MLP_L2)

    entree = keras.Input(shape=(x_entrainement.shape[1],), name="features_nuit")
    x = normalisation(entree)
    for i, unites in enumerate(config.MLP_COUCHES):
        x = keras.layers.Dense(
            unites, activation=config.MLP_ACTIVATION, kernel_regularizer=regularisation, name=f"dense_{i}"
        )(x)
        x = keras.layers.Dropout(config.MLP_DROPOUT, name=f"dropout_{i}")(x)
    sortie = keras.layers.Dense(1, activation="sigmoid", name="p_exploitable")(x)

    modele = keras.Model(entree, sortie, name="mlp_exploitabilite")
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.MLP_TAUX_APPRENTISSAGE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return modele


def entrainer_mlp(x_tr, y_tr, x_val, y_val, graine, epoques_max):
    """Entraîne une graine ; arrêt précoce sur la log-loss de validation 2022."""
    keras.utils.set_random_seed(graine)
    modele = construire_mlp(x_tr)
    rappels = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.MLP_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.MLP_REDUCTION_LR_FACTEUR,
            patience=config.MLP_REDUCTION_LR_PATIENCE,
            min_lr=config.MLP_LR_MIN,
        ),
    ]
    historique = modele.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=epoques_max,
        batch_size=config.MLP_TAILLE_LOT,
        class_weight=config.MLP_POIDS_CLASSES,
        shuffle=True,
        callbacks=rappels,
        verbose=config.MLP_VERBOSE,
    )
    return modele, pd.DataFrame(historique.history)
