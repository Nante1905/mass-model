"""Seuils, hyperparamètres et chemins du projet.

Tout choix qui se justifie en soutenance est ici, et nulle part ailleurs en dur.
"""

from pathlib import Path

# --- Chemins -----------------------------------------------------------------

RACINE = Path(__file__).resolve().parent
DOSSIER_DONNEES = RACINE / "data"
CHEMIN_FEATURES_HORAIRES = DOSSIER_DONNEES / "features_horaires.parquet"
CHEMIN_FEATURES_NUITS = DOSSIER_DONNEES / "derive" / "features_nuits.parquet"
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_MODELES = RACINE / "modeles"

# --- Cible -------------------------------------------------------------------

# Cible binaire (1 = exploitable), testée en « < 30 % » : c'est la définition de
# référence. Le label trois classes (pd.cut, borne fermée à droite) n'est pas
# utilisé ici, précisément à cause de son incohérence de frontière.
CIBLE = "label_binaire"

# Baseline ERA5 : même règle que le label, appliquée à la nébulosité moyenne ERA5.
SEUIL_ERA5_EXPLOITABLE = 30.0  # en %

# --- Features ----------------------------------------------------------------

# Variables météo horaires agrégées par nuit. met_aod et met_poussiere sont
# exclues : l'archive CAMS ne commence qu'en août 2022, donc absentes de tout
# l'entraînement. met_weather_code est traité à part (code WMO nominal).
VARIABLES_METEO = [
    "met_cloud_cover",
    "met_cloud_cover_low",
    "met_cloud_cover_mid",
    "met_cloud_cover_high",
    "met_temperature_2m",
    "met_relative_humidity_2m",
    "met_dew_point_2m",
    "met_vapour_pressure_deficit",
    "met_precipitation",
    "met_wind_speed_10m",
    "met_wind_gusts_10m",
    "met_surface_pressure",
]
VARIABLES_EXCLUES = ["met_aod", "met_poussiere"]

# Précipitations très asymétriques (majorité de zéros) : log1p avant agrégation.
VARIABLES_LOG1P = ["met_precipitation"]

# Statistiques sur la nuit (nom pandas -> suffixe de colonne).
AGREGATS = {"mean": "moy", "min": "min", "max": "max", "std": "std"}

# Valeurs ponctuelles : première heure, heure médiane, dernière heure de la nuit.
# Elles portent l'évolution intra-nuit (dissipation ou formation de nuages).
POSITIONS_NUIT = ["debut", "milieu", "fin"]

# Codes WMO >= 51 : bruine ou pluie. On en tire la fraction d'heures précipitantes
# et le code maximal (ordinal en intensité), pas une moyenne de codes nominaux.
WEATHER_CODE_PRECIP_MIN = 51

# Contexte non météo autorisé en entrée. altitude et latitude identifient les sites
# (voulu en temporel ; en spatial, Tana est une interpolation entre sites).
# Le site_id n'entre pas : il empêcherait toute généralisation à un site non vu.
FEATURES_CONTEXTE = ["tmp_doy_sin", "tmp_doy_cos", "altitude_site", "latitude_site"]

# Garde-fou anti-circularité et anti-fuite : aucune colonne de ces familles en entrée.
PREFIXES_INTERDITS = ("astro_", "sat_", "label", "split")

# --- Découpage et analyse ----------------------------------------------------

MODE_DECOUPAGE = "temporel"  # "temporel" ou "spatial"

# Saison des pluies australe à Madagascar : novembre à avril.
MOIS_SAISON_HUMIDE = (11, 12, 1, 2, 3, 4)

# --- MLP (TensorFlow / Keras) ------------------------------------------------

# Réseau volontairement petit : ~14 000 nuits-site d'entraînement pour n_eff ≈ 2
# sites indépendants. Deux couches suffisent à des interactions non linéaires
# entre agrégats ; au-delà, on ne ferait que mémoriser.
MLP_COUCHES = (64, 32)
MLP_ACTIVATION = "relu"
MLP_DROPOUT = 0.3
MLP_L2 = 1e-4
MLP_TAUX_APPRENTISSAGE = 1e-3
MLP_TAILLE_LOT = 128
MLP_EPOQUES_MAX = 300

# Arrêt précoce sur la log-loss de validation 2022 (sélection autorisée sur 2022).
MLP_PATIENCE = 25
MLP_REDUCTION_LR_FACTEUR = 0.5
MLP_REDUCTION_LR_PATIENCE = 10
MLP_LR_MIN = 1e-5

# Pas de pondération des classes (55/45, peu déséquilibré) : elle dégraderait la
# calibration, or la sortie est une probabilité multipliée par le potentiel.
MLP_POIDS_CLASSES = None

# Plusieurs graines : la variance d'initialisation d'un MLP sur si peu de données
# n'est pas négligeable. On rapporte la dispersion et on moyenne les probabilités.
MLP_GRAINES = (0, 1, 2, 3, 4)
MLP_VERBOSE = 0

# --- LightGBM ----------------------------------------------------------------

# Mêmes 90 features que le MLP, sans normalisation (inutile pour des arbres).
# Arbres petits et feuilles peuplées : avec n_eff ≈ 2 sites, une feuille de moins
# de ~100 nuits-site décrirait un seul épisode météo (8 sites × une douzaine de
# jours), pas une relation générale.
LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],  # la première pilote l'arrêt précoce
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    # Features très corrélées entre elles (moyenne, min, max d'une même variable) :
    # tirer 80 % des colonnes par arbre évite qu'une seule famille monopolise tout.
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
# 0 = nombre de fils OpenMP par défaut. deterministic ne garantit l'identité des
# résultats qu'à nombre de fils constant, donc sur une même machine.
LGBM_NUM_THREADS = 0

# Taux d'apprentissage faible et beaucoup d'arbres, arrêtés par la log-loss de 2022.
LGBM_ARBRES_MAX = 3000
LGBM_ARRET_PRECOCE = 100

# Le sous-échantillonnage (bagging, feature_fraction) rend l'entraînement aléatoire :
# mêmes graines que le MLP, même moyenne des probabilités.
LGBM_GRAINES = (0, 1, 2, 3, 4)

# --- Mode --essai : vérification du pipeline, pas un résultat ----------------

ESSAI_EPOQUES = 3
ESSAI_LGBM_ARBRES = 20
ESSAI_BOOTSTRAP_N = 50

# --- Seuil de décision -------------------------------------------------------

# Choisi sur la validation 2022, jamais sur le test. Kappa plutôt qu'accuracy :
# l'accuracy récompense le fait de prédire la classe majoritaire.
CRITERE_SEUIL = "kappa"  # "kappa" ou "accuracy"
GRILLE_SEUIL = (0.05, 0.95, 0.01)  # début, fin incluse, pas

# --- Bootstrap par blocs de nuits --------------------------------------------

# Blocs de nuits consécutives, les huit sites d'une même date tirés ensemble :
# respecte l'autocorrélation temporelle (3 à 5 jours) et la corrélation inter-sites.
BOOTSTRAP_N = 1000
BOOTSTRAP_LONGUEUR_BLOC = 7  # nuits
BOOTSTRAP_NIVEAU = 0.95
BOOTSTRAP_GRAINE = 12345
