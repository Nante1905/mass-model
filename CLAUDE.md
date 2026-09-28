# Contexte projet — modélisation de l'exploitabilité du ciel nocturne (mémoire M2)

## Cadre

Mémoire M2 Big Data / ML / DL. Système intelligent de gestion pour un club d'astronomie à Antananarivo. Ce projet couvre la modélisation : prédire, à partir de la météo, si une nuit sera exploitable pour l'observation. Le dataset a été constitué dans un projet séparé (`extraction`), dont ce projet ne reçoit que les fichiers finaux.

Attente de rédaction : raisonnement techniquement rigoureux, défendable devant un jury, qui reconnaît ouvertement ses faiblesses plutôt que de construire des justifications a posteriori. Les limites se documentent, elles ne se masquent pas. En prose propre, sans listes à puces dans les livrables.

## Décisions héritées de l'extraction — ne pas rouvrir

**Architecture à deux étages.** Un étage déterministe (éphémérides, Lune, objets visibles, essaims) produit un indice de potentiel astronomique par règle explicite ; il n'est jamais appris. Un étage stochastique, le seul qui relève du ML et l'objet de ce projet, prédit la probabilité que le ciel soit exploitable. La note finale présentée au club est `potentiel_astronomique × P(ciel exploitable | météo)`.

**Trois préfixes, trois rôles, jamais mélangés.** `met_` : features météo (réanalyse ERA5 via Open-Meteo). `astro_` : étage déterministe. `sat_` : grandeurs satellitaires dont dérive le label. Aucune colonne `astro_` ni `sat_` n'entre dans le vecteur d'entrée du modèle : ce serait de la circularité (le modèle réapprendrait la formule du label) ou de la fuite. Tout pipeline de modélisation doit porter un assert qui le vérifie.

**Source du label : masque nuageux Meteosat CLM, service océan Indien (IODC).** Observation satellitaire indépendante des features. Fraction de pixels nuageux dans une fenêtre 5×5 pixels (~17 km) autour du site, moyennée sur les créneaux horaires de la nuit astronomique. Classes : exploitable si fraction moyenne < 30 %, inexploitable au-delà de 70 %, partiel entre les deux. Une nuit n'est labellisée que si au moins 4 heures et 50 % de la nuit astronomique sont couvertes. Le label ERA5 (même règle appliquée à la nébulosité de la réanalyse) existe dans `labels_nuits.parquet` pour comparaison uniquement : avec des features ERA5 il est circulaire, il ne doit jamais servir de cible.

**Fenêtre d'entraînement : février 2017 – décembre 2024**, bornée par l'homogénéité de la série Meteosat IODC.

**Huit sites** en plan factoriel pour décorréler altitude, latitude et position côte/intérieur :

| id | nom | lat | lon | alt (m) | régime |
|---|---|---|---|---|---|
| tana | Antananarivo | −18.88 | 47.51 | 1280 | hautes_terres |
| antsirabe | Antsirabe | −19.87 | 47.03 | 1500 | hautes_terres |
| fianar | Fianarantsoa | −21.45 | 47.09 | 1200 | hautes_terres |
| toamasina | Toamasina | −18.15 | 49.40 | 5 | cote_est |
| diego | Antsiranana | −12.28 | 49.29 | 30 | nord |
| majunga | Mahajanga | −15.72 | 46.32 | 20 | nord_ouest |
| ihosy | Ihosy | −22.40 | 46.12 | 750 | interieur_sud |
| tulear | Toliara | −23.36 | 43.67 | 10 | sud_ouest |

Le couple Antananarivo–Toamasina est le contrôle principal (même latitude, 1275 m de dénivelé). Le club est centré sur Tana ; les autres sites servent à apprendre la relation météo-nébulosité sur un échantillon plus riche et à tester sur Tana jamais vu.

## Données d'entrée

Trois fichiers parquet et un CSV, copiés depuis `extraction/data/final/`.

`features_horaires.parquet` — 237 367 lignes, une par (site, heure de nuit), 29 colonnes. Identifiants et temps : `site_id`, `instant_utc`, `nuit` (date de la nuit, attribuée par décalage de −12 h en heure locale Indian/Antananarivo, dtype object à convertir), `annee`, `heure_locale`, `tmp_doy_sin`, `tmp_doy_cos`. Features météo : `met_cloud_cover`, `met_cloud_cover_low`, `met_cloud_cover_mid`, `met_cloud_cover_high`, `met_temperature_2m`, `met_relative_humidity_2m`, `met_dew_point_2m`, `met_vapour_pressure_deficit`, `met_precipitation`, `met_wind_speed_10m`, `met_wind_gusts_10m`, `met_surface_pressure`, `met_weather_code`, `met_aod`, `met_poussiere`. Site : `altitude_site`, `latitude_site`. Cible, recopiée sur chaque heure de la nuit : `label` (catégoriel ordonné exploitable / partiel / inexploitable), `label_binaire` (1 = exploitable), `label_source`. Découpages : `split` et `split_spatial`.

`labels_nuits.parquet` — 29 232 lignes, une par (site, nuit). Label satellite (`label_sat`, `label_sat_binaire`, `sat_frac_nuage_moy/min/std`, `sat_heures_comptees`, `sat_plateforme`) et label ERA5 (`label_era5`, `label_era5_binaire`, `met_cloud_cover_moy/min/std`) côte à côte. Sert à la baseline ERA5 et à l'analyse d'erreurs ; ses colonnes `sat_` ne sont jamais des features.

`potentiel_nuits.parquet` — 29 232 lignes, indice déterministe `astro_indice_potentiel` et détail (Lune, objets, planètes, essaims), 2015–2024. N'intervient qu'à la fin, pour composer la note finale.

`correlation_sites.csv` — matrice de corrélation inter-sites de la nébulosité nocturne ; nombre effectif de sites indépendants n_eff = 2,13.

## Découpages de validation

Le découpage aléatoire est disqualifiant : deux heures consécutives sont quasi identiques, la météo est autocorrélée sur trois à cinq jours, et les huit sites partagent la même circulation. Deux découpages, tous deux rapportés.

Temporel (`split`) : entraînement 2017–2021 (1 795 nuits), validation 2022 (365 nuits), test 2023–2024 (731 nuits). Spatial (`split_spatial`) : Antananarivo en test (29 679 heures), les sept autres sites en entraînement.

Toute sélection d'hyperparamètres, de seuil de décision ou de calibration se fait sur la validation 2022, jamais sur le test. Le test n'est touché qu'une fois, à la fin. En validation croisée à l'intérieur de l'entraînement, découper par années entières (leave-one-year-out), jamais par lignes.

L'unité d'évaluation est la nuit, pas l'heure : le label est constant sur la nuit, et compter chaque heure gonfle artificiellement l'effectif et pondère les nuits longues (hiver austral).

## Ce que vaut le label — plafond de performance

Contrôle croisé avec VIIRS (Suomi-NPP, autre instrument, autre algorithme) sur 69 nuits, 1 318 paires site-instant. Avec la définition « nuageux franc seulement » côté VIIRS : biais −3,3 points, accord binaire 0,881, kappa 0,742 sur la fenêtre locale ; 0,923 et 0,832 à ~30 km. Le label signifie donc « aucun nuage franchement détecté », et non « ciel parfaitement clair » : Meteosat ne compte pas les nuages ténus ou incertains.

Conséquence à écrire dans le mémoire : un accord d'environ 0,88 entre deux satellites sur le même instant borne ce qu'un modèle peut atteindre contre le ciel réel. Un modèle qui dépasserait nettement ce niveau contre le label apprend les biais du label, pas le ciel.

Sur les Hautes Terres, les deux satellites concordent et ERA5 diverge : c'est la prémisse du projet (la réanalyse à ~25 km ne résout pas le relief), confirmée par les données.

## Baselines de référence — à battre nettement

Distribution des classes par nuit : exploitable 55,3 %, partiel 21,1 %, inexploitable 23,5 %. La classe majoritaire donne donc 0,553 d'accuracy en trois classes comme en binaire.

Seuil ERA5 : nébulosité ERA5 moyenne de la nuit < 30 % → exploitable. Contre le label satellite : accord binaire 0,659, kappa 0,350, biais −22,2 points (ERA5 voit plus de nuages que Meteosat).

Troisième baseline à construire : climatologie site × mois (fréquence d'exploitabilité apprise sur l'entraînement).

Si le modèle ne bat pas nettement ces trois références, c'est l'information la plus utile du mémoire et elle doit être écrite comme telle.

## Plan de modélisation proposé

Cible principale binaire (`label_binaire`), la plus utile au club et la plus robuste au bruit de frontière du label ; le trois classes en variante ordonnée.

Agrégation par nuit des features horaires (moyenne, min, max, écart-type, éventuellement valeurs en début, milieu et fin de nuit), pour les modèles tabulaires. Ordre des modèles : baselines, régression logistique régularisée (référence interprétable), gradient boosting (LightGBM, attendu le meilleur), MLP sur les mêmes agrégats, puis un modèle séquentiel (GRU ou CNN 1D) sur la séquence horaire de la nuit, qui est la seule justification honnête d'une architecture profonde ici.

Métriques : log-loss et Brier (la sortie est une probabilité multipliée par l'indice de potentiel, la calibration compte), ROC-AUC, accuracy et kappa au seuil choisi sur 2022, avec intervalles par bootstrap par blocs de nuits. Calibration (isotonique ou Platt) ajustée sur 2022. Résultats rapportés par site et par saison, en particulier Tana.

Attendu réaliste : avec n_eff ≈ 2 sites et environ 2 900 nuits, un gradient boosting battra probablement le réseau de neurones. Le constat est inconfortable pour un mémoire estampillé deep learning, mais l'écrire avec la comparaison chiffrée est bien plus solide que de forcer une architecture profonde qui sous-performe.

## Pièges connus dans les données

`met_aod` et `met_poussiere` sont manquants sur 69,7 % des lignes : l'archive CAMS d'Open-Meteo ne commence qu'au 4 août 2022. Ils sont donc absents de tout l'entraînement et présents seulement en validation et test. Les exclure du modèle principal ; au mieux, une expérience annexe sur 2022–2024.

Incohérence de frontière : le `label` trois classes est produit par `pd.cut` (bornes fermées à droite, donc une fraction de exactement 30 % tombe en exploitable) alors que `label_binaire` teste `< 30`. 321 heures sont exploitables en trois classes et 0 en binaire. À harmoniser (recalculer depuis `sat_frac_nuage_moy` de `labels_nuits.parquet`) et à mentionner.

602 heures sans label (nuits sous le seuil de couverture ou jours satellites manquants : 2018-09-23 et 2019-07-25 au 28) : à exclure, pas à imputer.

Bascule Meteosat-8 (41,5°E) vers Meteosat-9 (45,5°E) en 2022, tracée dans `sat_plateforme`. Elle coïncide avec la frontière entraînement / validation-test : une chute de performance en test peut venir d'un changement du label et non du modèle. Comparer les fréquences de classes et le biais contre ERA5 avant et après la bascule.

Biais résiduel à Diego (−12,9 points contre VIIRS) : le label y est moins fiable, à signaler dans les résultats par site.

Les features sont de la réanalyse, pas de la prévision. Un modèle entraîné sur ERA5 et utilisé en production sur des prévisions à J-1 subira un décalage de distribution. C'est une question de cadrage encore ouverte : soit on présente le modèle comme une borne supérieure (météo parfaitement connue), soit on reconstitue des features de prévision via l'archive Open-Meteo. La première option est acceptable si elle est dite explicitement.

## Environnement

Ce projet n'a pas besoin d'ecCodes ni de satpy : un environnement pip ordinaire suffit (pandas, pyarrow, numpy, scikit-learn, lightgbm, matplotlib, et torch pour les modèles séquentiels). Si l'on réutilise micromamba, installer depuis conda-forge avec `nodefaults` (un ancien `.condarc` pointe sur le canal commercial Anaconda). Ne pas modifier `~/.condarc` ni `~/.conda`.

## Préférences de travail

Ne pas lancer d'entraînement long ni de tâche de fond sans accord explicite : l'utilisateur veut garder la main sur l'exécution. Quand il demande d'expliquer, expliquer sans lancer. Réponses en français. Tous les seuils et hyperparamètres regroupés dans un `config.py`, parce que ce sont des choix à justifier en soutenance. Les résultats chiffrés destinés au mémoire sont consignés dans un `notes.md`, en prose, avec tableaux numérotés.
