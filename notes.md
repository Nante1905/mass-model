# Notes de résultats : prédire si une nuit sera exploitable

Ce document rassemble les résultats chiffrés destinés au mémoire pour les deux premiers modèles de l'étage stochastique, un réseau de neurones (MLP) et un gradient boosting (LightGBM), dans les deux découpages de validation, temporel et spatial. Les résultats de validation portent sur l'année 2022. Le test spatial, c'est-à-dire Antananarivo jamais vue, a été évalué une fois pour les deux modèles ; ses résultats sont en section 5. Le test temporel (les huit sites en 2023–2024) n'a pas encore été ouvert.

## 1. Vocabulaire

Les résultats utilisent un vocabulaire technique qu'il vaut mieux fixer une fois pour toutes. Le tableau 1 décrit les notions générales, le tableau 2 les mesures de performance.

**Tableau 1 – Notions générales**

| Terme | Explication simple |
|---|---|
| Nuit-site | Une nuit à un site donné. La nuit du 3 mars 2022 à Tana et la même nuit à Toamasina font deux nuits-site. C'est l'unité de comptage de tous les résultats. |
| Label | La bonne réponse que le modèle doit retrouver. Ici, le satellite Meteosat dit si la nuit était exploitable (moins de 30 % de nuages en moyenne sur la nuit) ou non. |
| Meteosat-8, Meteosat-9 | Les deux satellites successifs qui fournissent le label. Meteosat-8 jusqu'en 2021, Meteosat-9 à partir de 2022. |
| ERA5 | La réanalyse météo européenne : une reconstitution de la météo passée, sur une grille d'environ 25 km. C'est la source des données d'entrée du modèle. |
| Feature | Une donnée d'entrée du modèle. Par exemple, la nébulosité moyenne de la nuit selon ERA5. Les modèles reçoivent 90 features par nuit. |
| Agrégat | Un résumé d'une variable horaire sur la nuit entière : moyenne, minimum, maximum, écart-type, ou valeur en début, milieu et fin de nuit. |
| Entraînement, validation, test | Les trois paquets de données. Le modèle apprend sur l'entraînement (2017–2021). La validation (2022) sert à régler le modèle : quand arrêter l'apprentissage, quel seuil de décision prendre. Le test sert à la note finale et n'est utilisé qu'une fois, sans rien modifier ensuite. |
| Découpage temporel | Le modèle apprend sur les années passées et il est jugé sur des années qu'il n'a pas vues. Les huit sites sont présents partout. |
| Découpage spatial | Antananarivo est retirée de l'entraînement et de la validation. Le modèle apprend sur les sept autres sites, puis il est jugé sur Tana, un site qu'il n'a jamais vu. |
| Généralisation | La capacité d'un modèle à bien fonctionner sur des données qu'il n'a jamais vues : un site nouveau, des années nouvelles. |
| Baseline (référence) | Une méthode simple qu'un modèle doit battre pour être utile. Trois baselines sont utilisées ici (classe majoritaire, climatologie, seuil ERA5). |
| Classe majoritaire | La baseline la plus naïve : elle répond toujours « exploitable », parce que c'est le cas le plus fréquent. |
| Climatologie | Une baseline qui répond avec la fréquence habituelle du site et du mois. Par exemple : « en juillet à Tana, 70 % des nuits étaient exploitables entre 2017 et 2021 ». Pour un site jamais vu, elle utilise la moyenne du mois sur les autres sites. |
| Seuil ERA5 | Une baseline qui applique directement la règle du label à ERA5 : nuit exploitable si la nébulosité ERA5 moyenne est sous 30 %. |
| MLP | Réseau de neurones simple (*multilayer perceptron*). Ici, 90 entrées, deux couches de 64 et 32 neurones, et une sortie qui donne une probabilité. |
| LightGBM | Une méthode de gradient boosting : elle construit des centaines de petits arbres de décision, chacun corrigeant les erreurs des précédents. |
| Graine | Le nombre qui fixe le hasard d'un entraînement (poids de départ, ordre des données). Chaque modèle est entraîné avec 5 graines, et on moyenne les 5 résultats. |
| Arrêt précoce | On arrête l'apprentissage dès que les résultats sur 2022 cessent de s'améliorer, pour éviter le surapprentissage. |
| Surapprentissage | Le modèle apprend par cœur les données d'entraînement au lieu d'apprendre une règle générale. Il réussit alors très bien sur l'entraînement et moins bien ailleurs. |
| Seuil de décision | Le modèle donne une probabilité, mais le club veut un oui ou un non. Si la probabilité dépasse le seuil, la réponse est « exploitable ». Le seuil est choisi sur 2022. |
| Calibration | Un modèle est bien calibré si ses probabilités sont justes : parmi les nuits annoncées à 70 %, environ 70 % doivent être réellement exploitables. |
| Intervalle de confiance (IC) | La fourchette dans laquelle se trouve probablement la vraie valeur d'un score. Ici, IC à 95 %. Une fourchette étroite veut dire un chiffre fiable. |
| Bootstrap par blocs | La méthode qui calcule les intervalles. On refait l'évaluation 1 000 fois sur des jeux de nuits tirés au hasard par blocs de 7 nuits consécutives, les sites d'une même date restant ensemble. Les blocs respectent le fait que la météo d'un jour ressemble à celle de la veille. |
| Différence appariée | L'écart de score entre deux modèles, calculé sur exactement les mêmes tirages. Si son intervalle ne contient pas zéro, l'écart est réel et pas dû au hasard ; on dit qu'il est significatif. |
| Importance (gain) | Pour LightGBM, la part de chaque feature dans l'amélioration apportée par les arbres. Elle montre sur quoi le modèle s'appuie, mais ne prouve pas une cause. |
| n_eff | Le nombre effectif de sites indépendants. Les huit sites ont une météo très liée ; en information, ils valent à peu près 2,13 sites indépendants. |

**Tableau 2 – Mesures de performance**

| Mesure | Ce qu'elle mesure | Meilleur si | Hasard | Parfait |
|---|---|---|---|---|
| Accuracy | La part des nuits bien classées. Simple, mais trompeuse quand une réponse domine : répondre toujours « exploitable » donne déjà environ 0,55. | plus haut | environ 0,5 à 0,56 | 1 |
| Kappa (de Cohen) | L'accord entre la prédiction et le label, une fois retiré l'accord obtenu par pur hasard. On le lit ainsi : 0 veut dire pas mieux que le hasard, 0,4 à 0,6 un accord modéré, 0,6 à 0,8 un accord substantiel, au-delà un accord presque parfait. C'est la mesure principale pour les décisions oui/non. | plus haut | 0 | 1 |
| AUC (aire sous la courbe ROC) | La capacité à ranger les nuits dans le bon ordre. C'est la probabilité qu'une nuit claire prise au hasard reçoive une probabilité plus haute qu'une nuit nuageuse prise au hasard. Elle ne dépend pas du seuil. | plus haut | 0,5 | 1 |
| Log-loss | La qualité des probabilités. Elle punit fortement un modèle très sûr de lui qui se trompe : annoncer 99 % pour une nuit finalement nuageuse coûte très cher. | plus bas | 0,69 | 0 |
| Brier | L'écart moyen au carré entre la probabilité annoncée et ce qui s'est passé (1 ou 0). Annoncer 80 % pour une nuit claire coûte (1 − 0,8)² = 0,04. | plus bas | 0,25 | 0 |
| Taux prédit | La part des nuits que le modèle déclare exploitables. À comparer au taux observé : un grand écart signale un modèle trop optimiste ou trop pessimiste. | proche du taux observé | — | — |

La log-loss et le Brier sont particulièrement importants ici. La probabilité produite par le modèle est multipliée par l'indice de potentiel astronomique pour donner la note finale du club : une probabilité fausse fausse donc la note.

## 2. Protocole

Les deux modèles reçoivent exactement les mêmes entrées. Ce sont 90 features calculées sur chaque nuit à partir des données horaires ERA5 : 12 variables météo résumées de 7 façons (moyenne, minimum, maximum, écart-type, début, milieu et fin de nuit), deux résumés du code météo, et quatre informations de contexte (jour de l'année, altitude et latitude du site). Aucune donnée satellite ni astronomique n'entre dans le modèle ; un contrôle automatique le vérifie à chaque lancement. Les variables d'aérosols sont écartées, car elles n'existent pas avant août 2022.

La cible est binaire : exploitable ou non. Chaque modèle est entraîné cinq fois avec cinq graines différentes, et la probabilité finale est la moyenne des cinq. L'arrêt précoce et le choix du seuil de décision (celui qui maximise le kappa) se font sur la validation 2022. Le seuil ainsi choisi est ensuite figé et appliqué tel quel au test. Les deux modèles et les trois baselines sont évalués par le même code, sur les mêmes nuits. Les intervalles viennent de 1 000 tirages de bootstrap par blocs de 7 nuits. Les entraînements sont reproductibles : relancés pour évaluer le test, ils redonnent exactement les mêmes résultats de validation. Le tableau 3 résume les réglages.

**Tableau 3 – Réglages des deux modèles**

| | MLP | LightGBM |
|---|---|---|
| Structure | 90 entrées → 64 → 32 → 1 neurone | arbres de 15 feuilles au plus, au moins 100 nuits-site par feuille |
| Protection contre le surapprentissage | dropout 0,3, pénalité L2 1e-4 | 80 % des features et 80 % des nuits tirées au hasard pour chaque arbre, pénalité L2 1,0 |
| Vitesse d'apprentissage | 0,001, divisée par 2 en cas de stagnation | 0,03 |
| Arrêt précoce | après 25 époques sans progrès sur 2022 | après 100 arbres sans progrès sur 2022 |
| Graines | 5 | 5 |

Ces réglages ont été fixés à des valeurs raisonnables, sans recherche systématique. Il est possible qu'un réglage plus fin améliore légèrement l'un ou l'autre modèle.

## 3. Découpage temporel, validation : apprendre sur 2017–2021, juger sur 2022

La validation contient 2 920 nuits-site : 365 nuits pour chacun des huit sites. En 2022, 56,5 % des nuits ont été exploitables, contre 55,0 % sur la période d'entraînement.

**Tableau 4 – Résultats en découpage temporel (validation 2022, huit sites, IC 95 % entre crochets)**

| Modèle | Accuracy | Kappa | AUC | Log-loss | Brier | Taux prédit |
|---|---|---|---|---|---|---|
| Classe majoritaire | 0,565 | 0,000 | 0,500 | 0,685 | 0,246 | 1,000 |
| Seuil ERA5 | 0,647 [0,610 ; 0,687] | 0,335 [0,288 ; 0,378] | — | — | — | 0,261 |
| Climatologie | 0,780 [0,747 ; 0,811] | 0,550 [0,487 ; 0,608] | 0,827 | 0,509 | 0,165 | 0,586 |
| MLP | 0,847 [0,829 ; 0,865] | 0,690 [0,654 ; 0,723] | 0,928 | 0,337 | 0,106 | 0,556 |
| LightGBM | 0,852 [0,834 ; 0,869] | 0,699 [0,661 ; 0,731] | 0,930 | 0,332 | 0,105 | 0,562 |

Les deux modèles battent nettement les trois baselines, sur toutes les mesures. Le tableau 5 montre que ces écarts sont réels : aucun intervalle de différence ne contient zéro. Face à la meilleure baseline, la climatologie, les modèles gagnent environ 7 points d'accuracy et 0,14 de kappa. Leurs probabilités sont bien meilleures aussi, avec une log-loss qui baisse d'environ 0,17.

**Tableau 5 – Gain des modèles sur les baselines en validation (différence appariée, IC 95 %)**

| Comparaison | Découpage | Accuracy | Kappa | Log-loss |
|---|---|---|---|---|
| MLP − climatologie | temporel | [+0,041 ; +0,098] | [+0,089 ; +0,199] | [−0,209 ; −0,137] |
| LightGBM − climatologie | temporel | [+0,045 ; +0,101] | [+0,097 ; +0,206] | [−0,214 ; −0,141] |
| MLP − seuil ERA5 | temporel | [+0,157 ; +0,242] | [+0,308 ; +0,402] | — |
| LightGBM − seuil ERA5 | temporel | [+0,160 ; +0,247] | [+0,314 ; +0,409] | — |
| MLP − climatologie | spatial | [+0,047 ; +0,096] | [+0,091 ; +0,188] | [−0,208 ; −0,137] |
| LightGBM − climatologie | spatial | [+0,051 ; +0,101] | [+0,105 ; +0,200] | [−0,212 ; −0,137] |
| MLP − seuil ERA5 | spatial | [+0,134 ; +0,205] | [+0,252 ; +0,342] | — |
| LightGBM − seuil ERA5 | spatial | [+0,138 ; +0,209] | [+0,267 ; +0,353] | — |

Pour la log-loss, une différence négative veut dire que le modèle fait mieux que la baseline.

Le résultat le plus parlant concerne le seuil ERA5. Appliquer directement la règle du label à ERA5 fait moins bien que la climatologie, qui ne regarde pourtant ni la météo du jour ni ERA5 (kappa 0,335 contre 0,550). La raison se lit dans le taux prédit : ERA5 ne déclare exploitables que 26 % des nuits, alors que le satellite en compte 56,5 %. La réanalyse voit beaucoup trop de nuages, ce qui correspond au biais d'environ −22 points mesuré lors de la constitution du dataset. Or les deux modèles, qui n'utilisent que des données ERA5, atteignent un kappa proche de 0,70. L'information utile est donc bien présente dans ERA5, mais déformée, et les modèles apprennent à corriger cette déformation. C'est la justification centrale de l'étage appris.

## 4. Découpage spatial, validation : apprendre sans Antananarivo

Dans ce découpage, Antananarivo est retirée de l'entraînement et de la validation. La validation spatiale contient donc les sept autres sites en 2022, soit 2 555 nuits-site. Tana n'y figure volontairement pas : elle constitue le test spatial, présenté en section 5. Cette section répond à une question préalable : retirer Tana de l'apprentissage abîme-t-il le modèle ?

**Tableau 6 – Résultats en découpage spatial (validation 2022, sept sites sans Tana, IC 95 % entre crochets)**

| Modèle | Accuracy | Kappa | AUC | Log-loss | Brier | Taux prédit |
|---|---|---|---|---|---|---|
| Classe majoritaire | 0,562 | 0,000 | 0,500 | 0,685 | 0,246 | 1,000 |
| Seuil ERA5 | 0,673 [0,642 ; 0,706] | 0,379 [0,336 ; 0,421] | — | — | — | 0,290 |
| Climatologie | 0,773 [0,745 ; 0,800] | 0,541 [0,487 ; 0,588] | 0,823 | 0,515 | 0,168 | 0,550 |
| MLP | 0,843 [0,823 ; 0,863] | 0,677 [0,633 ; 0,719] | 0,924 | 0,343 | 0,108 | 0,612 |
| LightGBM | 0,847 [0,829 ; 0,866] | 0,689 [0,650 ; 0,725] | 0,926 | 0,341 | 0,109 | 0,575 |

Le taux observé est de 0,562. Les résultats ressemblent beaucoup à ceux du découpage temporel. Cette ressemblance ne suffit pourtant pas à conclure, car les deux validations ne contiennent pas les mêmes sites. Le tableau 7 fait donc la comparaison juste : les deux versions de chaque modèle sont jugées sur exactement les mêmes nuits, c'est-à-dire les sept sites hors Tana en 2022.

**Tableau 7 – Même modèle, avec ou sans Tana à l'entraînement (mêmes 2 555 nuits-site)**

| Modèle | Entraîné avec Tana (temporel) | Entraîné sans Tana (spatial) | Différence sans − avec (IC 95 %) | Corrélation des probabilités |
|---|---|---|---|---|
| MLP, kappa | 0,679 | 0,677 | [−0,023 ; +0,017] | 0,997 |
| MLP, log-loss | 0,343 | 0,343 | [−0,003 ; +0,003] | |
| LightGBM, kappa | 0,690 | 0,689 | [−0,011 ; +0,009] | 0,999 |
| LightGBM, log-loss | 0,340 | 0,341 | [−0,001 ; +0,004] | |

Retirer Tana de l'apprentissage ne change pratiquement rien. Les probabilités des deux versions sont corrélées à 0,997 pour le MLP et 0,999 pour LightGBM, et toutes les différences sont minuscules, avec des intervalles qui contiennent zéro. Ce résultat est cohérent avec le nombre effectif de sites indépendants, environ 2 : la météo de Tana ressemble tellement à celle d'Antsirabe et de Fianarantsoa que Tana n'apporte presque aucune information nouvelle à l'apprentissage.

## 5. Test spatial : Antananarivo jamais vue

C'est le test le plus important pour le club, qui observe depuis Tana. Les modèles ont appris sur les sept autres sites (2017–2021) et ont été réglés sur ces mêmes sites en 2022 ; ils n'ont jamais vu une seule nuit de Tana. Ils sont jugés sur toutes les nuits labellisées de Tana, de février 2017 à décembre 2024, soit 2 884 nuits. Ils sont aussi jugés sur 2023–2024 seulement (731 nuits) : c'est le test le plus strict, avec à la fois un site et des années jamais vus. Pour Tana, la climatologie ne peut pas utiliser la fréquence propre du site, qu'elle ne connaît pas : elle utilise la fréquence moyenne du mois sur les sept autres sites.

**Tableau 8 – Résultats sur le test spatial, Antananarivo (IC 95 % entre crochets)**

| Période | Modèle | Accuracy | Kappa | AUC | Log-loss | Brier | Taux prédit |
|---|---|---|---|---|---|---|---|
| 2017–2024 (2 884 nuits, taux observé 0,537) | Classe majoritaire | 0,537 | 0,000 | 0,500 | 0,691 | 0,249 | 1,000 |
| | Seuil ERA5 | 0,538 [0,508 ; 0,571] | 0,129 [0,108 ; 0,151] | — | — | — | 0,095 |
| | Climatologie | 0,760 [0,737 ; 0,783] | 0,514 [0,464 ; 0,560] | 0,812 | 0,553 | 0,184 | 0,591 |
| | MLP | 0,847 [0,833 ; 0,862] | 0,693 [0,662 ; 0,723] | 0,924 | 0,349 | 0,109 | 0,550 |
| | LightGBM | 0,831 [0,816 ; 0,846] | 0,664 [0,634 ; 0,694] | 0,927 | 0,354 | 0,113 | 0,470 |
| 2023–2024 (731 nuits, taux observé 0,539) | Classe majoritaire | 0,539 | 0,000 | 0,500 | 0,691 | 0,249 | 1,000 |
| | Seuil ERA5 | 0,534 [0,475 ; 0,598] | 0,124 [0,082 ; 0,173] | — | — | — | 0,086 |
| | Climatologie | 0,770 [0,720 ; 0,820] | 0,534 [0,428 ; 0,634] | 0,830 | 0,540 | 0,178 | 0,585 |
| | MLP | 0,845 [0,813 ; 0,877] | 0,688 [0,621 ; 0,748] | 0,932 | 0,330 | 0,102 | 0,568 |
| | LightGBM | 0,850 [0,821 ; 0,879] | 0,699 [0,641 ; 0,757] | 0,932 | 0,337 | 0,106 | 0,506 |

Le résultat principal est rassurant. Sur un site qu'ils n'ont jamais vu, les deux modèles gardent le niveau obtenu en validation : environ 85 % d'accuracy, un kappa entre 0,66 et 0,70, une AUC de 0,92 à 0,93. On craignait que les chiffres de validation soient trop optimistes, puisque 2022 a servi aux réglages. La perte sur le test est en réalité très faible. Le modèle a donc appris une relation entre la météo ERA5 et le ciel qui se transpose à un nouveau site, et non des particularités des sites d'entraînement.

Le contraste avec ERA5 seul est encore plus fort à Tana qu'ailleurs. Appliqué directement, ERA5 ne déclare exploitables que 9,5 % des nuits de Tana, alors que le satellite en compte 53,7 %. Son accuracy (0,538) est au niveau de la réponse constante « toujours exploitable », et son kappa n'est que de 0,13. Sur ce site des Hautes Terres, la réanalyse brute est presque inutilisable, et les modèles, qui partent pourtant des mêmes données, la rendent exploitable. Le tableau 9 confirme que les gains sur les baselines sont tous significatifs.

**Tableau 9 – Gain des modèles sur les baselines, test spatial Tana (différence appariée, IC 95 %)**

| Comparaison | Période | Accuracy | Kappa | Log-loss |
|---|---|---|---|---|
| MLP − climatologie | 2017–2024 | [+0,066 ; +0,109] | [+0,137 ; +0,221] | [−0,226 ; −0,183] |
| LightGBM − climatologie | 2017–2024 | [+0,050 ; +0,094] | [+0,107 ; +0,197] | [−0,222 ; −0,177] |
| MLP − seuil ERA5 | 2017–2024 | [+0,270 ; +0,343] | [+0,528 ; +0,597] | — |
| LightGBM − seuil ERA5 | 2017–2024 | [+0,260 ; +0,325] | [+0,499 ; +0,569] | — |
| MLP − climatologie | 2023–2024 | [+0,034 ; +0,121] | [+0,074 ; +0,250] | [−0,254 ; −0,166] |
| LightGBM − climatologie | 2023–2024 | [+0,038 ; +0,128] | [+0,082 ; +0,261] | [−0,252 ; −0,158] |

Le coût de ne jamais avoir vu Tana peut être mesuré directement sur l'année 2022. Le modèle du découpage temporel a appris avec Tana et le modèle du découpage spatial sans elle ; on les compare sur les mêmes 365 nuits de Tana (tableau 10).

**Tableau 10 – Tana en 2022 : modèle ayant appris avec Tana ou sans Tana**

| Modèle | Kappa avec Tana | Kappa sans Tana | Log-loss avec Tana | Log-loss sans Tana | Corrélation des probabilités |
|---|---|---|---|---|---|
| MLP | 0,765 | 0,749 | 0,294 | 0,305 | 0,995 |
| LightGBM | 0,758 | 0,754 | 0,279 | 0,303 | 0,992 |

La perte existe, mais elle est faible : environ 0,01 à 0,02 de kappa et 0,01 à 0,02 de log-loss. Les deux versions donnent des probabilités presque identiques. Pour le club, cela veut dire que le modèle ne dépend pas de la présence de Tana dans les données, et qu'il devrait fonctionner de façon comparable sur un autre site des Hautes Terres. Ce n'est vrai qu'à condition que ce site ressemble à ceux de l'entraînement.

**Tableau 11 – Test spatial Tana, année par année**

| Année | Nuits | Taux observé | Kappa MLP | Kappa LightGBM | AUC MLP | AUC LightGBM | Kappa climatologie | Kappa seuil ERA5 |
|---|---|---|---|---|---|---|---|---|
| 2017 | 334 | 0,506 | 0,641 | 0,582 | 0,892 | 0,895 | 0,393 | 0,152 |
| 2018 | 363 | 0,532 | 0,696 | 0,613 | 0,919 | 0,921 | 0,504 | 0,123 |
| 2019 | 360 | 0,458 | 0,692 | 0,677 | 0,928 | 0,931 | 0,561 | 0,140 |
| 2020 | 366 | 0,579 | 0,690 | 0,650 | 0,927 | 0,929 | 0,506 | 0,178 |
| 2021 | 365 | 0,556 | 0,678 | 0,620 | 0,913 | 0,922 | 0,436 | 0,134 |
| 2022 | 365 | 0,584 | 0,749 | 0,754 | 0,942 | 0,947 | 0,633 | 0,059 |
| 2023 | 365 | 0,518 | 0,660 | 0,678 | 0,926 | 0,924 | 0,487 | 0,091 |
| 2024 | 366 | 0,560 | 0,714 | 0,717 | 0,935 | 0,937 | 0,582 | 0,150 |

La performance est stable d'une année à l'autre, avec une AUC entre 0,89 et 0,95 : aucune année ne s'effondre. Un point mérite l'attention. Les années 2017 à 2021 sont celles où le modèle a vu les sept autres sites pendant l'apprentissage, aux mêmes dates. Si le modèle profitait de la météo commune à tous les sites un même jour, Tana serait mieux prédite sur ces années. C'est l'inverse qui se produit : le kappa y est un peu plus bas. Il n'y a donc pas de signe que le modèle « triche » en reconnaissant des journées déjà vues.

**Tableau 12 – Test spatial Tana selon la période du satellite**

| Période | Satellite du label | Nuits | Taux observé | Nébulosité ERA5 moyenne | Taux prédit MLP | Taux prédit LightGBM | Kappa MLP | Kappa LightGBM |
|---|---|---|---|---|---|---|---|---|
| 2017–2021 | Meteosat-8 | 1 788 | 0,527 | 64,2 % | 0,527 | 0,436 | 0,683 | 0,631 |
| 2022–2024 | Meteosat-9 | 1 096 | 0,554 | 71,5 % | 0,586 | 0,525 | 0,708 | 0,717 |

Le tableau 12 révèle un changement entre les deux périodes. À Tana, ERA5 voit en moyenne plus de nuages en 2022–2024 qu'en 2017–2021 (71,5 % contre 64,2 %), alors que le satellite compte davantage de nuits claires (55,4 % contre 52,7 %). La relation entre ERA5 et le label n'est donc pas tout à fait la même avant et après 2022. Ce changement coïncide avec le passage de Meteosat-8 à Meteosat-9. Mais il pourrait aussi venir d'une vraie variation du climat d'une année à l'autre, ou d'une évolution d'ERA5. Les données disponibles dans ce projet ne permettent pas de trancher. Il faudra comparer, nuit par nuit, le label et la plateforme satellite dans `labels_nuits.parquet`. Ce constat ne remet pas en cause les résultats, puisque les deux modèles restent bons sur les deux périodes, mais il doit être signalé.

**Tableau 13 – Test spatial Tana par saison (2017–2024)**

| Saison | Nuits | Taux observé | Kappa MLP | AUC MLP | Kappa LightGBM | AUC LightGBM | Kappa climatologie | Kappa seuil ERA5 | Accuracy seuil ERA5 |
|---|---|---|---|---|---|---|---|---|---|
| Humide (nov.–avr.) | 1 419 | 0,273 | 0,593 | 0,904 | 0,529 | 0,909 | 0,216 | 0,156 | 0,753 |
| Sèche (mai–oct.) | 1 465 | 0,793 | 0,525 | 0,870 | 0,514 | 0,874 | 0,000 | 0,056 | 0,330 |

À l'intérieur d'une saison, les modèles restent nettement utiles, avec une AUC de 0,87 à 0,91, alors que la climatologie ne sait presque plus rien dire : en saison sèche, elle répond toujours « exploitable », d'où son kappa nul. En saison sèche, ERA5 seul se trompe deux fois sur trois (accuracy 0,33). Il ne déclare exploitables que 14 % des nuits, alors que 79 % le sont. C'est l'illustration la plus nette du biais de la réanalyse sur les Hautes Terres.

## 6. MLP contre LightGBM

En validation, LightGBM obtient des chiffres très légèrement meilleurs que le MLP dans les deux découpages, mais sans écart significatif. Le test spatial nuance ce constat (tableau 14).

**Tableau 14 – Différence LightGBM − MLP (différence appariée, IC 95 %)**

| Données | Accuracy | Kappa | AUC | Log-loss | Brier |
|---|---|---|---|---|---|
| Validation temporelle (8 sites, 2022) | [−0,002 ; +0,011] | [−0,004 ; +0,022] | [−0,000 ; +0,006] | [−0,012 ; +0,004] | [−0,003 ; +0,002] |
| Validation spatiale (7 sites, 2022) | [−0,004 ; +0,012] | [−0,006 ; +0,030] | [−0,001 ; +0,006] | [−0,009 ; +0,007] | [−0,002 ; +0,003] |
| Test spatial Tana 2017–2024 | **[−0,028 ; −0,005]** | **[−0,051 ; −0,007]** | [−0,000 ; +0,005] | [−0,003 ; +0,013] | **[+0,000 ; +0,006]** |
| Test spatial Tana 2023–2024 | [−0,016 ; +0,026] | [−0,029 ; +0,053] | [−0,006 ; +0,006] | [−0,014 ; +0,029] | [−0,004 ; +0,011] |

Les écarts significatifs sont en gras. Pour la log-loss et le Brier, une valeur positive veut dire que LightGBM fait moins bien.

Sur Tana de 2017 à 2024, le MLP fait significativement mieux que LightGBM en accuracy (environ +1,6 point), en kappa (environ +0,03) et en Brier. En revanche, l'AUC est la même pour les deux modèles. Ils rangent donc les nuits de Tana dans le même ordre, du plus nuageux au plus clair ; la différence tient à la valeur des probabilités, pas au classement. Le taux prédit le montre : LightGBM ne déclare exploitables que 47 % des nuits de Tana, contre 53,7 % observés, alors que le MLP en déclare 55 %. LightGBM est trop pessimiste à Tana, et le tableau 15 montre que ce défaut est systématique.

**Tableau 15 – Calibration sur le test spatial Tana (2017–2024) : probabilité annoncée et fréquence réellement observée**

| Tranche de probabilité | MLP : nuits | MLP : annoncé | MLP : observé | LightGBM : nuits | LightGBM : annoncé | LightGBM : observé |
|---|---|---|---|---|---|---|
| 0 – 0,1 | 677 | 0,03 | 0,03 | 811 | 0,03 | 0,05 |
| 0,1 – 0,2 | 181 | 0,15 | 0,15 | 231 | 0,14 | 0,21 |
| 0,2 – 0,3 | 138 | 0,25 | 0,20 | 153 | 0,25 | 0,43 |
| 0,3 – 0,4 | 139 | 0,35 | 0,31 | 131 | 0,35 | 0,49 |
| 0,4 – 0,5 | 181 | 0,45 | 0,51 | 122 | 0,45 | 0,60 |
| 0,5 – 0,6 | 202 | 0,55 | 0,61 | 130 | 0,55 | 0,64 |
| 0,6 – 0,7 | 165 | 0,65 | 0,76 | 166 | 0,66 | 0,80 |
| 0,7 – 0,8 | 230 | 0,75 | 0,78 | 199 | 0,75 | 0,79 |
| 0,8 – 0,9 | 310 | 0,86 | 0,87 | 314 | 0,86 | 0,89 |
| 0,9 – 1 | 661 | 0,96 | 0,97 | 627 | 0,95 | 0,97 |

Pour LightGBM, la fréquence observée dépasse la probabilité annoncée dans presque toutes les tranches, avec un écart qui atteint 14 à 18 points entre 0,2 et 0,7. Quand LightGBM annonce 25 % à Tana, 43 % des nuits sont en réalité claires. Le MLP est plus proche de la diagonale, même s'il sous-estime lui aussi de 6 à 11 points entre 0,4 et 0,7. Une partie de ce défaut de LightGBM existait déjà en validation sur les autres sites (section 9), mais il est nettement amplifié sur Tana.

Une explication plausible, non vérifiée, tient à la façon dont chaque modèle traite un site nouveau. Un arbre de décision découpe l'altitude et la latitude en paliers : face à la combinaison inédite de Tana, il la range dans le palier d'un site connu et en reprend le comportement. Un réseau de neurones, au contraire, varie de façon continue et peut interpoler entre les sites. Quelle qu'en soit la cause, le constat pratique est clair : sur un site jamais vu, les probabilités du MLP sont plus fiables que celles de LightGBM.

Un point de méthode doit accompagner ce constat. La différence entre les deux modèles n'apparaît qu'au test ; en validation, ils sont équivalents. Si ce résultat sert à choisir le MLP, il faut l'écrire explicitement, car le test aura alors servi à une décision, et le chiffre du modèle retenu devient légèrement optimiste. De même, corriger la calibration de LightGBM en s'appuyant sur ces nuits de Tana serait une fuite : un recalibrage ne peut s'ajuster que sur la validation 2022.

Au-delà de cette nuance, les deux modèles restent très proches : ils prennent la même décision sur 91 à 96 % des nuits, et se trompent en grande partie sur les mêmes nuits (tableau 16).

**Tableau 16 – Ressemblance entre les deux modèles**

| Données | Corrélation des probabilités | Même décision | Erreurs du MLP | Erreurs de LightGBM | Erreurs communes |
|---|---|---|---|---|---|
| Validation temporelle | 0,982 | 95,9 % des nuits | 446 | 433 | 379 |
| Validation spatiale | 0,983 | 95,2 % des nuits | 401 | 390 | 334 |
| Test spatial Tana 2017–2024 | 0,978 | 91,5 % des nuits | 440 | 486 | 341 |

Deux méthodes de nature très différente, un réseau de neurones et des arbres de décision, butent sur les mêmes nuits dans 70 à 85 % des cas. La limite principale ne vient donc pas du choix du modèle, mais des données elles-mêmes. Pour ces nuits difficiles, soit ERA5 ne contient pas l'information nécessaire (un nuage local trop petit pour une grille de 25 km), soit le label est lui-même incertain.

Ce plafond était attendu. Le contrôle croisé entre Meteosat et un second satellite (VIIRS) donne un accord de 0,88 et un kappa de 0,74 au même instant. Aucun modèle ne peut s'accorder avec le label beaucoup mieux que deux satellites ne s'accordent entre eux. Avec une accuracy d'environ 0,85 et un kappa proche de 0,70, en validation comme sur Tana, les deux modèles sont déjà à environ trois points de ce plafond. La comparaison reste indicative, puisque l'accord entre satellites est mesuré heure par heure et non nuit par nuit.

Pour le mémoire, la conclusion est donc double. Sur les données qu'ils connaissent, les deux modèles sont équivalents : le MLP sur agrégats n'apporte pas plus que le gradient boosting en capacité à distinguer les nuits. Sur un site nouveau, le MLP produit des probabilités mieux calibrées, ce qui est un avantage réel pour la note finale, où la probabilité est multipliée par l'indice de potentiel. LightGBM garde des avantages pratiques : il est plus léger, plus rapide, ne dépend pas de TensorFlow et indique l'importance de chaque feature. Le modèle séquentiel, qui lirait la nuit heure par heure, reste la seule piste qui pourrait justifier une architecture plus profonde ; vu que les deux modèles butent sur les mêmes nuits, il n'est pas certain qu'il fasse mieux.

## 7. Résultats par site

Le tableau 17 donne le kappa de chaque site. Le kappa permet de comparer des sites où la proportion de nuits claires est très différente.

**Tableau 17 – Kappa par site (année 2022)**

| Site | Seuil ERA5 | Climatologie | MLP temporel | LightGBM temporel | MLP spatial | LightGBM spatial |
|---|---|---|---|---|---|---|
| Antananarivo | 0,059 | 0,633 | 0,765 | 0,758 | 0,749 (test) | 0,754 (test) |
| Antsirabe | 0,112 | 0,602 | 0,713 | 0,777 | 0,763 | 0,765 |
| Fianarantsoa | 0,065 | 0,431 | 0,540 | 0,534 | 0,529 | 0,546 |
| Ihosy | 0,177 | 0,547 | 0,773 | 0,748 | 0,745 | 0,752 |
| Toamasina | 0,300 | 0,000 | 0,490 | 0,530 | 0,496 | 0,532 |
| Antsiranana (Diego) | 0,329 | 0,405 | 0,555 | 0,574 | 0,587 | 0,562 |
| Mahajanga | 0,653 | 0,503 | 0,719 | 0,718 | 0,697 | 0,727 |
| Toliara | 0,583 | 0,000 | 0,656 | 0,628 | 0,641 | 0,611 |

Les colonnes ERA5 et climatologie sont celles du découpage temporel ; en spatial, elles diffèrent de moins de 0,02. Pour Antananarivo en découpage spatial, les valeurs sont celles de l'année 2022 dans le test spatial.

Ce tableau confirme la prémisse du projet. Sur les Hautes Terres et à l'intérieur (Antananarivo, Antsirabe, Fianarantsoa, Ihosy), ERA5 appliqué directement ne vaut presque rien, avec un kappa entre 0,06 et 0,18. La réanalyse à 25 km ne résout pas le relief. Les modèles y redressent fortement la situation : à Tana, le kappa passe de 0,06 avec ERA5 seul à environ 0,75. Sur la côte ouest (Mahajanga, Toliara), ERA5 fonctionne déjà assez bien, et les modèles n'ajoutent qu'un gain modeste.

Trois sites restent difficiles pour les deux modèles, ce qui renforce l'idée d'une limite dans les données. À Toamasina, le kappa reste autour de 0,50 malgré une accuracy de 82 %, parce que ce site pluvieux n'a que 22 % de nuits claires et que le kappa juge sévèrement la classe rare. À Fianarantsoa, l'AUC est la plus basse (0,84). Pour Antsiranana, le résultat est à prendre avec prudence, car le label y est moins fiable : son biais contre VIIRS est de −12,9 points. Le kappa nul de la climatologie à Toamasina et Toliara n'est pas une erreur : sur ces sites, la climatologie donne toujours la même réponse, ce qui donne un kappa de 0 par construction.

Les écarts entre modèles sur un même site, jusqu'à ±0,06 de kappa, ne doivent pas être interprétés. Avec 365 nuits par site, ils sont de l'ordre du bruit, et ils dépendent aussi du seuil de décision (section 9). Antsirabe l'illustre bien : le MLP y obtient 0,713 en temporel et 0,763 en spatial avec des probabilités presque identiques, parce que seul le seuil a changé.

## 8. Résultats par saison, validation

**Tableau 18 – Kappa et AUC par saison (validation 2022)**

| Modèle | Saison humide (nov.–avr.), kappa | Saison humide, AUC | Saison sèche (mai–oct.), kappa | Saison sèche, AUC |
|---|---|---|---|---|
| Seuil ERA5 (temporel) | 0,369 | — | 0,212 | — |
| Climatologie (temporel) | 0,392 | 0,712 | 0,441 | 0,792 |
| MLP temporel | 0,610 | 0,912 | 0,618 | 0,903 |
| LightGBM temporel | 0,616 | 0,912 | 0,628 | 0,910 |
| MLP spatial | 0,638 | 0,913 | 0,578 | 0,899 |
| LightGBM spatial | 0,626 | 0,913 | 0,617 | 0,905 |

La saison sèche est plus facile en apparence, parce que 76 % des nuits y sont claires contre 37 % en saison humide, mais le kappa est semblable dans les deux saisons. Une nuance importante apparaît : le kappa par saison (environ 0,62) est plus bas que le kappa global (environ 0,70). Une partie du score global vient donc simplement de la capacité à distinguer la saison sèche de la saison humide, ce que la climatologie sait déjà faire. La vraie valeur ajoutée de la météo du jour se mesure à l'intérieur d'une saison ; elle reste élevée, avec une AUC autour de 0,91 en validation et de 0,87 à 0,91 sur Tana (tableau 13).

## 9. Seuil de décision et calibration, validation

**Tableau 19 – Seuils de décision choisis sur 2022**

| Modèle | Temporel | Spatial |
|---|---|---|
| MLP | 0,63 | 0,49 |
| LightGBM | 0,61 | 0,57 |
| Climatologie | 0,46 | 0,48 |

Le seuil du MLP passe de 0,63 à 0,49 d'un découpage à l'autre, alors que ses probabilités sur les sept sites communs sont presque identiques (corrélation 0,997), et que l'accuracy et le kappa restent pratiquement les mêmes. Cela veut dire que le kappa varie très peu sur une large plage de seuils : son maximum est plat, et un petit changement dans les nuits de validation suffit à déplacer beaucoup le seuil retenu. Le seuil est donc un réglage fragile. Pour le club, il est plus solide de montrer la probabilité elle-même, qui est stable, que la décision oui/non, qui dépend de ce seuil.

**Tableau 20 – Calibration en validation temporelle (2022, huit sites) : probabilité annoncée et fréquence réellement observée**

| Tranche de probabilité | MLP : nuits | MLP : annoncé | MLP : observé | LightGBM : nuits | LightGBM : annoncé | LightGBM : observé |
|---|---|---|---|---|---|---|
| 0 – 0,1 | 603 | 0,03 | 0,02 | 691 | 0,03 | 0,03 |
| 0,1 – 0,2 | 148 | 0,15 | 0,12 | 175 | 0,14 | 0,14 |
| 0,2 – 0,3 | 136 | 0,24 | 0,17 | 117 | 0,25 | 0,36 |
| 0,3 – 0,4 | 101 | 0,35 | 0,33 | 94 | 0,36 | 0,44 |
| 0,4 – 0,5 | 116 | 0,45 | 0,47 | 93 | 0,45 | 0,46 |
| 0,5 – 0,6 | 142 | 0,55 | 0,49 | 97 | 0,55 | 0,45 |
| 0,6 – 0,7 | 183 | 0,65 | 0,61 | 136 | 0,65 | 0,59 |
| 0,7 – 0,8 | 222 | 0,75 | 0,69 | 189 | 0,75 | 0,71 |
| 0,8 – 0,9 | 301 | 0,85 | 0,83 | 324 | 0,86 | 0,81 |
| 0,9 – 1 | 968 | 0,96 | 0,96 | 1004 | 0,96 | 0,95 |

En validation, les deux modèles sont bien calibrés aux extrémités, là où se trouvent la majorité des nuits. Au milieu, LightGBM sous-estime déjà les nuits annoncées entre 20 et 40 %, dont environ 40 % sont en réalité claires, et ce défaut se retrouve en validation spatiale. C'est le même défaut, en plus fort, qui apparaît sur Tana (tableau 15). Le MLP surestime légèrement entre 20 et 30 % et entre 70 et 80 %. Ces tranches ne contiennent qu'une centaine de nuits chacune, soit une incertitude d'environ ±5 points. Un recalibrage ajusté sur 2022 pourrait corriger une partie de ces défauts.

## 10. Ce qui compte pour LightGBM

**Tableau 21 – Importance des features pour LightGBM (part du gain, moyenne des 5 graines)**

| Feature | Temporel | Spatial |
|---|---|---|
| Nébulosité totale minimale de la nuit | 29,5 % | 31,3 % |
| Nébulosité moyenne des nuages de niveau moyen | 10,1 % | 10,3 % |
| Nébulosité totale en début de nuit | 9,0 % | 8,7 % |
| Nébulosité maximale des nuages hauts | 6,0 % | 5,1 % |
| Jour de l'année (cosinus) | 2,9 % | 2,3 % |
| Nébulosité totale moyenne | 2,4 % | 2,5 % |
| Latitude du site | 0,9 % | 0,9 % |
| Altitude du site | 0,7 % | 0,8 % |

La nébulosité domine très largement, ce qui est logique puisque le label mesure des nuages ; température, humidité, vent et pression pèsent peu. Le minimum de la nébulosité compte beaucoup plus que sa moyenne. Autrement dit, le modèle regarde surtout si ERA5 prévoit, à un moment de la nuit, une trouée dans les nuages. C'est cohérent avec le biais d'ERA5, qui voit trop de nuages : quand même ERA5 descend bas à un moment, la nuit a de bonnes chances d'être claire pour le satellite. L'état du ciel en début de nuit compte aussi. L'altitude et la latitude pèsent moins de 1 % chacune, bien que le biais d'ERA5 dépende fortement du site ; la correction passe sans doute par les valeurs de nébulosité elles-mêmes. Les importances sont presque identiques dans les deux découpages, signe que le modèle s'appuie sur une relation stable. L'importance par gain doit toutefois être lue avec prudence : quand plusieurs features sont très liées (minimum, moyenne et début d'une même nébulosité), la répartition du gain entre elles est en partie arbitraire, et elle décrit le fonctionnement du modèle, pas une cause physique.

## 11. Stabilité et surapprentissage

**Tableau 22 – Dispersion entre les 5 graines (validation 2022)**

| Modèle | Log-loss (min – max) | Durée d'apprentissage retenue |
|---|---|---|
| MLP temporel | 0,338 – 0,340 | 31 à 42 époques |
| LightGBM temporel | 0,332 – 0,334 | 310 à 510 arbres |
| MLP spatial | 0,345 – 0,346 | 21 à 37 époques |
| LightGBM spatial | 0,342 – 0,343 | 315 à 387 arbres |

Les cinq graines donnent des résultats quasi identiques, donc le hasard de l'entraînement n'influence pas les conclusions. LightGBM colle davantage aux données d'entraînement que le MLP : sa log-loss d'entraînement descend à 0,26–0,29 alors qu'elle reste vers 0,33–0,34 en validation. Il mémorise donc une partie de l'entraînement, mais l'arrêt précoce empêche que cela dégrade la validation. Pour le MLP, l'écart entre entraînement et validation est faible, mais il se compare mal, car la perte d'entraînement de Keras inclut la pénalité L2 et le dropout. Dans les deux cas, l'essentiel de l'apprentissage se fait très vite : le MLP atteignait déjà 0,844 d'accuracy après 3 époques lors d'un essai. La relation entre ERA5 et le label est donc assez simple, et une régression logistique pourrait s'en approcher.

## 12. Limites

Plusieurs limites doivent accompagner ces chiffres.

Les résultats de validation 2022 sont optimistes par construction, puisque 2022 a servi à l'arrêt précoce et au choix du seuil. Le test spatial montre que cet optimisme est faible, puisque Tana jamais vue est prédite au même niveau. Mais le test temporel sur les huit sites en 2023–2024 reste à faire.

Le test spatial a été évalué pour les deux modèles avant la fin du travail de modélisation, et c'est lui seul qui les départage. Toute décision prise désormais en s'appuyant sur ses chiffres, comme choisir le MLP, recalibrer LightGBM ou modifier les features, doit être déclarée comme telle. Dans ce cas, le test spatial ne pourra plus être présenté comme une évaluation entièrement indépendante.

Tana n'est qu'un seul site, et le test spatial ne mesure que la généralisation à ce site. Il est encourageant parce que Tana ressemble aux sites d'entraînement des Hautes Terres. Un site au climat très différent de tous les sites d'entraînement pourrait être moins bien prédit.

Le satellite source du label change en 2022 (de Meteosat-8 à Meteosat-9), au moment exact de la frontière entre entraînement et validation. À Tana, la relation entre ERA5 et le label n'est pas la même avant et après 2022 (tableau 12), sans qu'on puisse encore dire si le changement de satellite en est la cause. La vérification nuit par nuit reste à faire avec `labels_nuits.parquet`, absent de ce projet.

Les entrées sont de la réanalyse, c'est-à-dire la météo passée reconstituée, et non de la prévision. Ces scores décrivent donc une borne supérieure, celle d'un modèle qui connaîtrait parfaitement la météo de la nuit. En service réel, avec des prévisions, la performance sera plus basse, dans une proportion encore inconnue.

Enfin, le label ne dit pas « ciel parfaitement clair » mais « aucun nuage franchement détecté par Meteosat ». Les nuages fins ou incertains ne sont pas comptés, et un modèle qui dépasserait nettement le niveau d'accord entre deux satellites apprendrait les défauts du label plutôt que le ciel réel. Les hyperparamètres n'ont pas été optimisés, et l'importance des features ne se lit pas comme une relation de cause à effet.

## 13. Bilan provisoire

Les deux modèles battent nettement les trois baselines, en validation comme au test : environ 85 % d'accuracy, un kappa proche de 0,70 et une AUC de 0,92 à 0,93, contre environ 77 % et un kappa de 0,51 à 0,55 pour la climatologie, et bien moins pour ERA5 appliqué directement. L'apport est le plus fort sur les Hautes Terres, où ERA5 seul échoue presque complètement : à Tana, il ne déclare exploitables que 9,5 % des nuits, contre 53,7 % en réalité.

Sur Antananarivo jamais vue, les modèles gardent le niveau de la validation. Ne pas avoir vu Tana ne coûte qu'environ 0,01 à 0,02 de kappa, et la performance est stable d'une année à l'autre. En validation, le MLP et LightGBM sont statistiquement équivalents et se trompent sur les mêmes nuits, près du plafond fixé par la qualité du label. Sur Tana, ils rangent les nuits aussi bien l'un que l'autre, mais LightGBM sous-estime systématiquement les nuits claires, si bien que le MLP y donne des probabilités plus fiables et de meilleures décisions.

Les étapes suivantes sont la régression logistique, pour savoir si une méthode linéaire suffit, puis éventuellement le modèle séquentiel et un recalibrage des probabilités ajusté sur 2022. Le test temporel sera évalué à la fin, une seule fois, sur tous les modèles retenus.
