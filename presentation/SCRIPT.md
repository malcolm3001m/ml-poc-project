# Script de présentation — DAT0424

**Format des indications :** `[entre crochets]` = ce que vous faites (cliquer, défiler, montrer). **Texte en gras** = ce que vous dites.

**Durée cible :** 6 à 8 minutes pour le récit, puis Q&A.

---

## 🎬 OUVERTURE — Section 1 · Le problème

`[Page chargée sur la section 1. La grande question s'affiche : "Combien vaut vraiment un joueur à son prime ?"]`

> **« Quand Manchester United a signé Antony pour 95 millions d'euros en 2022, le marché pensait avoir un crack à 130 millions. Trois ans plus tard, sa valeur est tombée à 12 millions. Les clubs perdent des centaines de millions sur des décisions de transfert mal calibrées. »**

> **« Aujourd'hui, je vous présente un outil qui pose une question simple : combien un joueur va-t-il valoir au sommet de sa carrière, à partir uniquement de ses données biographiques et de performance ? »**

`[Pointez vers les 5 chiffres sous l'intro — 39 226 joueurs · 189 nationalités · 34 variables · 1,53 M€ d'erreur moyenne · R² 0,798]`

> **« 39 000 joueurs, 34 variables, et une erreur moyenne d'1,5 million sur le pic prédit. On va voir comment, pourquoi, et surtout — qui ça intéresse. »**

`[Pause 1 seconde]`

---

## 📊 Section 2 · Les données

`[Cliquez sur "02 · Les données" dans la nav latérale, OU faites défiler]`

> **« Première étape : qu'est-ce qu'on a comme matière première ? »**

`[L'histogramme s'affiche]`

> **« Voici la distribution des valeurs de pic des joueurs. Le joueur médian vaut 500 000 euros. Mbappé en vaut 200 millions. C'est un facteur 400. Le modèle doit savoir prédire à toutes ces échelles — c'est notre premier défi technique. »**

`[Faites défiler doucement jusqu'à la carte des pays]`

> **« On a regardé d'où viennent les joueurs les plus valorisés. Mais attention — la principale leçon d'un cours de viz que j'ai eu cette année, c'est que les données pondérées sont meilleures que les données brutes. Si je vous montrais le nombre de joueurs par pays, je vous montrerais juste la population. »**

> **« Là, on montre la valeur médiane par pays — uniquement pour les pays avec au moins 50 joueurs. Ça isole la densité de talent, pas la taille du pays. Brésil, France, Espagne, Argentine en tête. Sans surprise — mais maintenant on peut le défendre. »**

`[Faites défiler — boîte de moustaches positions, scatter âge]`

> **« Le poste compte — les attaquants tirent plus haut. L'âge tout seul ne dit pas grand-chose — et on assume une limite ici : c'est l'âge actuel, pas l'âge à l'apogée. On y reviendra. »**

---

## 🔧 Section 3 · L'approche

`[Cliquez sur "03 · L'approche"]`

> **« Étape deux : comment on s'attaque à ce problème ? »**

`[Le callout vert s'affiche — "La cible est brutale"]`

> **« Premier choix technique, en français simple : la cible est brutale. Une erreur de 500 000 sur Mbappé à 180 millions, c'est rien. La même erreur de 500 000 sur un joueur médian à 500 000, c'est 100 % d'erreur. »**

> **« La solution : on log-transforme la cible. Le modèle traite ensuite un joueur à 1 million et un joueur à 100 millions avec la même importance. »**

`[Le carrousel 3D apparaît — quatre cartes qui tournent]`

> **« On a testé quatre algorithmes — vous les voyez tourner ici. Décision tree comme baseline. Random Forest, l'ensemble classique. XGBoost, le champion du gradient boosting. Et CatBoost. »**

`[Pointez vers la carte CatBoost — celle avec le contour vert et le trophée]`

> **« CatBoost gagne. Pas par hasard. Les variables catégorielles — la position, le pays, le championnat — XGBoost les encode en entiers, puis les traite comme un ordre. C'est absurde : France = 12, Italie = 15. CatBoost les gère nativement avec une protection anti-fuite intégrée. Mêmes données, mêmes variables : XGBoost obtient 0,759, CatBoost 0,798. Toute la différence est dans le traitement des catégorielles. »**

---

## 🏆 Section 4 · Les résultats

`[Cliquez sur "04 · Les résultats"]`

> **« Est-ce que ça a marché ? Spoiler : oui. »**

`[Les quatre grands chiffres s'affichent — le R² 0,798 brille avec sa bordure conique animée]`

> **« CatBoost — R² test de 0,798. Erreur moyenne de 1,53 millions. Validation croisée à 5 plis : 0,777 plus ou moins 0,005 — bande extrêmement serrée. Écart train/test : +0,007, surajustement quasiment nul. »**

`[Faites défiler à la table des modèles]`

> **« Les quatre modèles s'empilent comme prévu. Arbre simple en bas, CatBoost en haut. »**

`[Faites défiler au tableau des baselines]`

> **« Mais 0,80 c'est bon par rapport à quoi ? On a deux baselines. Prédire la médiane à tout le monde : R² négatif de 0,08. Régression linéaire : 0,12 — elle s'écroule parce qu'elle traite les catégorielles comme des ordres. CatBoost coupe l'erreur moyenne de la baseline triviale de 54 %. »**

`[Faites défiler aux barres de CV]`

> **« La validation croisée — cinq plis. CatBoost reste dans une bande de plus ou moins 0,005. Notre 0,798 n'est pas un coup de chance. »**

`[Faites défiler à l'importance des variables]`

> **« Et qu'a appris le modèle ? Les trois variables les plus prédictives : la compétition du club actuel, le nombre de matchs joués, et les minutes en carrière. Traduit : à quel niveau on joue, combien de temps on a joué, et sur quelle durée. »**

---

## 🔬 Section 5 · Mise à l'épreuve

`[Cliquez sur "05 · Mise à l'épreuve"]`

> **« Maintenant la question qu'un examinateur sceptique va poser : comment ça pourrait être faux, et est-ce qu'on a testé ? »**

`[Le scatter prédiction-vs-réel s'affiche]`

> **« Premier test, la calibration. Chaque point est un joueur du test. Un modèle parfait pose chaque point sur la diagonale. On est bien calibré jusqu'à 20 millions. Au-dessus — les superstars de la queue droite — on sous-prédit systématiquement. Le modèle reste prudent là où il devrait extrapoler. »**

`[Faites défiler aux barres par poste]`

> **« Où le modèle échoue par poste ? L'erreur en pourcentage est à peu près constante entre tous les postes — pas de biais positionnel. Le modèle ne traite pas mal les gardiens. »**

`[Faites défiler à l'ablation]`

> **« On a identifié une variable suspecte : le championnat actuel du joueur. Un joueur du PSG est valorisé parce qu'il est au PSG — possible fuite. On a testé : on retire la variable, on réentraîne. »**

`[Pointez aux trois cartes vert/orange/gris]`

> **« Résultat : on perd 0,002 de R² — virtuellement rien. Les autres variables ont absorbé son signal. Le modèle n'est pas qu'une simple mémorisation du prestige des championnats. »**

`[Faites défiler au tableau du diagnostic par tranche]`

> **« Enfin — la métrique qu'un humain ressent vraiment : l'erreur en pourcentage par tranche de prix. R² en euros est dominé par le fait qu'on a la bonne magnitude. En pourcentage par tranche, on voit la vérité : le modèle est meilleur sur la zone moyenne — 1 à 20 millions — où il a le plus de données. Il sous-prédit l'élite, sur-prédit les petits joueurs. C'est un biais vers la moyenne — et notre feuille de route propose la correction. »**

---

## 🎯 Section 6 · Essayez vous-même

`[Cliquez sur "06 · Essayez vous-même". C'est le moment fort de la démo.]`

> **« Maintenant — la partie la plus parlante. À quoi ça sert dans la vraie vie ? »**

`[La bande "Top 10 opportunités" est visible en haut]`

> **« Voici ce que le modèle remarque dans la base de test. Dix joueurs à l'apogée, jeunes — 28 ans ou moins — dont le prix actuel sous-évalue le potentiel selon nos fondamentaux. Classés par gain absolu en euros. »**

`[Cliquez sur la carte Rayan Cherki — première de la liste]`

> **« Rayan Cherki, 22 ans, attaquant — actuellement à 65 millions. Le modèle voit 100 millions de pic. C'est 35 millions de marge potentielle. »**

`[Le bloc joueur apparaît avec la photo, les highlights, le signal vert "▲ Potentiel"]`

> **« Vous voyez sa photo, ses statistiques qui le placent dans le top 5 % ou top 10 % de la population. Le signal est vert — ▲ Potentiel. La sortie brute du modèle est 100 millions, la valeur actuelle 65 — la différence est honnête. »**

`[Cliquez sur "Joueurs TEST uniquement" pour le désactiver. Puis cherchez un nom dans la barre de recherche.]`

> **« On peut aussi chercher n'importe quel joueur. »**

`[Tapez "Cas" dans la barre de recherche. Le mot Casemiro apparaît en gris clair en autocomplétion. Appuyez sur Entrée.]`

> **« Casemiro — 34 ans, ancien Real Madrid, maintenant Manchester United. »**

`[Le bloc se met à jour. Le signal est gris avec une horloge "🕘 Pic historique"]`

> **« Et là — important — le modèle prédit 83 millions. Mais Casemiro est en déclin : sa valeur actuelle est de 8 millions seulement, alors que son pic était de 80 millions. C'est moins 90 % depuis le sommet. »**

> **« Le système le détecte. Il dit : "Pic historique reconnu. Ce pic est dans le passé — pas un potentiel à capturer." On ne ment pas au scout en lui disant qu'il peut acheter Casemiro à 8 millions et le revendre à 83. Le modèle reconnaît juste le profil d'une apogée déjà atteinte. »**

`[Faites défiler à la comparaison des 4 modèles en bas]`

> **« Et on peut comparer les quatre modèles pour ce joueur — la barre rouge en pointillé, c'est la valeur réelle. CatBoost en vert est le plus proche. »**

---

## ⚠️ Section 7 · Limites assumées

`[Cliquez sur "07 · Limites assumées". Les trois cartes empilées apparaissent. Faites défiler doucement pour qu'elles bougent.]`

> **« Trois limites qu'on assume. Trois — pas une qu'on cache. »**

`[Les cartes défilent et se rétractent en arrière au scroll]`

> **« Premièrement — c'est conceptuel — notre modèle est rétrospectif. Les variables sont cumulatives sur la carrière, la cible est le pic de carrière. Pour un joueur retraité, les variables incluent l'après-pic. La reformulation honnête : "Prédire le pic à 25 ans à partir des données jusqu'à 21 ans" — c'est dans la feuille de route. »**

> **« Deuxièmement — l'âge dans les variables est l'âge actuel, pas l'âge à l'apogée. Bruit ajouté, mais pas de biais systématique. »**

> **« Troisièmement — notre split train/test est aléatoire, pas temporel. Un modèle déployé s'entraînerait sur les joueurs ayant peak avant 2018 et testerait sur ceux qui peakent en 2018 ou plus tard. C'est aussi dans la feuille de route. »**

`[Ouvrez le panneau "FAQ de défense" si vous voulez montrer que vous avez les réponses prêtes — sinon passez]`

---

## 🚀 Section 8 · La suite

`[Cliquez sur "08 · La suite". Les phases numérotées en grand apparaissent.]`

> **« Si cette preuve de concept devenait une v1 — cinq évolutions. Chacune testable aujourd'hui, aucune n'exige de nouvelles données. »**

`[Survolez le titre "Reformulation à l'âge d'apogée" pour déclencher l'effet de texte hacker, ou pointez simplement.]`

> **« Un — geler les variables à 21 ans. Ça transforme le modèle de rétrospectif à véritablement prospectif. »**

`[Pointez phase 02]`

> **« Deux — split train/test temporel. On teste si le modèle généralise entre les époques. »**

`[Pointez phase 03]`

> **« Trois — régression quantile. Au lieu de dire "Mbappé vaut 130 millions", on dit "Mbappé vaut entre 110 et 180 millions, confiance à 80 %". Beaucoup plus utile pour un scout. »**

`[Pointez phase 04]`

> **« Quatre — SHAP. Pour chaque prédiction, on explique précisément quelle variable contribue à monter ou descendre la valeur. Le modèle devient un outil de négociation, pas une boîte noire. »**

`[Pointez phase 05]`

> **« Cinq — variables de trajectoire. Au lieu des stats cumulatives, on regarde la tendance — buts par 90 minutes, croissance des minutes par saison. On capture si un joueur monte ou plafonne. »**

`[Faites défiler au callout final amber]`

> **« Le brief de monsieur Desjuzeur dit qu'un travail théoriquement défendable, même pas encore réalisé, compte dans la preuve de concept. Ces cinq évolutions sont testables. Elles transforment ce qui est aujourd'hui un modèle unique en fondation d'un vrai produit. »**

---

## 🎬 FERMETURE

`[Restez sur le footer. Laissez la dernière phrase respirer.]`

> **« Le pari de ce projet — prédire le pic de marché d'un joueur à partir de sa biographie. Quatre algorithmes testés, 34 variables, une méthodologie cross-validée. CatBoost gagne avec un R² de 0,798. »**

> **« Et plus important : un outil concret. Voilà Rayan Cherki — 35 millions de potentiel selon le modèle. À un scout de valider. À un club de décider. »**

> **« Merci. Questions ? »**

---

## 📋 Cue-card — points à ne PAS oublier

| À ne pas faire | Pourquoi |
|---|---|
| ❌ Ne pas dire « R² is 0.798 » | Commencez par le problème, pas par les chiffres |
| ❌ Ne pas dire « TransformedTargetRegressor » | Dites « on log-transforme la cible » |
| ❌ Ne pas s'excuser pour les limites | Volontairement les exposer = crédibilité |
| ❌ Ne pas dire « on a manqué de temps » | Dire « la prochaine itération c'est… » |
| ❌ Ne pas lire l'écran | L'app est votre support, pas votre script |
| ❌ Ne pas survoler la feuille de route | C'est ce qui compte comme part de la PoC selon le brief |

## 🎯 Chiffres à mémoriser

| Chose | Valeur |
|---|---|
| Joueurs | **39 226** |
| Variables | **34** |
| R² test CatBoost | **0,798** |
| R² val. croisée | **0,777 ± 0,005** |
| MAE CatBoost | **1,53 M€** |
| Écart train/test | **+0,007** |
| Baseline médiane | R² −0,08, MAE 3,33 M€ |
| Réduction d'erreur vs trivial | **54 %** |
| Ablation de fuite | **−0,002 R²** |

## ⏱ Repère temporel

- Ouverture + Section 1 : ~45 s
- Section 2 (données) : ~75 s
- Section 3 (approche) : ~60 s
- Section 4 (résultats) : ~75 s
- Section 5 (mise à l'épreuve) : ~75 s
- **Section 6 (démo) : ~90 s** — le moment fort
- Section 7 (limites) : ~45 s
- Section 8 (suite) : ~60 s
- Fermeture : ~15 s

**Total : ~7 minutes 30** + Q&A

---

*Répétez à froid une fois. Notez où le récit traîne. Répétez chronométré une fois.
Après deux passes vous êtes prêt.*
