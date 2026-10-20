# 🛍️ Simulateur de ventes Black Friday · Prévision des ventes 2025

🇫🇷 Français · [🇬🇧 English](README.en.md)

🔗 **[Démo en ligne](https://prevision-ventes-almiron.streamlit.app)** · Testez le simulateur directement dans votre navigateur.

Application qui **prévoit les ventes quotidiennes de novembre 2025** de 24 produits d'un retailer sportif et permet de **simuler des scénarios de prix et de concurrence** pour préparer les décisions commerciales du Black Friday.

> Projet réalisé dans le cadre de la formation Data Science + IA de **DS4B**, enrichi d'analyses et d'améliorations personnelles (voir [Mes contributions](#-mes-contributions)).

<!-- Ajouter ici une capture du simulateur : docs/img/simulateur.png -->
![Simulateur](docs/img/simulateur.png)

---

## 🎯 Problématique métier

Un retailer sportif doit anticiper ses ventes sur la période clé de l'année (fin octobre à fin novembre) pour décider de ses **remises**, de son **stock** et de son **positionnement prix** face à Amazon, Decathlon et Deporvillage.

Questions auxquelles répond l'application :
- Combien d'unités vendra chaque produit, jour par jour, en novembre 2025 ?
- Quel est l'impact d'une remise appliquée uniquement le jour du Black Friday ?
- Que se passe-t-il si la concurrence baisse ou augmente ses prix de 5 % ?
- La hausse du chiffre d'affaires se traduit-elle par une hausse de la marge ?

## 📦 Données

| Jeu de données | Contenu |
|---|---|
| Ventes 2021-2024 | 24 produits, du 25 octobre au 30 novembre de chaque année (3 552 enregistrements) : prix de base, prix de vente, unités, chiffre d'affaires, catégorie |
| Concurrence | Prix quotidiens d'Amazon, Decathlon et Deporvillage par produit |
| Inférence 2025 | Fin octobre avec ventes réelles (base des lags) et novembre à prévoir |

## 🛠️ Démarche

| Étape | Contenu |
|---|---|
| **Qualité des données** | Types, valeurs manquantes, doublons, cohérence `chiffre d'affaires = prix × unités` |
| **Analyse exploratoire** | Saisonnalité, jours de la semaine, catégories, top produits, distribution des prix face à Amazon |
| **Feature engineering** | Calendrier espagnol (jours fériés, week-end, début et fin de mois), Black Friday et Cyber Monday, lags 1 à 7 et moyenne mobile sur 7 jours par produit, taux de remise, prix moyen de la concurrence et ratio de prix, One Hot Encoding |
| **Validation** | Entraînement 2021-2023, test sur 2024 (découpage temporel) |
| **Modèle** | `HistGradientBoostingRegressor` aux paramètres conservateurs, comparé à une baseline (moyenne) |
| **Diagnostic** | Erreur par produit et par période, réel vs prévu, importance des variables (permutation) |
| **Prévision 2025** | Modèle final entraîné sur 2021-2024, prévision **récursive** jour par jour |
| **Application** | Streamlit, en français |

**Stack :** Python · pandas · scikit-learn · seaborn · Streamlit · GitHub Codespaces · GitHub Copilot (mode agent)

## 📊 Résultats (validation sur 2024)

| | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (moyenne) | 3,67 | 6,69 | 0,00 |
| **Modèle final** | **1,05** | **2,23** | **0,89** |

Le modèle réduit l'erreur de **71 %** par rapport à la méthode simple.

**Exemple de simulation** (Adidas Ultraboost 23, novembre 2025) : une remise de −15 % appliquée uniquement le jour du Black Friday génère **+10,6 % d'unités** et **+8,7 % de chiffre d'affaires**, avec un pic d'environ 62 unités le 28 novembre, cohérent avec l'historique.

## 💡 Mes contributions

| Contribution | Impact |
|---|---|
| **Lags calculés par produit** (et non sur l'ensemble des produits) | Suppression d'une fuite de données qui gonflait artificiellement les métriques |
| **Correction de la date du Black Friday 2024** (la règle « quatrième vendredi » marquait le 22 au lieu du 29 novembre) | MAE de 1,44 à 1,05 (**−27 %**) grâce à la seule correction d'une date |
| **Détection d'un biais de confusion remise / Black Friday** | Explique pourquoi le modèle surestime l'effet des remises hors événement |
| **Expérience pour séparer les deux effets** | Résultat négatif documenté : limite des données, non de la méthode |
| **Application enrichie** | Remise par jour, interface en français, analyse des écarts prix / volume, impact sur la marge, alertes sur les limites du modèle |

## 🔎 Principaux constats

| # | Constat | Conséquence |
|---|---|---|
| 1 | Le modèle surestime l'effet des petites remises (jours de remise sans événement) | Simulations de remise à interpréter avec prudence |
| 2 | La règle « quatrième vendredi » plaçait le Black Friday 2024 au 22 novembre | Corrigé : MAE −27 % |
| 3 | Des lags calculés tous produits confondus créent une fuite de données | Lags calculés par produit et par année |
| 4 | L'erreur dépend davantage du produit que du moment du mois | Nike Pegasus est le produit le moins bien prévu, Trek Marlin le mieux |
| 5 | La remise est la variable la plus importante, devant le Black Friday | Confirme la confusion remise / événement |
| 6 | Une remise appliquée tout le mois est amplifiée par la prévision récursive (« effet boule de neige ») | Usage recommandé : remise limitée au Black Friday |
| 7 | Le modèle n'extrapole pas hors de la plage de prix historique (−15 % à +3 % du prix de base) | Une hausse de +40 % est traitée comme +3 % ; l'application affiche une alerte |

## 🖥️ L'application

- Sélection du produit, ajustement du prix (−50 % à +50 %) et choix des jours concernés : Black Friday seul, semaine du Black Friday, mois entier ou sélection personnalisée
- Trois scénarios de concurrence : prix actuels, −5 % et +5 %
- Indicateurs clés, courbe quotidienne avec le Black Friday mis en évidence et tableau détaillé
- Impact économique : chiffre d'affaires avec et sans ajustement et **analyse des écarts prix / volume** (effet volume, effet prix et contrôle de cohérence), selon la logique du contrôle de gestion
- **Graphique en cascade** : CA sans ajustement → effet volume → effet prix → CA avec ajustement
- Impact sur la marge si un coût unitaire est renseigné
- **Alertes** : résultat atypique (écart de CA supérieur à +100 %) et scénario hors de la plage de prix historique

## 📐 Hypothèses

1. Le comportement de 2025 sera similaire à celui de 2021-2024.
2. Les prix de la concurrence 2025 sont connus ; les scénarios les déplacent tous du même pourcentage.
3. Le prix de base de chaque produit est constant sur la période.
4. Ventes = demande : pas de rupture de stock.
5. Les produits sont indépendants : pas de cannibalisation.
6. Seuls les jours fériés nationaux espagnols sont pris en compte.
7. Black Friday = lendemain du quatrième jeudi de novembre.
8. Sans ajustement, le prix de vente simulé est égal au prix de base (les variations quotidiennes historiques ne sont pas reproduites).
9. Les ventes prévues sont arrondies à l'unité la plus proche ; le chiffre d'affaires est calculé sur ces unités arrondies.
10. Le chiffre d'affaires est calculé sur les prix affichés, probablement TTC ; une analyse financière devrait le convertir en HT.

## ⚠️ Limites

| Limite | Conséquence |
|---|---|
| Données limitées à la saison du Black Friday (≈ 37 jours par an) | Le modèle ne s'applique pas aux autres mois |
| Remises historiquement concentrées sur le Black Friday | Surestimation de l'effet d'une remise hors événement, sous-estimation du pic sans remise |
| Prévision récursive | Les erreurs s'accumulent, surtout avec des remises longues |
| Modèle à base d'arbres | Pas d'extrapolation hors de la plage de prix historique (−15 % à +3 %) |
| Pas d'intervalles de confiance | L'application donne une valeur, pas une fourchette |
| Volume réduit : 24 produits, 4 saisons | Résultats par produit fragiles |
| Données issues d'une formation | La valeur du projet réside dans la méthode, pas dans les chiffres |

**Usage recommandé :** simuler des remises sur la période de l'événement, comme dans la pratique commerciale, et interpréter les résultats comme des scénarios indicatifs.

## 🗺️ Prochaines étapes

1. Intervalles de confiance (scénarios optimiste et pessimiste)
2. Tests de prix hors Black Friday pour mesurer l'effet réel des remises
3. Coûts réels par produit pour optimiser la marge
4. Variables de cannibalisation entre produits d'une même catégorie
5. Déploiement sur Streamlit Community Cloud

## 📁 Structure du projet

```
├── app.py                         # Application Streamlit
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── entrenamiento/         # Ventes et concurrence 2021-2024
│   │   └── inferencia/            # Données 2025
│   └── processed/                 # Données transformées
├── models/
│   └── modelo_final.joblib        # Modèle entraîné
├── notebooks/
│   ├── Entrenamiento.ipynb        # Qualité, EDA, variables, modèle, constats
│   └── forecasting.ipynb          # Préparation des données 2025
└── .github/
    └── copilot-instructions.md    # Règles pour l'assistant IA
```

## ▶️ Lancer l'application

**Avec GitHub Codespaces :** bouton **Code → Codespaces → Create codespace**, puis dans le terminal :

```bash
pip install -r requirements.txt
streamlit run app.py
```

**En local :**

```bash
git clone https://github.com/florenciaalmironfr-svg/prevision-ventes.git
cd prevision-ventes
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💼 Autrice

**Florencia Almiron** · Contrôleuse de gestion | Business Analyst SI Finance | Data & BI | IA appliquée

*Relier métiers, processus et données pour mieux décider.*
