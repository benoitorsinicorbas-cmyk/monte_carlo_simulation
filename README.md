# Simulation Monte Carlo (GBM)

Petit script Python qui projette la trajectoire d'un actif sur un an à l'aide d'un mouvement brownien géométrique. Le drift et la volatilité ne sont pas posés à la main : ils sont estimés directement sur l'historique des cours récupéré via Yahoo Finance.

L'idée de départ était de regarder à quoi pouvait ressembler la distribution d'un indice à horizon un an, puis de pouvoir rejouer le même exercice sur n'importe quel ticker en changeant une seule ligne.

## Ce que fait le script

1. Télécharge l'historique de cours (`Close` ajusté) depuis 2015.
2. Calcule les rendements log quotidiens, puis en déduit un drift et une volatilité annualisés (base 252 jours de bourse).
3. Simule 1000 trajectoires sur 252 pas avec la dynamique GBM.
4. Sort les statistiques utiles : niveau moyen attendu et intervalle à 90% (5e et 95e percentiles).
5. Trace deux graphiques : un échantillon de trajectoires et la distribution des niveaux finaux.

## Installation

Besoin de Python 3 et de quatre paquets :

```bash
pip install yfinance pandas numpy matplotlib
```

## Utilisation

```bash
python nasdaq_monte_carlo.py
```

Les résultats chiffrés s'affichent dans la console et les deux graphiques s'ouvrent à la suite.

## Paramètres

Tout se règle en haut du fichier :

- `ticker` : l'actif à simuler. Par défaut `COV.PA`, mais n'importe quel symbole Yahoo Finance fonctionne (`^IXIC`, `QQQ`, etc.).
- `start_date` / `end_date` : la fenêtre d'historique utilisée pour l'estimation. `end_date = None` va jusqu'à aujourd'hui.
- `T` : horizon de simulation en années.
- `n_steps` : nombre de pas (252 pour des pas journaliers).
- `n_sims` : nombre de trajectoires.

## Méthode

On part de la forme classique du GBM :

$$S_{t+\Delta t} = S_t \cdot \exp\left[(\mu - \tfrac{1}{2}\sigma^2)\,\Delta t + \sigma\sqrt{\Delta t}\,Z\right]$$

avec $Z \sim \mathcal{N}(0,1)$. Le drift $\mu$ et la volatilité $\sigma$ sont annualisés à partir des moments empiriques des rendements log.

## À garder en tête

C'est un modèle d'illustration, pas un outil de prévision. Le GBM suppose une volatilité constante, des rendements log gaussiens et indépendants, et il extrapole simplement les paramètres passés vers le futur. Autrement dit, il sous-estime largement les queues de distribution et ne capte ni les régimes de volatilité ni les sauts. À prendre pour ce que c'est : une manière visuelle de raisonner sur une enveloppe de scénarios.
