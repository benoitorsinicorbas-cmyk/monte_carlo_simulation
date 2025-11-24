# nasdaq_monte_carlo.py
# Simulation de Monte Carlo du Nasdaq (^IXIC) avec un GBM

# --- Imports ---
# A installer si besoin :
# pip install yfinance pandas matplotlib numpy

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------
# 1. Paramètres de base
# ------------------------------
ticker = "^IXIC"        # Nasdaq Composite ; tu peux mettre "QQQ" ou autre
start_date = "2020-01-01"
end_date = None          # jusqu'à aujourd'hui

# Horizon de la simulation
T = 1.0          # en années (1 an)
n_steps = 252    # nombre de pas (jours de bourse)
n_sims = 1000    # nombre de trajectoires simulées

# ------------------------------
# 2. Récupération des données
# ------------------------------
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

if data.empty:
    raise ValueError("Pas de données récupérées. Vérifie le ticker ou les dates.")

# On utilise la colonne Close (déjà ajustée)
prices = data["Close"].dropna()

# Si jamais yfinance renvoie un DataFrame à une colonne, on squeeze en Series
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]

# On force en float (Series de float)
prices = prices.astype(float)

# Niveau initial (dernier cours)
last_value = prices.iloc[-1]          # doit être un scalaire np.float64
S0 = float(last_value)                # conversion explicite

print(f"Niveau initial S0 = {S0:.2f}")

# ------------------------------
# 3. Estimation du drift (mu) et de la volatilité (sigma)
# ------------------------------
# Rendements log quotidiens (Series)
log_returns = np.log(prices / prices.shift(1)).dropna()

# On convertit explicitement en float pour éviter d'avoir une Series
mu_daily = float(log_returns.mean())      # drift quotidien
sigma_daily = float(log_returns.std())    # volatilité quotidienne

# Passage en base annuelle (252 jours de bourse)
mu = mu_daily * 252
sigma = sigma_daily * np.sqrt(252)

print(f"Drift annuel estimé (mu)   = {mu:.4f}")
print(f"Volatilité annuelle (sigma) = {sigma:.4f}")

# ------------------------------
# 4. Simulation Monte Carlo (GBM)
# ------------------------------
dt = T / n_steps  # pas de temps en années

# Matrice des trajectoires : (n_steps + 1) x n_sims
paths = np.zeros((n_steps + 1, n_sims))
paths[0] = S0

# Tirages aléatoires ~ N(0,1)
Z = np.random.normal(size=(n_steps, n_sims))

for t in range(1, n_steps + 1):
    # Formule du GBM :
    # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    paths[t] = paths[t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
    )

# ------------------------------
# 5. Analyse simple des résultats
# ------------------------------
S_T = paths[-1, :]  # niveau de l'indice à l'horizon T

expected_price = S_T.mean()
p5 = np.percentile(S_T, 5)
p95 = np.percentile(S_T, 95)

print(f"\n=== Résultats à horizon {T} an(s) pour {ticker} ===")
print(f"Niveau moyen simulé : {expected_price:.2f}")
print(f"Intervalle 90% (5e - 95e percentile) : [{p5:.2f} ; {p95:.2f}]")

# ------------------------------
# 6. Graphiques
# ------------------------------

# a) Quelques trajectoires simulées
plt.figure(figsize=(10, 6))
n_plot = 20  # nombre de trajectoires à afficher
for i in range(n_plot):
    plt.plot(paths[:, i], linewidth=0.8)

plt.title(f"Simulation Monte Carlo de {ticker} ({n_plot} trajectoires sur {n_sims})")
plt.xlabel("Pas de temps (jours de bourse)")
plt.ylabel("Niveau simulé")
plt.grid(True)
plt.tight_layout()
plt.show()

# b) Distribution du niveau final
plt.figure(figsize=(8, 5))
plt.hist(S_T, bins=50, density=True)
plt.title(f"Distribution du niveau simulé de {ticker} à horizon {T} an(s)")
plt.xlabel("Niveau final simulé")
plt.ylabel("Densité")
plt.grid(True)
plt.tight_layout()
plt.show()
