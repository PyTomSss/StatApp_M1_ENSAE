from creation_matrices import rendements
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

P = rendements()  # DataFrame pour les rendements passés
F = rendements()  # DataFrame pour les rendements futurs

# Fonction pour créer des features et targets : 
# Pour chaque point dans data, on prend les rendements des window_size jours 
# précédents comme features (X) et les rendements du jour actuel comme target (y).
def create_features_targets(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values.flatten())
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)

# Paramètres
window_size = 10  # Fenêtre temporelle pour les features

# Création des ensembles d'entraînement et de test
X, y = create_features_targets(P, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle Random Forest
model = RandomForestRegressor(n_estimators=30, random_state=42)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Stratégie d'allocation basée sur le classement des rendements prévus
def allocate_based_on_ranking(predictions, top_n):
    rankings = np.argsort(predictions, axis=1)[:, -top_n:]
    allocations = np.zeros_like(predictions)
    for i, ranking in enumerate(rankings):
        allocations[i, ranking] = 1 / top_n
    return allocations

# Choix du nombre d'actifs à inclure dans le portefeuille
top_n = 5
allocations = allocate_based_on_ranking(predictions, top_n)

# Calcul du P&L pour la période de test
daily_pnl = np.sum(predictions * allocations, axis=1)
cumulative_pnl = np.cumsum(daily_pnl)

# Afficher les résultats pour la période de test
print("Cumulative P&L for the test period:", cumulative_pnl[-1])

# Préparation et application de la stratégie pour les données futures
X_future, _ = create_features_targets(F, window_size)
future_predictions = model.predict(X_future)
future_allocations = allocate_based_on_ranking(future_predictions, top_n)

# Calcul du P&L pour la période future
future_daily_pnl = np.sum(future_predictions * future_allocations, axis=1)
future_cumulative_pnl = np.cumsum(future_daily_pnl)

# Afficher les résultats pour la période future
print("Cumulative P&L for the future period:", future_cumulative_pnl[-1])
