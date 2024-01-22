from creation_matrices import rendements
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Chargement des données
P = rendements()  # DataFrame pour les rendements passés

# Séparation des données de P en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(P.iloc[:-1], P.iloc[1:], test_size=0.2, random_state=42)

# Création et entraînement du modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test et évaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# Stratégie d'allocation basée sur les prédictions du modèle
# Cette stratégie peut être ajustée selon vos critères spécifiques
# Par exemple, choisir l'actif avec le rendement prédit le plus élevé pour chaque jour
allocation_strategy = y_pred.argmax(axis=1)

# Appliquer la stratégie d'allocation sur F (en pratique, F est inconnu ici)
# Notez que F_pred est inutilisé dans ce contexte, mais montré pour l'exemple
F = rendements()  # En pratique, F est inconnu ici
F_pred = model.predict(F)
allocation_on_F = F_pred.argmax(axis=1)

# Afficher l'allocation pour le premier jour à titre d'exemple
print("Allocation pour le premier jour sur F :")
print(allocation_on_F[0])
