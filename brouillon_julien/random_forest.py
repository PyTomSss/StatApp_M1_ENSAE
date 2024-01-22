from creation_matrices import rendements
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Supposons que P et F sont vos DataFrames de rendements passés et futurs
P = rendements() # Votre DataFrame pour les rendements passés
F = rendements() # Votre DataFrame pour les rendements futurs

# Séparation des données en ensembles d'entraînement et de test
# Ici, nous utilisons P pour l'entraînement et F pour le test (ou la validation)
X_train, X_test, y_train, y_test = train_test_split(P.iloc[:-1], P.iloc[1:], test_size=0.2, random_state=42)

# Création du modèle XGBoost
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# Utiliser le modèle pour faire des prédictions sur F
F_pred = model.predict(F)

# Suggestion d'allocation (ex : choisir l'actif avec le rendement prédit le plus élevé chaque jour)
allocation = F_pred.argmax(axis=1)
