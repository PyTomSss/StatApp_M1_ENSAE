from brouillon_julien.modele_initial.creation_matrices import rendements
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

P = rendements()

X = P.iloc[:, :-1]
y = P.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement d'un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calcul des rendements moyens et de la matrice de covariance
mean_returns = P.mean()
cov_matrix = P.cov()

# Nombre d'actifs dans le portefeuille
num_assets = len(mean_returns)

# Fonction pour calculer le rendement et la volatilité du portefeuille
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

# Optimisation
from scipy.optimize import minimize

def minimize_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# Contraintes
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # La somme des poids est 1
bounds = tuple((0, 1) for asset in range(num_assets))  # Les poids entre 0 et 1
initial_guess = num_assets * [1. / num_assets,]  # Distribution uniforme initiale

# Optimisation pour minimiser la volatilité
optimal_weights = minimize(minimize_volatility, initial_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

print("Répartition Optimale des Actifs :")
for i, weight in enumerate(optimal_weights.x):
    print(f"Actif {i}: {weight:.2f}")