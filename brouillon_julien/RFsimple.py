from creation_matrices import rendements
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

C = rendements(1)
F = rendements(1)

# Paramètre look_back définit le nombre de jours précédents à considérer
look_back = 30  
num_assets = 20  # Nombre d'actifs
models = {}  # Dictionnaire pour stocker les modèles

for asset in range(num_assets):
    # Préparation des données pour cet actif
    P = C.iloc[:, asset]  # Sélection de la colonne pour l'actif actuel
    features = [P.shift(i) for i in range(look_back, 0, -1)]
    features.append(P)

    data = pd.concat(features, axis=1).dropna()
    data.columns = [f'N-{i}' for i in range(look_back, 0, -1)] + ['Target']

    X = data.drop('Target', axis=1)
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création et entraînement du modèle de Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Stockage du modèle
    models[f'asset_{asset}'] = model

    # Evaluation optionnelle du modèle
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Actif {asset} - Mean Squared Error: {mse}")


### Portefeuille

# Prediction J31

# Initialisation de la variable pour stocker le meilleur rendement et l'indice de l'actif correspondant

# Nombre total de jours pour lesquels faire des prédictions
nombre_jours_predictions = F.shape[0] - look_back

# Matrice pour stocker les résultats : 19 zéros et 1 un par ligne
resultats = np.zeros((nombre_jours_predictions, num_assets))

column_names = [f'N-{i}' for i in range(look_back, 0, -1)]

for jour in range(nombre_jours_predictions):
    rendement_max = -np.inf
    index_meilleur_actif = -1

    for i in range(num_assets):
        # Extraire les données pour le jour actuel et l'actif actuel
        rendements_actif_i = F.iloc[jour:jour+look_back, i].values.reshape(1, -1)
        
        # Prédiction pour l'actif actuel
        prediction_actif_i = models[f'asset_{i}'].predict(rendements_actif_i)

        # Comparaison pour trouver le rendement maximum
        if prediction_actif_i > rendement_max:
            rendement_max = prediction_actif_i
            index_meilleur_actif = i
    
    # Stocker le résultat (1 pour le meilleur actif, 0 pour les autres)
    resultats[jour, index_meilleur_actif] = 1

# Affichage des résultats
print(resultats)
rendements_reels = F.iloc[30:]
rendement_portefeuille = np.sum(np.sum(resultats * rendements_reels, axis=1))
print(rendement_portefeuille)
