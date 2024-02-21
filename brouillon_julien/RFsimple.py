from creation_matrices import rendements
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

C = rendements(1)

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

sharpe_ratios = [] 

u = 0
v = 0

for l in range(200):

    F = rendements(1)
    # Initialisation de la variable pour stocker le meilleur rendement et l'indice de l'actif correspondant

    # Nombre total de jours pour lesquels faire des prédictions
    nombre_jours_predictions = F.shape[0] - look_back

    # Matrice pour stocker les résultats : 19 zéros et 1 un par ligne
    resultats = np.zeros((nombre_jours_predictions, num_assets))

    # Préparation des noms de colonnes pour correspondre à l'entraînement
    column_names = [f'N-{i}' for i in range(look_back, 0, -1)]

    for jour in range(nombre_jours_predictions):
        rendement_max = -np.inf
        index_meilleur_actif = -1

        for i in range(num_assets):
            rendements_actif_i = F.iloc[jour:jour+look_back, i]
            data_F = pd.DataFrame([rendements_actif_i.values], columns=column_names)
            prediction_actif_i = models[f'asset_{i}'].predict(data_F)

            # Comparaison pour trouver le rendement maximum
            if prediction_actif_i > rendement_max:
                rendement_max = prediction_actif_i
                index_meilleur_actif = i
        
        # Stocker le résultat (1 pour le meilleur actif, 0 pour les autres)
        resultats[jour, index_meilleur_actif] = 1


    # Affichage des résultats
    print(resultats)
    rendements_reels = F.iloc[30:]
    rendements_journaliers = np.sum(resultats * rendements_reels, axis=1)
    sharpe_ratio = np.mean(rendements_journaliers)/np.std(rendements_journaliers)
    v = v + sharpe_ratio
    rendement_portefeuille = np.sum(rendements_journaliers)
    print(v/(l+1))
    print(rendement_portefeuille)
    u = u + rendement_portefeuille
    print(u/(l+1))
    sharpe_ratios.append(sharpe_ratio)  

sharpe_moyen = np.mean(sharpe_ratios)

# Trier les ratios de Sharpe
sharpe_ratios_sorted = np.sort(sharpe_ratios)

# Calculer la fonction de répartition pour chaque valeur triée
cdf = np.arange(1, len(sharpe_ratios_sorted) + 1) / len(sharpe_ratios_sorted)

# Créer le graphique
plt.plot(sharpe_ratios_sorted, cdf, marker='o', linestyle='none', color='blue')

# Ajouter la ligne verticale pour le Sharpe moyen
plt.axvline(x=sharpe_moyen, color='red', linestyle='--', label=f'Sharpe Moyen: {sharpe_moyen:.2f}')

plt.xlabel('Ratio de Sharpe')
plt.ylabel('Répartition')
plt.title('')
plt.legend()

# Afficher une grille pour mieux visualiser les niveaux de répartition
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()