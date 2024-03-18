import numpy as np
import pandas as pd
from print_matrices import retourner_matrices
from sklearn.linear_model import LinearRegression
from regression_linaire_avancee import estimer_beta, sharpe_ratio
import matplotlib.pyplot as plt

sharpe_ratio_optimal = []
sharpes_ratios_is = []
sharpes_ratios_os = []

taille_souhaitee = 200

# Boucle sur le nombre d'actifs
for nb_signals in range(5, taille_souhaitee):
    # Générer les matrices in-sample et out-sample avec le nombre d'actifs actuel
    matrices_in_sample, matrices_out_sample = retourner_matrices(nb_signals=nb_signals, nb_assets=50)
    signals_is, assets_is = matrices_in_sample[0], matrices_in_sample[1]
    signals_os, assets_os = matrices_out_sample[0], matrices_out_sample[1]

    # Estimer les coefficients beta pour les données in-sample
    estimated_beta = estimer_beta(signals_is, assets_is)
    
    # Calculer le portefeuille et les rendements par jour pour les données in-sample
    portefeuille_is = signals_is @ estimated_beta
    rendements_par_jour_par_actifs_is = assets_is * portefeuille_is
    rendements_par_jour_is = np.sum(rendements_par_jour_par_actifs_is, axis=1)
    
    # Calculer le rendement moyen et la volatilité du portefeuille pour les données in-sample
    rendement_moyen_is = np.sum(rendements_par_jour_is)
    volatilite_portefeuille_is = np.std(rendements_par_jour_is)*256
    
    # Calculer le ratio de Sharpe pour les données in-sample
    sharperatio_is = rendement_moyen_is / volatilite_portefeuille_is
    
    # Ajouter le ratio de Sharpe à la liste pour les données in-sample
    sharpes_ratios_is.append(sharperatio_is)
    
    # Calculer le portefeuille et les rendements par jour pour les données out-sample
    portefeuille_os = signals_os @ estimated_beta
    rendements_par_jour_par_actifs_os = assets_os * portefeuille_os
    rendements_par_jour_os = np.sum(rendements_par_jour_par_actifs_os, axis=1)
    
    # Calculer le rendement moyen et la volatilité du portefeuille pour les données out-sample
    rendement_moyen_os = np.sum(rendements_par_jour_os)
    volatilite_portefeuille_os = np.std(rendements_par_jour_os)*256
    
    # Calculer le ratio de Sharpe pour les données out-sample
    sharperatio_os = rendement_moyen_os / volatilite_portefeuille_os
    
    # Ajouter le ratio de Sharpe à la liste pour les données out-sample
    sharpes_ratios_os.append(sharperatio_os)

# Plot des deux graphiques
plt.figure(figsize=(10, 5))

# Plot des données in-sample
plt.plot(range(5, taille_souhaitee), sharpes_ratios_is, label='In-sample')

# Plot des données out-sample
plt.plot(range(5, taille_souhaitee), sharpes_ratios_os, label='Out-sample')

plt.xlabel('Nombre d\'actifs')
plt.ylabel('Ratio de Sharpe')
plt.title('Evolution du ratio de Sharpe en fonction du nombre d\'actifs')
plt.legend()
plt.grid(True)
plt.show()
