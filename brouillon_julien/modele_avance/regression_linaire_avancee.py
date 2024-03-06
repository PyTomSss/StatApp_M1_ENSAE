from print_matrices import retourner_matrices
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

matrices_in_sample, matrices_out_sample = retourner_matrices()
signals_is, assets_is = matrices_in_sample[0], matrices_in_sample[1]

def estimer_beta(signals, assets):
    lr = LinearRegression()
    lr.fit(assets, signals)
    return lr.coef_

estimated_beta = estimer_beta(signals_is, assets_is)

portfeuille = signals_is@estimated_beta
rendements_par_jour_par_actifs = assets_is*portfeuille
rendements_par_jour = np.mean(rendements_par_jour_par_actifs, axis=1)

def sharpe_ratio(rendements_par_jour):
    rendement_moyen = np.mean(rendements_par_jour)
    volatilite_portefeuille = np.std(rendements_par_jour)
    sharpe_ratio = rendement_moyen/volatilite_portefeuille
    return sharpe_ratio
