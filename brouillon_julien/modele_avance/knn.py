from print_matrices import retourner_matrices
import numpy as np 
import pandas as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from prediction_sharpe import prediction_sharpe

k = 5  
modele = KNeighborsRegressor(n_neighbors=k)
nb_simus = 10
nb_signals = 5
nb_dates_in_sample = 260
nb_dates_out_sample = 260
nb_assets = 5

vol_assets = 0.2/16
correl_assets = 0.8

# vol_signals = 1.0
# correl_signals = 0.0

vol_signals = 0.2/16
correl_signals = 0.8

signal_to_noise = 0.001   # la relation entre rendements X et signaux Y n'est pas parfaitement proportionnelle à cause de ça ?

rank_betas = 5

