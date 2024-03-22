#On importe les packages ici : 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Cette fonction permet de générer un modèle, à savoir : 
# - Matrice de Signaux
# - Betas
# - Matrice de Noise
# - Matrice des assets
#Elle prend en paramètres (..............)
#Pour l'instant on ne permet pas à la fonction de faire suivre plusieurs lois différentes pour les assets car on n'a pas trop compris le fonctionnement
def random_signals_generator(nb_signals, nb_dates, vol_signals, correl_signals, nb_assets, vol_assets, correl_assets, 
                            signal_to_noise, rank_betas, rand_vector_signals = None, rand_vector_betas = None, 
                            rand_vector_noise = None) :
        
    #Définition des matrices de var-covar
    cov_signals = vol_signals**2 * pd.DataFrame(correl_signals * np.ones((nb_signals,nb_signals)) + (1-correl_signals)*np.eye(nb_signals))
    cov_assets = vol_assets**2 * pd.DataFrame(correl_assets * np.ones((nb_assets,nb_assets)) + (1-correl_assets)*np.eye(nb_assets))

    #Potentiel stockage de l'aléa pour réutilisation
    if rand_vector_betas is None:
        betas = pd.DataFrame(np.random.randn(nb_signals,nb_assets))
    else:
        betas = pd.DataFrame(rand_vector_betas)

    #On normalise les betas (si oui, pourquoi pas juste à la fin ?) ? Utilité réelle de la SVD ? Pourquoi on renormalise ensuite ? 
    betas *= np.sqrt(signal_to_noise * np.trace(cov_assets) / np.trace(betas.T @ cov_signals @ betas))
    u,d,v = np.linalg.svd(betas)
    betas = pd.concat([pd.DataFrame(d[k] * np.outer(u[:,k],v[k,:]),index=betas.index,columns=betas.columns) for k in range(rank_betas)],keys=range(rank_betas)).groupby(level=1).sum()
    betas *= np.sqrt(signal_to_noise * np.trace(cov_assets) / np.trace(betas.T @ cov_signals @ betas))

    #Création de la matrice de Noise (quelles conditions doit elle respecter ?)
    cov_noise = cov_assets - betas.T @ cov_signals @ betas
    # Optimisable???
    d,P = np.linalg.eigh(cov_noise)
    cov_noise = pd.DataFrame(P @ np.diag(np.maximum(d,1e-10)) @ P.T)
    print('signal to noise = ' + str(np.round(np.trace(betas.T @ cov_signals @ betas) / np.trace(cov_assets),3))) #Vérification retirable

    #Cholesky decomposition
    sqrt_cov_signals = np.linalg.cholesky(cov_signals)
    sqrt_cov_noise = np.linalg.cholesky(cov_noise)

    #Potentiel stockage de l'aléa pour réutilisation
    if rand_vector_signals is None:
        signals = pd.DataFrame(np.random.randn(nb_dates,nb_signals) @ sqrt_cov_signals.T)
    else:
        signals = pd.DataFrame(rand_vector_signals @ sqrt_cov_signals.T)

    if rand_vector_noise is None:   
        noise = pd.DataFrame(np.random.randn(nb_dates,nb_assets) @ sqrt_cov_noise.T)
    else:
        noise = pd.DataFrame(rand_vector_noise @ sqrt_cov_noise.T)
    
    #Création de la matrice d'assets    
    assets = signals @ betas + noise

    return [signals , assets]

def pred_sharpe_signals(assets_matrix, signals_matrix, linear_model) : 
    
    w_model = signals_matrix @ linear_model
    pnl_model = (w_model * assets_matrix).sum(axis=1) #Pourquoi la sum et pas juste la multiplication ?
    sharpe = pnl_model.mean() / pnl_model.std()*16
    
    return sharpe