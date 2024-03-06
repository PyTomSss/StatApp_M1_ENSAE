

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso

import pandas as pd
import numpy as np
import time 
import os 
#matplotlib inline

seeds=np.random.randint(0, 1000000, size=1000)

def prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas):
    
    liste_sharpes_predit = []
    liste_sharpes_optimaux_os=[]
    écart_sharpe_moyen=[]
    
    cov_assets = vol_assets**2 * pd.DataFrame(correl_assets * np.ones((nb_assets,nb_assets)) + (1-correl_assets)*np.eye(nb_assets))
    cov_signals = vol_signals**2 * pd.DataFrame(correl_signals * np.ones((nb_signals,nb_signals)) + (1-correl_signals)*np.eye(nb_signals))

    for i in range(nb_simus): 
        #partie generation des matrices de signaux et de rendements, d'entrainement et de test 
        np.random.seed(seeds[i])
        betas = pd.DataFrame(np.random.randn(nb_signals,nb_assets)) 
        betas *= np.sqrt(signal_to_noise * np.trace(cov_assets) / np.trace(betas.T @ cov_signals @ betas))
        u,d,v = np.linalg.svd(betas)
        betas = pd.concat([pd.DataFrame(d[k] * np.outer(u[:,k],v[k,:]),index=betas.index,columns=betas.columns) for k in range(rank_betas)],keys=range(rank_betas)).groupby(level=1).sum()
        betas *= np.sqrt(signal_to_noise * np.trace(cov_assets) / np.trace(betas.T @ cov_signals @ betas))
    
        cov_noise = cov_assets - betas.T @ cov_signals @ betas
        d,P = np.linalg.eigh(cov_noise)
        cov_noise = pd.DataFrame(P @ np.diag(np.maximum(d,1e-10)) @ P.T)

        sqrt_cov_assets = np.linalg.cholesky(cov_assets)
        sqrt_cov_signals = np.linalg.cholesky(cov_signals)
        sqrt_cov_noise = np.linalg.cholesky(cov_noise)
    
        signals = pd.DataFrame(np.random.randn(nb_dates_in_sample,nb_signals) @ sqrt_cov_signals.T)
        noise = pd.DataFrame(np.random.randn(nb_dates_in_sample,nb_assets) @ sqrt_cov_noise.T)
        assets = signals @ betas+noise
    
        signals_os = pd.DataFrame(np.random.randn(nb_dates_out_sample,nb_signals) @ sqrt_cov_signals.T)
        noise_os = pd.DataFrame(np.random.randn(nb_dates_out_sample,nb_assets) @ sqrt_cov_noise.T)
        assets_os = signals_os @ betas+noise_os

    
    
        #partie entrainement et test du modele 
        modele.fit(signals,assets)
        prediction = modele.predict(signals_os) #ce truc est le vecteur des pondérations w=Y'f(Y)
        pnl_predit=(prediction * assets_os).sum(axis=1)   #Xw = le vecteur des pnls, on obtient pnl quand on somme les rendements quotidiens 
        liste_sharpes_predit.append(pnl_predit.mean()/pnl_predit.std()*16)
    
        #question : a quoi comparer pnl_predit et sharpe_predit? 
        #à pnl optimal out sample ou in sample ? ie avec Xtrain ou avec Xtest ?
        pnl_optimal_os = ((signals_os @ betas)*assets_os).sum(axis=1)
        liste_sharpes_optimaux_os.append(pnl_optimal_os.mean()/pnl_optimal_os.std()*16)
        
        écart_sharpe_moyen.append(pnl_optimal_os.mean()/pnl_optimal_os.std()*16 - pnl_predit.mean()/pnl_predit.std()*16)
       
    
    dic = {'nb_simus': nb_simus,
       'moyenne sharpes prédits': np.mean(liste_sharpes_predit),
       'moyenne sharpe optimaux': np.mean(liste_sharpes_optimaux_os),
       'écart moyen sharpe': np.mean(écart_sharpe_moyen),
     }

    resultat = pd.DataFrame([dic])

    
    return resultat 
