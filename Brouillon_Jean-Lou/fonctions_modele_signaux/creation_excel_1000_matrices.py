import pandas as pd
import numpy as np

#cette fonction créé une liste de 20 dictionnaires, chaque dictionnaire contient 1000 x 4 matrices
#(signal train, signal test, asets train, assets test) et chaque dictionnaire est associé à une valeur
#de corrélation différente 


#Liste_dictionnaires = 

#[dictionnaire_correl_0.1, dictionnaire_correl_0.15, ... dictionnaire_correl_1]

#où dictionnaire_correl_0.1 = {"train_signal" : [liste contenant les 1000 matrices de signal pour le train],
#                              "train_assets": [liste contenant les 1000 matrices d'assets pour le train],
#                              "test_signal":  [liste contenant les 1000 matrices de signal pour le test],
#                              "test_assets": [liste contenant les 1000 matrices d'assets pour le test]}

#chaque dictionnaire contient 4000 matrices, et liste_dictionnaire contient 20x4000 = 80 000 matrices


seeds=np.random.randint(0, 1000000, size=1000)

def creer_excel_matrices_correl_varie(min_abscisse,max_abscisse,pas,nb_simus,nb_signals,
                                      nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas):

    abscisse = np.arange(min_abscisse,max_abscisse,pas)

    cov_assets = vol_assets**2 * pd.DataFrame(correl_assets * np.ones((nb_assets,nb_assets)) + (1-correl_assets)*np.eye(nb_assets))
    cov_signals = vol_signals**2 * pd.DataFrame(correl_signals * np.ones((nb_signals,nb_signals)) + (1-correl_signals)*np.eye(nb_signals))

    liste_dictionnaires = []

    for i in abscisse: 
        data = f"matrices_pour_correl_{i}"

        data = {
                    "train_signal":[],
                    "train_assets":[],
                    "test_signals":[],
                    "test_assets" :[] 
                }

        

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

            sqrt_cov_signals = np.linalg.cholesky(cov_signals)
            sqrt_cov_noise = np.linalg.cholesky(cov_noise)
    
            signals = pd.DataFrame(np.random.randn(nb_dates_in_sample,nb_signals) @ sqrt_cov_signals.T)
            noise = pd.DataFrame(np.random.randn(nb_dates_in_sample,nb_assets) @ sqrt_cov_noise.T)
            assets = signals @ betas+noise
    
            signals_os = pd.DataFrame(np.random.randn(nb_dates_out_sample,nb_signals) @ sqrt_cov_signals.T)
            noise_os = pd.DataFrame(np.random.randn(nb_dates_out_sample,nb_assets) @ sqrt_cov_noise.T)
            assets_os = signals_os @ betas+noise_os

            data["train_signal"].append(signals)
            data["train_assets"].append(assets)
            data["test_signals"].append(signals_os)
            data["test_assets"].append(assets_os)
    
        liste_dictionnaires.append(data)

    return liste_dictionnaires



    

        


    
    