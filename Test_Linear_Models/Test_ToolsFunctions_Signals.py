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

#param_var = 1 : on fait varier le nombre de signaux
#param_var = 2 : on fait varier le nombre de dates
#param_var = 3 : on fait varier le nombre d'actifs
#param_var = 4 : on fait varier le signal to noise (qui représente ?)
#param_var = 5 : on fait varier la variance des signaux
#param_var = 6 : on fait varier la corrélation des signaux
#On ne fait varier que les paramètres relatifs à la taille des matrices et de loi des signaux, pas pour les actifs (serait-ce pertinent ?)

#Plot les graphes utiles pour nous
def perf_modelsignals_plot(K, param_var, param_var_max, vect_in, vect_out, linear_model, zoom_out = False) :
    if param_var == 1 : 

        x = np.arange(2,param_var_max, 2) 

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction du nombre d'actifs {str(linear_model)}")
        plt.xlabel("Nombre d'actifs")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction du nombre d'actifs {str(linear_model)}")
        plt.xlabel("Nombre d'actifs")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 2 : 

        x = np.arange(256,param_var_max, 256/2) 
        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction du nombre de dates d'entraînement {str(linear_model)}")
        plt.xlabel("Nombre de dates")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction du nombre de dates d'entraînement {str(linear_model)}")
        plt.xlabel("Nombre de dates")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 3 : 

        x = np.arange(0,param_var_max, 0.01) 
        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la corrélation entre les actifs {str(linear_model)}")
        plt.xlabel("Corrélations entre les actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la corrélation entre les actifs {str(linear_model)}")
        plt.xlabel("Corrélations entre les actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 4 : 

        x = np.arange(0.1, param_var_max, 0.01) 
        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la volatilité des actifs {str(linear_model)}")
        plt.xlabel("Volatilité des actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la volatilité des actifs {str(linear_model)}")
        plt.xlabel("Volatilité des actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 5 : 

        x = np.arange(-0.3, param_var_max, 0.005) 
        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de l'espérance de rendement des actifs {str(linear_model)}")
        plt.xlabel("Espérance de rendement des actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de l'espérance de rendement des actifs {str(linear_model)}")
        plt.xlabel("Espérance de rendement des actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')
    #Non fait encore
    elif param_var == 6 : 

        x = np.arange(0,param_var_max, 0.01) 
        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la corrélation entre les actifs {str(linear_model)}")
        plt.xlabel("Corrélations entre les actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la corrélation entre les actifs {str(linear_model)}")
        plt.xlabel("Corrélations entre les actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')
    #Non fait encore
    elif param_var == 7 : 

        x = np.arange(0,param_var_max, 0.01) 
        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la corrélation entre les actifs {str(linear_model)}")
        plt.xlabel("Corrélations entre les actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la corrélation entre les actifs {str(linear_model)}")
        plt.xlabel("Corrélations entre les actifs (même pour tous les actifs)")
        plt.ylabel('Sharpe Ratio moyen')
