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
    #print('signal to noise = ' + str(np.round(np.trace(betas.T @ cov_signals @ betas) / np.trace(cov_assets),3))) #Vérification retirable

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
    
    w_model = signals_matrix @ linear_model.T #On prend la transposée car la matrice de betas est de la forme (nb_actifs, nb_signals) et on veut (nb_signals, nb_actifs)
    pnl_model = (w_model * assets_matrix).sum(axis=1) #Pourquoi la sum et pas juste la multiplication ?
    sharpe = pnl_model.mean() / pnl_model.std()*16
    
    return sharpe

#param_var = 1 : on fait varier le nombre de signaux
#param_var = 2 : on fait varier le nombre de dates
#param_var = 3 : on fait varier le nombre d'actifs
#param_var = 4 : on fait varier le signal to noise (qui représente ?)
#param_var = 5 : on fait varier la variance des signaux
#param_var = 6 : on fait varier la corrélation des signaux
#param_var = 7 : on fait varier la volatilité des actifs
#param_var = 8 : on fait varier la corrélation des actifs
#On ne fait varier que les paramètres relatifs à la taille des matrices et de loi des signaux, pas pour les actifs (serait-ce pertinent ?)

#MODIFIER LES PAS APRÈS AVOIR TESTÉ LES FONCTIONS
#Plot les graphes utiles pour nous
def perf_modelsignals_plot(param_var, param_var_max, vect_in, vect_out, linear_model, zoom_out = False) :
    
    if param_var == 1 : 

        x = np.arange(2,param_var_max, 2) 

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction du nombre de signaux {str(linear_model)}")
        plt.xlabel("Nombre de signaux")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction du nombre de signaux {str(linear_model)}")
        plt.xlabel("Nombre de signaux")
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

    elif param_var == 4 : 

        x = np.arange(0.0005, param_var_max, 0.0005) #à tester car pas l'ordre de grandeur (celui de P est de 0.001)

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction du ratio Signal/Bruit {str(linear_model)}")
        plt.xlabel("Ratio Signal/Bruit")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction du ratio Signal/Bruit {str(linear_model)}")
        plt.xlabel("Ratio Signal/Bruit")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 5 : 

        x = np.arange(0.01, param_var_max, 0.005) 

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        if zoom_out == False : 
            plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la volatilité des signaux {str(linear_model)}")
        plt.xlabel("Volatilité des signaux (égale pour tous)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la volatilité des signaux {str(linear_model)}")
        plt.xlabel("Volatilité des signaux (égale pour tous)")
        plt.ylabel('Sharpe Ratio moyen')
    
    elif param_var == 6 : 

        x = np.arange(0,param_var_max, 0.005) 

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la corrélation entre les signaux {str(linear_model)}")
        plt.xlabel("Corrélations entre les signaux (même pour tous les signaux)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la corrélation entre les signaux {str(linear_model)}")
        plt.xlabel("Corrélations entre les signaux (même pour tous les signaux)")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 7 : 

        x = np.arange(0.01, param_var_max, 0.005)

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
        plt.plot(x, vect_in, label = 'Sharpe In Sample Modèle', color = 'green')
        plt.legend()

        plt.title(f"Évolution des sharpes ratios moyens en fonction de la volatilité des actifs {str(linear_model)}")
        plt.xlabel("Volatilité des actifs (égale pour tous)")
        plt.ylabel('Sharpe Ratio moyen')

        #ici on plot l'évolution du ration in/out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, np.array(vect_out) / np.array(vect_in), label = 'Ratio Sharpe In & Out Sample', color = 'blue')
        plt.legend()

        plt.title(f"Évolution de l'overfitting en fonction de la volatilité des actifs {str(linear_model)}")
        plt.xlabel("Volatilité des actifs (égale pour tous)")
        plt.ylabel('Sharpe Ratio moyen')

    elif param_var == 8 : 

        x = np.arange(0,param_var_max, 0.005) 

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

    #POur l'instant, on ne peut pas avoir plusieurs lois différentes pour les signaux et les actifs
    #Fonction de test des modèles linéaires sur les signaux
def test_linear_model_signals(param_var, param_var_max, linear_model, zoom_out = False, K = 30) : 

    if param_var == 1 : 
        #Pour l'instant on fixe manuellement la taille de l'historique (10ans) et le nb actifs (20) et le nombre max de signaux est donné par l'utilisateur
        nb_actifs = 20
        nb_signals = param_var_max
        nb_dates = 256*10
        signals_vol = 0.2/16 #fixé comme P
        correl_signals = 0.8
        signal_to_noise = 0.001
        vol_assets = 0.2/16
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
        signaux = []
        for i in range(K) : 
            tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, correl_signals, nb_actifs, vol_assets, correl_assets, 
                                    signal_to_noise, 5, list_rand_vect_signals[i], list_rand_vect_betas[i], list_rand_vect_noise[i])
            signaux.append(tabs[0]) 
            rdts.append(tabs[1])
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(2,nb_signals, 2)  : 

            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]
                train = train.iloc[:, :i]                         
                #On fit le modèle
                model = linear_model.fit(train, rdts[j])          #On fit le modèle sur les i premiers signaux et le j-ème vecteur de signaux (labels -> j-ème vecteur de rdts)
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    train = train.iloc[:, :i]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts[k], train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts[k], train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)

    elif param_var == 2 : 
        #Pour l'instant on fixe manuellement la taille de l'historique (10ans) et le nb actifs (20) et le nombre max de signaux est donné par l'utilisateur
        nb_actifs = 20
        nb_signals = 100
        nb_dates = param_var_max 
        signals_vol = 0.2/16 #fixé comme P
        correl_signals = 0.8
        signal_to_noise = 0.001
        vol_assets = 0.2/16
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
        signaux = []
        for i in range(K) : 
            tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, correl_signals, nb_actifs, vol_assets, correl_assets, 
                                    signal_to_noise, 5, list_rand_vect_signals[i], list_rand_vect_betas[i], list_rand_vect_noise[i])
            signaux.append(tabs[0]) 
            rdts.append(tabs[1])
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(256,param_var_max, 256/2) : 
            i = int(i)
            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]
                train = train.iloc[:i, :]                         #On prend les i premières dates 
                rdts_TRAIN = rdts[j]
                rdts_TRAIN = rdts_TRAIN.iloc[:i, :]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les i premiers signaux et le j-ème vecteur de signaux (labels -> j-ème vecteur de rdts)
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    train = train.iloc[:i, :]
                    rdts_TEST = rdts[k]
                    rdts_TEST = rdts_TEST.iloc[:i, :]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)

    elif param_var == 3 : 

        nb_actifs = param_var_max
        nb_signals = 100
        nb_dates = 256 * 10 
        signals_vol = 0.2/16 #fixé comme P
        correl_signals = 0.8
        signal_to_noise = 0.001
        vol_assets = 0.2/16
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
        signaux = []
        for i in range(K) : 
            tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, correl_signals, nb_actifs, vol_assets, correl_assets, 
                                    signal_to_noise, 5, list_rand_vect_signals[i], list_rand_vect_betas[i], list_rand_vect_noise[i])
            signaux.append(tabs[0]) 
            rdts.append(tabs[1])
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(2,param_var_max, 2) : 
            sharpe_out_models = 0
            sharpe_in_models = 0

            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]                                  #Toujours la même matrice de signaux
                rdts_TRAIN = rdts[j]
                rdts_TRAIN = rdts_TRAIN.iloc[:, :i]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les signaux et les i premiers actifs
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    rdts_TEST = rdts[k]
                    rdts_TEST = rdts_TEST.iloc[:, :i]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)
    
    elif param_var == 4 : 

        nb_actifs = 20
        nb_signals = 100
        nb_dates = 256 * 10 
        signals_vol = 0.2/16 #fixé comme P
        correl_signals = 0.8
        signal_to_noise = param_var_max
        vol_assets = 0.2/16
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        #ON NE PEUT PAS TROP LE FAIRE A MOINS DE DIFFERENCIER LA CREATION DES SIGNAUX ET DES BETAS -> A REFLECHIR
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(0.0005, param_var_max, 0.0005) : 

            rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
            signaux = []
            for j in range(K) : 
                tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, correl_signals, nb_actifs, vol_assets, correl_assets, 
                                                i, 5, list_rand_vect_signals[j], list_rand_vect_betas[j], list_rand_vect_noise[j])
                signaux.append(tabs[0]) 
                rdts.append(tabs[1])

            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]                                  #Toujours la même matrice de signaux
                rdts_TRAIN = rdts[j]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les signaux et les i premiers actifs
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    rdts_TEST = rdts[k]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)

    elif param_var == 5 : 

        nb_actifs = 20
        nb_signals = 100
        nb_dates = 256 * 10 
        signals_vol = param_var_max #fixé comme P
        correl_signals = 0.8
        signal_to_noise = 0.001
        vol_assets = 0.2 / 16
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        #ON NE PEUT PAS TROP LE FAIRE A MOINS DE DIFFERENCIER LA CREATION DES SIGNAUX ET DES BETAS -> A REFLECHIR
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(0.01, param_var_max, 0.005) : 

            rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
            signaux = []
            for j in range(K) : 
                tabs = random_signals_generator(nb_signals, nb_dates, i, correl_signals, nb_actifs, vol_assets, correl_assets, 
                                                signal_to_noise, 5, list_rand_vect_signals[j], list_rand_vect_betas[j], list_rand_vect_noise[j])
                signaux.append(tabs[0]) 
                rdts.append(tabs[1])

            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]                                  #Toujours la même matrice de signaux
                rdts_TRAIN = rdts[j]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les signaux et les i premiers actifs
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    rdts_TEST = rdts[k]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)
    
    elif param_var == 6 : 

        nb_actifs = 20
        nb_signals = 100
        nb_dates = 256 * 10 
        signals_vol = 0.2/16 #fixé comme P
        correl_signals = param_var_max
        signal_to_noise = 0.001
        vol_assets = 0.2 / 16
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        #ON NE PEUT PAS TROP LE FAIRE A MOINS DE DIFFERENCIER LA CREATION DES SIGNAUX ET DES BETAS -> A REFLECHIR
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(0,param_var_max, 0.005) : 

            rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
            signaux = []
            for j in range(K) : 
                tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, i, nb_actifs, vol_assets, correl_assets, 
                                                signal_to_noise, 5, list_rand_vect_signals[j], list_rand_vect_betas[j], list_rand_vect_noise[j])
                signaux.append(tabs[0]) 
                rdts.append(tabs[1])

            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]                                  #Toujours la même matrice de signaux
                rdts_TRAIN = rdts[j]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les signaux et les i premiers actifs
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    rdts_TEST = rdts[k]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)

    elif param_var == 7 : 

        nb_actifs = 20
        nb_signals = 100
        nb_dates = 256 * 10 
        signals_vol = 0.2 / 16 #fixé comme P
        correl_signals = 0.8
        signal_to_noise = 0.001
        vol_assets = param_var_max
        correl_assets = 0.8

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        #ON NE PEUT PAS TROP LE FAIRE A MOINS DE DIFFERENCIER LA CREATION DES SIGNAUX ET DES BETAS -> A REFLECHIR
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(0.01, param_var_max, 0.005) : 

            rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
            signaux = []
            for j in range(K) : 
                tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, correl_signals, nb_actifs, i, correl_assets, 
                                                signal_to_noise, 5, list_rand_vect_signals[j], list_rand_vect_betas[j], list_rand_vect_noise[j])
                signaux.append(tabs[0]) 
                rdts.append(tabs[1])

            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]                                  #Toujours la même matrice de signaux
                rdts_TRAIN = rdts[j]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les signaux et les i premiers actifs
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    rdts_TEST = rdts[k]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)

    elif param_var == 8 : 

        nb_actifs = 20
        nb_signals = 100
        nb_dates = 256 * 10 
        signals_vol = 0.2 / 16 #fixé comme P
        correl_signals = 0.8
        signal_to_noise = 0.001
        vol_assets = 0.2 / 16
        correl_assets = param_var_max

        #Génération des vecteurs d'aléas, gardés en mémoire -> 1 pour les signaux, 1 pour le noise, 1 pour les betas
        list_rand_vect_signals = []
        list_rand_vect_betas = []
        list_rand_vect_noise = []
        for i in range(K) : 
            list_rand_vect_signals.append(np.random.randn(nb_dates, nb_signals)) 
            list_rand_vect_betas.append(np.random.randn(nb_signals, nb_actifs)) 
            list_rand_vect_noise.append(np.random.randn(nb_dates, nb_actifs))   
            
        #Génération des matrices de rendements -> 1 fois pour toutes
        #ON NE PEUT PAS TROP LE FAIRE A MOINS DE DIFFERENCIER LA CREATION DES SIGNAUX ET DES BETAS -> A REFLECHIR
            
        sharpe_out_vector = []
        sharpe_in_vector = []
        #itérations sur le nombre de signaux
        for i in np.arange(0,param_var_max, 0.005) : 

            rdts = [] #On va tout stocker dans une liste et voir si cela fonctionne bien en terme de mémoire
            signaux = []
            for j in range(K) : 
                tabs = random_signals_generator(nb_signals, nb_dates, signals_vol, correl_signals, nb_actifs, vol_assets, i, 
                                                signal_to_noise, 5, list_rand_vect_signals[j], list_rand_vect_betas[j], list_rand_vect_noise[j])
                signaux.append(tabs[0]) 
                rdts.append(tabs[1])

            sharpe_out_models = 0
            sharpe_in_models = 0
            #Entraînement et test des modèles
            for j in range(K) :
                train = signaux[j]                                  #Toujours la même matrice de signaux
                rdts_TRAIN = rdts[j]                   
                #On fit le modèle
                model = linear_model.fit(train, rdts_TRAIN)          #On fit le modèle sur les signaux et les i premiers actifs
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    train = signaux[k]
                    rdts_TEST = rdts[k]
                    if k != j :
                        sharpe_out += pred_sharpe_signals(rdts_TEST, train, model)
                    else : 
                        sharpe_in = pred_sharpe_signals(rdts_TEST, train, model)
                sharpe_out = sharpe_out / (K-1)                     #Sharpe moyen out-sample DU MODÈlE testé sur les K-1 matrices de test
                sharpe_out_models += sharpe_out 
                sharpe_in_models += sharpe_in                    
            sharpe_out_vector.append(sharpe_out_models / K)         #Sharpe moyen sur les K modèles 
            sharpe_in_vector.append(sharpe_in_models / K)

        #On plot les graphes
        perf_modelsignals_plot(param_var,param_var_max, sharpe_in_vector, sharpe_out_vector, linear_model, zoom_out = zoom_out)




        