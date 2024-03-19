#On importe les packages ici : 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#Cette fonction permet de générer une matrice de rendements suivant une loi normale multivariée
#Elle prend en paramètres le nombre d'actifs souhaités, le nombre de dates, les paramètres de la loi normale souhaitée
def rdt_matrix_generator(nb_actifs, nb_dates, list_esp_rdts, list_vols, correl, rand_vector = None) :     
    #Définition des lois des actifs
    idx = pd.Index(np.arange(nb_actifs))
    idx_date = pd.Index(np.arange(nb_dates))
    mu = np.zeros(nb_actifs)
    nb = len(list_esp_rdts)
    vol = np.zeros(nb_actifs)

    for i in range(nb):
        #On split le vecteur des espérances en nb blocs ayant chacun une espérance commune
        valeur_commune_mu = list_esp_rdts[i]
        debut_bloc = i * (nb_actifs // nb)
        fin_bloc = (i + 1) * (nb_actifs // nb)
        mu[debut_bloc:fin_bloc] = valeur_commune_mu
    
    for i in range(nb):
        #On split le vecteur des volatilités en nb blocs ayant chacun une volatilité commune
        valeur_commune_vol = list_vols[i]
        debut_bloc = i * (nb_actifs // nb)
        fin_bloc = (i + 1) * (nb_actifs // nb)
        vol[debut_bloc:fin_bloc] = valeur_commune_vol

    #Pour l'instant la corrélation entre les actifs est la même, on pourra jouer sur la matrice des corrélations plus tard

    #Matrice var-covar théorique selon la vraie loi des rendements
    covar = np.diag(vol) @ pd.DataFrame(correl * np.ones((nb_actifs,nb_actifs)) + (1-correl) * np.eye(nb_actifs),index=idx,columns=idx) @ np.diag(vol)

    #On simuler un tirage de rendements suivant une loi normale d'espérence Mu et de Var la matrice Covar
    A = np.linalg.cholesky(covar)

    #Cette boucle nous permet, si l'on souhaite de réutiliser un vecteur de rendements aléatoires déjà généré -> aléa fixé
    if rand_vector is None: 
        rdts_observes = mu/256 + pd.DataFrame(np.random.randn(nb_dates,nb_actifs)) @ A.T / 16
    else:
        rdts_observes = mu/256 + pd.DataFrame(rand_vector) @ A.T / 16

    return rdts_observes

#Cette fonction permet de fournir un sharpe ratio moyen in & out sample pour une matrice de test donnée et un modèle donné
def pred_sharpe(rdt_matrix, linear_model) : 
    pnl = rdt_matrix @ linear_model
    sharpe = pnl.mean() / pnl.std() * 16
    return sharpe

#param_var = 1 : on fait varier le nombre d'actifs
#param_var = 2 : on fait varier le nombre de dates
#param_var = 3 : on fait varier la corrélation
#param_var = 4 : on fait varier la volatilité
#param_var = 5 : on fait varier l'espérance
#param_var = 6 : on fait varier le nombre de blocs au sein du vecteur d'actifs
#param_var = 7 : on fait varier le ratio lignes/colonnes (nb_dates/nb_actifs)

#Plot les graphes utiles pour nous
def perf_model_plot(K, param_var, vect_in, vect_out, linear_model) :
     if param_var == 1 : 
        x = np.arange(2,100, 2) #nb actifs étant manuellement fixé à 100

        #On plot l'évolution des sharpes in & out en fonction de l'évolution du paramètre d'intérêt
        plt.figure(figsize=(10, 8))
        plt.plot(x, vect_out, label = 'Sharpe Out Sample Modèle', color = 'blue')
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
        plt.ylabel('Sharpe Ratio moyen pour 1000 simulations')


#Cette fonction va tester le modèle linéaire sur K matrices de train et effectuer une sorte de CV pour avoir le sharpe Out of Sample le plus fiable
#K := nombre de modèles à tester => nombres de matrices de test par modèles (K-1)*10 (car 10 ans d'historique)
#Cela doit return un vecteur de sharpe ratio in & out sample moyen pour chaque pas de variation du param

def test_linear_model(param_var, linear_model, K = 30) : 
    #Pour l'instant on fixe manuellement le nombre max d'actifs dans le modèle (100) et le fait qu'on train sur 10 ans
    nb_dates = 256*10
    nb_actifs = 100
    if param_var == 1 : 
        #Génération des vecteurs d'aléas, gardés en mémoire
        list_rand_vect = np.array()
        for i in range(K) : 
            list_rand_vect.append(np.random.randn(nb_dates, nb_actifs)) 
        list_rand_vect = pd.Dataframe(list_rand_vect)
        
        #Génération des matrices de rendements -> 1 fois pour toutes
        rdts = np.array()
        for i in range(K) : 
            rdts.append(rdt_matrix_generator(i,nb_dates,[0.05], [0.2], 0.7, list_rand_vect[i])) #On fixe manuellement les paramètres de la loi normale et on garde en mémoire les matrices de train
        
        #itérations sur le nombre d'actifs
        for i in np.arange(2,nb_actifs, 2) : 
            idx_date = pd.Index(np.arange(nb_dates))
            Y_train = pd.Series(1, index = idx_date)                    #On fixe arbitrairement le vecteur de rendements du portefeuille égalitaire
            sharpe_out_vector = []
            sharpe_in_vector = []

            #Entraînement et test des modèles
            for j in range(K) :
                #On fit le modèle
                model = linear_model.fit(rdts[j][:i], Y_train)          #On fit le modèle sur les i premiers actifs et le j-ème vecteur de rendements
                model = model.coef_

                #On teste le modèle
                sharpe_out = 0
                sharpe_in = 0
                for k in range(K) : 
                    if k != j :
                        sharpe_out += pred_sharpe(rdts[k][i:], model)
                    else : 
                        sharpe_in = pred_sharpe(rdts[k][:i], model)
                sharpe_out = sharpe_out / K                             #Sharpe moyen out-sample sur les K-1 matrices de test

            sharpe_out_vector.append(sharpe_out)
            sharpe_in_vector.append(sharpe_in)

        #On plot les graphes
        perf_model_plot(K, param_var, sharpe_in_vector, sharpe_out_vector, linear_model)