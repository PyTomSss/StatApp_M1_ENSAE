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