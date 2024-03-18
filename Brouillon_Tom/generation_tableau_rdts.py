import pandas as pd
import matplotlib.pyplot as plt

def generation_tableau_rdts(nb_actifs, nb_dates, list_esp_rdts, list_vols, nb_simu, nb_dates_test, correl, model) :     
    #On crée une fonction à qui on donne les paramètres que l'on souhaite pour la simulation de la matrice de rendements, 
    #le nombre de simulations pour le test out-sample et le modèle que l'on souhaite tester. 
    #La fonction va renvoyer le graphique des quantiles des sharpe ratios out-samples et la moyenne des sharpe ratios in & out sample.
    
    #Définition des lois des actifs
    idx = pd.Index(np.arange(nb_actifs))
    idx_date = pd.Index(np.arange(nb_dates))
    mu = np.zeros(nb_actifs)
    nb = len(list_esp_rdts)
    vol = np.zeros(nb_actifs)
    correl = correl

    for i in range(nb):
        valeur_commune_mu = list_esp_rdts[i]
        debut_bloc = i * (nb_actifs // nb)
        fin_bloc = (i + 1) * (nb_actifs // nb)
        mu[debut_bloc:fin_bloc] = valeur_commune_mu
    
    for i in range(nb):
        valeur_commune_vol = list_vols[i]
        debut_bloc = i * (nb_actifs // nb)
        fin_bloc = (i + 1) * (nb_actifs // nb)
        vol[debut_bloc:fin_bloc] = valeur_commune_vol


    #Matrice var-covar théorique selon la vraie loi des rendements
    covar = np.diag(vol) @ pd.DataFrame(correl * np.ones((nb_actifs,nb_actifs)) + (1-correl) * np.eye(nb_actifs),index=idx,columns=idx) @ np.diag(vol)

    #On simuler un tirage de rendements suivant une loi normale d'espérence Mu et de Var la matrice Covar
    A = np.linalg.cholesky(covar)
    rdts_observes = mu/256 + pd.DataFrame(np.random.randn(nb_dates,nb_actifs)) @ A.T / 16
    Y_train = pd.Series(1, index = idx_date)
    #print(rdts_observes)

    model_ =  model.fit(rdts_observes, Y_train)
    coeffs = model_.coef_
    
    #On va sortir le graphique avec les sharpe moyens 
    pnl_out_sample = []
    sharpe_out_sample = []

    #Ici, j'ai tout remplacé par les coefficients non normalisés afin de voir si cela réduit la variance des résultats
    for i in range(nb_simu) : 
        simu = mu/256 + pd.DataFrame(np.random.randn(nb_dates_test,nb_actifs)) @ A.T / 16
        pnl_simu = simu @ coeffs
        pnl_out_sample.append(pnl_simu)

        sharpe_simu = pnl_simu.mean() / pnl_simu.std() * 16
        sharpe_out_sample.append(sharpe_simu)

    sharpe_out_sample_DF = pd.DataFrame(sharpe_out_sample)
    #sharpe_out_sample_DF.describe()
    #print(sharpe_out_sample)

    quantiles = sharpe_out_sample_DF.quantile(np.arange(0.01,1.0,0.01)).reset_index().rename({'index':'proba',0:'quantile'},axis=1)
    plt.scatter(quantiles['quantile'],quantiles['proba'])

    moyenne_sharpe = float(sharpe_out_sample_DF.mean())
    plt.axvline(x=moyenne_sharpe, color='orange', linestyle='-', linewidth=5,  label='Moyenne des Sharpes')

    plt.xlabel('Sharpes Ratios')
    plt.ylabel('Quantile associé')
    plt.legend()
    print(f"Voici le sharpe moyen de cette méthode {int(moyenne_sharpe*1000)/1000}")

    return f"Voici le portfolio optimal selon cette méthode : {coeffs}"