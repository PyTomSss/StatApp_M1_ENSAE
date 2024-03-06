import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
import os 
#matplotlib inline



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso


from prediction_sharpe_py import prediction_sharpe



def pour_faire_jolis_graphiques(modele,parametre_en_abscisse, min_abscisse, max_abscisse,pas, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas):

    
    start_time = time.time()

    abscisse = np.arange(min_abscisse,max_abscisse,pas) #abscisse
    sharpe_os = []
    sharpe_theo = []
    #ecart_sharpe = []

    
    if parametre_en_abscisse == 'nombre_signaux' : 
        
        for i in abscisse : 

            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, i,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,i,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,i,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
             #         correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,3])


    elif parametre_en_abscisse == 'nombre_dates_in_sample' : 

        for i in abscisse:

            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,i,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,i,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,i,nb_dates_out_sample,nb_assets,vol_assets,
             #         correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,3])


    elif parametre_en_abscisse == 'nombre_dates_out_sample' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,i,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,i,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,i,nb_assets,vol_assets,
             #         correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,3])


    elif parametre_en_abscisse == 'nombre_actifs' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,nb_dates_out_sample,i,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,i,vol_assets,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,i,vol_assets,
             #         correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,3])


    elif parametre_en_abscisse == 'vol_actifs' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,i,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,i,
                      correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,i,
             #         correl_assets,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,3])


    elif parametre_en_abscisse == 'correl_actifs' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      i,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      i,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
             #         i,vol_signals,correl_signals,signal_to_noise,rank_betas).iloc[0,3])



    elif parametre_en_abscisse == 'vol_signaux' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,i,correl_signals,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,i,correl_signals,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
             #         correl_assets,i,correl_signals,signal_to_noise,rank_betas).iloc[0,3])



    elif parametre_en_abscisse == 'correl_signaux' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,i,signal_to_noise,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,i,signal_to_noise,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
             #         correl_assets,vol_signals,i,signal_to_noise,rank_betas).iloc[0,3])



    elif parametre_en_abscisse == 'signal_sur_bruit' : 

        for i in abscisse:
            print(i)
            sharpe_os.append(prediction_sharpe(modele, nb_simus, nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,i,rank_betas).iloc[0,1])
            sharpe_theo.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
                      correl_assets,vol_signals,correl_signals,i,rank_betas).iloc[0,2])
            #ecart_sharpe.append(prediction_sharpe(modele, nb_simus,nb_signals,nb_dates_in_sample,nb_dates_out_sample,nb_assets,vol_assets,
             #         correl_assets,vol_signals,correl_signals,i,rank_betas).iloc[0,3])



    else: 

        return ("pas un paramètre à faire varier fdp")



    end_time = time.time()
    execution_time = end_time - start_time
    print("Le temps d'exécution est de {:.2f} secondes.".format(execution_time))

    return abscisse, sharpe_os, sharpe_theo

    