import pandas as pd
import numpy as np

#Modelisation
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

def test_model(X_train, X_test, y_train, y_test, DF, model) : 

    #on run le modèle avec les fonctions prebuilt
    ols =  model.fit(X_train, y_train)
    y_pred = ols.predict(X_test)


    #On affiche le R^2 du modèle calculé à partir des échantillons Train
    print("R_2 du modèle : " +str(ols.score(X_test, y_test)))
    print(f"Le MSE (Mean Squared Error) de notre prédiction est de {mean_squared_error(y_pred, y_test)}")
    print(f"Le MAE (Mean Absolute Error) de notre prédiction est de {mean_absolute_error(y_pred, y_test)}")

    #On affiche le résultat sous forme de graphique
    fig, axs = plt.subplots( figsize=(8, 8))
    plt.scatter(y_pred, y_test, color='blue', label='Valeurs Réelles vs. Prédites')

    #On va voir si la droite des moindres carrés entre les valeurs prédites et les vraies valeurs est proche de la droite x=y
    reg = LinearRegression()
    reg.fit(y_test, y_pred)

    slope = reg.coef_[0]
    intercept = reg.intercept_
    #print(f"Pente : {slope[0]}")
    #print(f"Intercept: {intercept[0]}")

    #On plot le tout sur le même graphique
    plt.plot(y_test, reg.predict(y_test), color='red',linewidth = 3, label='Régression Linéaire')
    plt.plot([x for x in range(100)], color='g', marker = ".", label="y=x")
    plt.title('Modèle de régression multivariée testée sur échantillon test')
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Valeurs Réelles')
    plt.legend()
    plt.show()


    coefs = pd.DataFrame(ols.coef_, columns=['Coefficients'], index = DF.columns)
    coefs.sort_values(by='Coefficients', ascending=True, inplace=True)
    coefs.plot(kind='barh', figsize=(15, 20), color='royalblue')
    plt.title('Régression linéaire')
    plt.axvline(x=0, color='1')
    plt.grid()
    plt.subplots_adjust(left=.3)