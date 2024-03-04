from print_matrices import retourner_matrices
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

matrices_in_sample, matrices_out_sample = retourner_matrices()
signals_is, assets_is = matrices_in_sample[0], matrices_in_sample[1]

def estimer_beta(signals, assets):
    lr = LinearRegression()
    lr.fit(signals, assets)
    return lr.coef_

estimated_beta = estimer_beta(signals_is, assets_is)

