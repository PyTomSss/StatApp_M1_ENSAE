import numpy as np
import pandas as pd

def rendements():

    nb_actifs = 20
    nb_dates = 256 * 5

    idx = pd.Index(np.arange(nb_actifs))

    mu = pd.Series(0.05,index=idx)
    vols = pd.Series(0.2,index=idx)
    correl = 0.7

    covar = np.diag(vols) @ pd.DataFrame(correl * np.ones((nb_actifs,nb_actifs)) + (1-correl) * np.eye(nb_actifs),index=idx,columns=idx) @ np.diag(vols)
    A = np.linalg.cholesky(covar)
    rdts_observes = mu/256 + pd.DataFrame(np.random.randn(nb_dates,nb_actifs)) @ A.T / 16

    mu_estime = rdts_observes.mean() * 256
    covar_estimee = rdts_observes.cov() * 256
    alloc_optimale_in_sample = pd.DataFrame(np.linalg.inv(covar_estimee),index=idx,columns=idx) @ mu_estime

    return rdts_observes

