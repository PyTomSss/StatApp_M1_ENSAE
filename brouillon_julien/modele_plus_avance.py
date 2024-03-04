import numpy as np
import pandas as pd

def retourner_matrices(nb_signals = 100, nb_dates_in_sample = 1300, nb_dates_out_sample = 260,
    nb_assets = 80, vol_assets = 0.2/16, correl_assets = 0.8):

    vol_signals = 0.2/16
    correl_signals = 0.8

    signal_to_noise = 0.001

    rank_betas = 5

    cov_assets = vol_assets**2 * pd.DataFrame(correl_assets * np.ones((nb_assets,nb_assets)) + (1-correl_assets)*np.eye(nb_assets))
    cov_signals = vol_signals**2 * pd.DataFrame(correl_signals * np.ones((nb_signals,nb_signals)) + (1-correl_signals)*np.eye(nb_signals))

    betas = pd.DataFrame(np.random.randn(nb_signals,nb_assets))
    betas *= np.sqrt(signal_to_noise * np.trace(cov_assets) / np.trace(betas.T @ cov_signals @ betas))
    u,d,v = np.linalg.svd(betas)
    betas = pd.concat([pd.DataFrame(d[k] * np.outer(u[:,k],v[k,:]),index=betas.index,columns=betas.columns) for k in range(rank_betas)],keys=range(rank_betas)).groupby(level=1).sum()
    betas *= np.sqrt(signal_to_noise * np.trace(cov_assets) / np.trace(betas.T @ cov_signals @ betas))
    cov_noise = cov_assets - betas.T @ cov_signals @ betas
    d,P = np.linalg.eigh(cov_noise)
    cov_noise = pd.DataFrame(P @ np.diag(np.maximum(d,1e-10)) @ P.T)

    sqrt_cov_assets = np.linalg.cholesky(cov_assets)
    sqrt_cov_signals = np.linalg.cholesky(cov_signals)
    sqrt_cov_noise = np.linalg.cholesky(cov_noise)

    signals = pd.DataFrame(np.random.randn(nb_dates_in_sample,nb_signals) @ sqrt_cov_signals.T)
    noise = pd.DataFrame(np.random.randn(nb_dates_in_sample,nb_assets) @ sqrt_cov_noise.T)
    assets = signals @ betas+noise

    signals_os = pd.DataFrame(np.random.randn(nb_dates_out_sample,nb_signals) @ sqrt_cov_signals.T)
    noise_os = pd.DataFrame(np.random.randn(nb_dates_out_sample,nb_assets) @ sqrt_cov_noise.T)
    assets_os = signals_os @ betas+noise_os

    return [signals, assets], [signals_os, assets_os]

matrices_in_sample, matrices_out_sample = retourner_matrices()
print("Matrices In Sample:")
print(matrices_in_sample)
print("Matrices Out Sample:")
print(matrices_out_sample)

