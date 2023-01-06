import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import esda
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy as sp

import esda
import time


# Load data
anndat = ad.read_h5ad('../data/10XMultiome/10XPBMC.all.pseudo.h5ad')
peaks_nearby_all = anndat.uns['peaks_nearby'].copy()

from multiome.moran_vec import Moran
from libpysal.weights import W
import warnings
warnings.filterwarnings("ignore")

def subset(anndat_cp, n_obs, n_vars):
    # cells
    ind = np.arange(anndat_cp.n_obs)
    subind = np.random.choice(ind, n_obs, replace=False)
    anndat = anndat_cp[subind].copy()
    
    # vars
    ind = np.arange(anndat.n_vars)
    subind = np.random.choice(ind, n_vars, replace=False)
    anndat = anndat[:,subind].copy()
    
    return anndat

def run_sc(anndat):
    start = time.time()
    
    # w object
    cong_mtx = anndat.obsp['connectivities'].toarray()
    cong_mtx = cong_mtx/cong_mtx.sum(axis=1)[:,np.newaxis]
    neighbors = {}
    weights = {}
    for i in range(cong_mtx.shape[0]):
        neighbors[i] = np.nonzero(cong_mtx[i])[0].tolist()
        weights[i] = cong_mtx[i][np.nonzero(cong_mtx[i])[0]].tolist()
    w = W(neighbors, weights)

    mi = Moran(anndat.n_obs, w, permutations=0)
    mi.calc_i(anndat.X, seed=1)
    
    return time.time() - start

def run_esda(anndat):
    start = time.time()
    
    # w object
    cong_mtx = anndat.obsp['connectivities'].toarray()
    cong_mtx = cong_mtx/cong_mtx.sum(axis=1)[:,np.newaxis]
    neighbors = {}
    weights = {}
    for i in range(cong_mtx.shape[0]):
        neighbors[i] = np.nonzero(cong_mtx[i])[0].tolist()
        weights[i] = cong_mtx[i][np.nonzero(cong_mtx[i])[0]].tolist()
    w = W(neighbors, weights)
    for i in range(anndat.n_vars):
        X = anndat.X[:,i].flatten()
        mi = esda.moran.Moran(X, w)
    
    return time.time() - start


n_bulks = [100, 500, 1000, 2000, 5000]
n_features = [1, 10, 100, 500, 1000, 2000, 5000, 10000]
n_times = 8


time_df = pd.DataFrame(np.zeros((0,4)),
                       columns=['time','n_bulks','n_features','method'])

for n_obs in n_bulks:
    for n_var in n_features:
        for i in range(n_times):
            ad_sub = subset(anndat, n_obs, n_var)
            time_sc = run_sc(ad_sub)
            time_es = run_esda(ad_sub)
            time_df.loc[-1] = [time_sc, n_obs, n_var, 'sc']
            time_df.index = time_df.index + 1
            time_df.loc[-1] = [time_es, n_obs, n_var, 'esda']
            time_df.index = time_df.index + 1
            time_df.to_csv('time.moransi.csv',index=False)


