import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import esda

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

import multiome.lee_vec as lee_vec
import warnings
warnings.filterwarnings("ignore")

import multiome.lee_vec as lee_vec
import warnings
warnings.filterwarnings("ignore")

def subset(anndat_cp, n_obs, n_vars):
    # cells
    ind = np.arange(anndat_cp.n_obs)
    subind = np.random.choice(ind, n_obs, replace=False)
    anndat = anndat_cp[subind].copy()
    
    # pairs
    peaks_nearby = anndat.uns['peaks_nearby'].copy()
    ind = np.arange(peaks_nearby.shape[0])
    subind = np.random.choice(ind, n_vars, replace=False)
    peaks_nearby = peaks_nearby.iloc[subind]
    anndat.uns['peaks_nearby'] = peaks_nearby.copy()
    
    return anndat

def run_sc(anndat):
    start = time.time()
    
    cong_mtx = anndat.obsp['connectivities'].toarray()

    eSP = lee_vec.Spatial_Pearson_Local(cong_mtx, permutations=0)
    index_df = pd.DataFrame({'index':np.arange(anndat.n_vars)})
    index_df.index = anndat.var.index

    peaks_nearby = anndat.uns['peaks_nearby'].copy()

    gene = peaks_nearby['genes']
    peak = peaks_nearby['peaks']
    gene_index = index_df.loc[gene,'index'].to_numpy()
    peak_index = index_df.loc[peak,'index'].to_numpy()

    gene_X = anndat.X[:,gene_index]
    peak_X = anndat.X[:,peak_index]

    eSP = eSP.fit(gene_X, peak_X, seed=1)
    
    return time.time() - start

def run_esda(anndat):
    start = time.time()
    
    cong_mtx = anndat.obsp['connectivities'].toarray()
    index_df = pd.DataFrame({'index':np.arange(anndat.n_vars)})
    index_df.index = anndat.var.index

    peaks_nearby = anndat.uns['peaks_nearby'].copy()
    genes = peaks_nearby['genes'].tolist()
    peaks = peaks_nearby['peaks'].tolist()
    
    for i in range(peaks_nearby.shape[0]):
        gene_index = index_df.loc[genes[i],'index']
        peak_index = index_df.loc[peaks[i],'index']

        gene_X = anndat.X[:,gene_index].reshape(-1,1) #.flatten()
        peak_X = anndat.X[:,peak_index].reshape(-1,1) #.flatten()
        eSP = esda.Spatial_Pearson_Local(cong_mtx, permutations=0)
        eSP = eSP.fit(gene_X,peak_X)
    
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
            time_df.to_csv('time.localL.csv',index=False)






