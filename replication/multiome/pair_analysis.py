import numpy as np
import pandas as pd
import anndata as ad
import random
import time
import multiome.graph_measure as graph_measure
import scanpy as sc


'''
def HighlyCorrelatedPair(anndat_multiome, permutations=999,
                         l_cutoff=0.25, p_cutoff=0.01, seed=1):

    random.seed(seed)
    np.random.seed(seed)

    print('Start selecting highly-correlated gene-peak pair in the graph...')
    start = time.time()

    peaks_nearby = anndat_multiome.uns['peaks_nearby'].copy()
    global_L_df = graph_measure.Global_L(anndat_multiome, peaks_nearby, permutations=0, seed=seed)
    mask = (np.abs(global_L_df['L']) >= l_cutoff).to_numpy()

    peaks_nearby_tmp = peaks_nearby.loc[mask,:].copy()

    ###### p_sim name changed !!!!!
    global_L_df_2 = graph_measure.Global_L(anndat_multiome, peaks_nearby_tmp, permutations=permutations, seed=seed)
    global_L_df.loc[mask, 'p_sim'] = global_L_df_2['p_sim']

    peaks_nearby['L'] = global_L_df['L'].to_numpy()
    peaks_nearby['p_sim'] = global_L_df['p_sim'].to_numpy()

    mask = (global_L_df['p_sim'] <= p_cutoff).to_numpy()
    peaks_nearby['Correlated'] = mask

    print('Finished in %.2fs, %s Gene-Peak pairs selected.'%(time.time()-start, int(np.sum(mask))))

    print("Start calculating local spatial pearson correlation for all selected pairs...")
    start = time.time()

    local_L_df, local_p_df= graph_measure.Local_L(anndat_multiome, peaks_nearby.loc[mask,:],
                                     permutations=0, seed=seed)

    print('Finished in %.2fs.'%(time.time()-start))

    anndat_multiome.uns['cluster.local_L_names'] = local_L_df.columns.to_numpy()
    L_mtx = local_L_df.to_numpy()
    GP_names = anndat_multiome.uns['cluster.local_L_names']
    Dropout_mtx = graph_measure.dropout_filter(anndat_multiome, GP_names)
    anndat_multiome.uns['cluster.local_L'] = L_mtx * Dropout_mtx


    print("Local Spatial Pearson L saved in AnnData oject obsm['cluster.local_L'].\nColumn names saved in uns['cluster.Local_L_names'].")

    anndat_multiome.uns['peaks_nearby'] = peaks_nearby

    return anndat_multiome
'''



def RegulatoryAnalysis(anndat_multiome, binary=True, weighted=False,
                       trajectory=True,
                       leiden_resol=0.5, root=None):

    print('Extracting [obs x G-P defined region] matrix...')
    ident_mtx = anndat_multiome.uns['Local_L']

    if binary:
        ident_mtx[ident_mtx>0] = 1.0
        ident_mtx[ident_mtx<0] = -1.0

    if weighted:
        gene_names = [gp.split('_')[0] for gp in GP_names]
        weights = anndat_multiome.var.loc[gene_names, 'variances_norm'].to_numpy()
        ident_mtx = ident_mtx * weights

    print("Define coarse groups of the cluster for partition-based graph abstraction (PAGA)...")
    anndat_L = ad.AnnData(
        X = ident_mtx
    )
    #sc.tl.pca(anndat, svd_solver='randomized', use_highly_variable=False)
    sc.pp.neighbors(anndat_L, use_rep='X', n_neighbors=20)
    sc.tl.leiden(anndat_L, resolution=leiden_resol)

    n_clus = len(anndat_L.obs['leiden'].unique())
    print('%s clusters found. Try tuning leiden_resol to get different numbers.'%(n_clus))
    anndat_multiome.obs['leiden'] = anndat_L.obs['leiden'].to_numpy()
    anndat_multiome.obs['leiden'] = anndat_multiome.obs['leiden'].astype('category')

    if trajectory:

        print("Generate the PAGA for determining the root for trajectory analysis...")
        #sc.pp.neighbors(anndat_multiome, n_neighbors=20, n_pcs=40)
        sc.tl.paga(anndat_L, groups='leiden')

        if root is None:
            print("Please select the root cluster and run the code again with root = cluster number chosen.")
            sc.pl.paga(anndat_L, color=['leiden'])
            print()
        else:
            anndat_L.uns['iroot'] = np.flatnonzero(anndat_L.obs['leiden']  == str(root))[0]
            sc.tl.dpt(anndat_L)
            anndat_multiome.obs['dpt_pseudotime'] = anndat_L.obs['dpt_pseudotime'].to_numpy()
            print()

    print('Following changes made to the AnnData object:')
    #print("\t[obs x G-P defined region] matrix\tobsm['region_ident_n'], obsm['region_ident_b']")
    print("\tSub-cluster labels\tobs['leiden'] as category dtype.")
    if trajectory and root is not None:
        print("\tTrajectory pseudotime\tobs['dpt_pseudotime'] as numerical values.")

    return anndat_multiome









