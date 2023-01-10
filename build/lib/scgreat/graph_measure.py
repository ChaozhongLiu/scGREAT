import numpy as np
import pandas as pd
import scanpy as sc
import random
import anndata as ad
import time

from scgreat.moran_vec import Moran
import scgreat.lee_vec as lee_vec
from libpysal.weights import W


from scipy import stats
#from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection


#=========================================================================================
# Graph/Embedding related measuring functions
#=========================================================================================

def Morans_I(anndat_multiome, seed=1, max_RAM=16):
    """ 
    Function to calculate Moran's I for all the features in anndat_multiome

    Arguments
    --------------

        anndat_multiome: multiome AnnData object

        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        anndat added with
            var['Morans.I']

    """

    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

    cong_mtx = anndat_multiome.obsp['connectivities'].toarray()
    #cong_mtx = cong_mtx/cong_mtx.sum(axis=1)[:,np.newaxis]
    neighbors = {}
    weights = {}

    for i in range(cong_mtx.shape[0]):
        neighbors[i] = np.nonzero(cong_mtx[i])[0].tolist()
        weights[i] = cong_mtx[i][np.nonzero(cong_mtx[i])[0]].tolist()
    w = W(neighbors, weights)

    mi = Moran(anndat_multiome.n_obs, w, permutations=0)
    anndat_multiome.var['Morans.I'] = np.NaN

    #for i in range(anndat_multiome.n_vars):
    mi.calc_i(anndat_multiome.X, seed=seed, max_RAM=max_RAM)
    anndat_multiome.var['Morans.I'] = mi.I
    print("Following changes made to the AnnData object:")
    print("\tMoran's I value saved in var['Morans.I']")
    
    #print(mi.I)
    print("%.3fs past"%(time.time()-start))
    return anndat_multiome




def Global_L(anndat_multiome, peaks_nearby_orig=None, permutations=0, percent=0.1, seed=1, max_RAM=16):
    """ 
    Function to calculate the global L index for all the pairs in anndat_multiome

    Arguments
    --------------

        anndat_multiome: multiome AnnData object

        peaks_nearby_orig: peaks_nearby DataFrame containing pairs to be calculated
                           if not provided, use the anndat_multiome.uns['peaks_nearby']

        permutations: number of permutations for significance test.
                      Default is 0, meaning no significance test
                      999 is a good choice for the test, but it might take a long time (hours) to finish depending on the number of pairs
        
        percent: percentage of cells to shuffle during permutation.
                 For most of the time, default 0.1 is alread a good choice.
        
        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        anndat added with
            Global L results and QC metrics (sparsity) updated in uns['peaks_nearby']

    """

    random.seed(seed)
    np.random.seed(seed)

    print("Calculating global graph-based correlation value...")
    cong_mtx = anndat_multiome.obsp['connectivities'].toarray()

    eSP = lee_vec.Spatial_Pearson(cong_mtx, permutations=permutations)
    index_df = pd.DataFrame({'index':np.arange(anndat_multiome.n_vars)})
    index_df.index = anndat_multiome.var.index

    # L for all pairs
    if peaks_nearby_orig is None:
        peaks_nearby = anndat_multiome.uns['peaks_nearby'].copy()
    else:
        peaks_nearby = peaks_nearby_orig.copy()

    gene = peaks_nearby['genes']
    peak = peaks_nearby['peaks']
    gene_index = index_df.loc[gene,'index'].to_numpy()
    peak_index = index_df.loc[peak,'index'].to_numpy()

    gene_X = anndat_multiome.X[:,gene_index]
    peak_X = anndat_multiome.X[:,peak_index]

    eSP = eSP.fit(gene_X, peak_X, percent=percent, seed=seed, max_RAM=max_RAM)
    peaks_nearby['L'] = eSP.association_
    peaks_nearby['L.p_value'] = eSP.significance_ if permutations else 1.0

    if permutations:
        _,peaks_nearby['L.FDR'] = fdrcorrection(peaks_nearby['L.p_value'],
                                                alpha=0.05, method='indep')
    else:
        peaks_nearby['L.FDR'] = 1.0

    peaks_nearby['gene.pct'] = anndat_multiome.var.loc[anndat_multiome.var_names[gene_index],'Frac.all'].to_numpy()
    peaks_nearby['peak.pct'] = anndat_multiome.var.loc[anndat_multiome.var_names[peak_index],'Frac.all'].to_numpy()
    anndat_multiome.uns['peaks_nearby'] = peaks_nearby

    print("Following changes made to the AnnData object:")
    print("\tGlobal L results and QC metrics updated in uns['peaks_nearby'].")

    #global_L_df = pd.DataFrame({'L':eSP.association_,
    #                            'L.p_value': eSP.significance_ if permutations else 1.0})
    #global_L_df.index = peaks_nearby['genes'] + '_' + peaks_nearby['peaks']
    
    return anndat_multiome




def Local_L(anndat_multiome, peaks_nearby_orig=None,
            I_cutoff=0.25, pct_cutoff=0.05,
            dropout_rm=True,
            seed=1, max_RAM=16):
    """ 
    Function to calculate the local L index for all the pairs in anndat_multiome

    Arguments
    --------------

        anndat_multiome: multiome AnnData object

        peaks_nearby_orig: peaks_nearby DataFrame containing pairs to be calculated
                           if not provided, use the anndat_multiome.uns['peaks_nearby']

        I_cutoff: Moran's I cutoff for pairs to be measured

        pct_cutoff: sparsity cutoff for pairs to be measured

        dropout_rm: make local L index to be 0 if feature value is 0 (dropout)

        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        anndat added with
            Global L results and QC metrics (sparsity) updated in uns['peaks_nearby']

    """

    random.seed(seed)
    np.random.seed(seed)
    #start = time.time()

    # Moran's I as pre-filtering
    print("Filtering features by Moran's I and feature sparsity in single-cell data...")
    if 'Morans.I' in anndat_multiome.var.columns:
        print("Previously calculated Moran's I found.")
    else:
        print("Calculating Moran's I for all features...")
        anndat_multiome = Morans_I(anndat_multiome, seed=seed)

    print("Keep features with Moran's I >= %s and at least %s%% expressed in single-cell data..."%(I_cutoff,pct_cutoff*100))
    feature_pass = anndat_multiome.var_names[(anndat_multiome.var['Morans.I']>=I_cutoff) & (anndat_multiome.var['Frac.all']>=pct_cutoff)]

    if peaks_nearby_orig is None:
        peaks_nearby = anndat_multiome.uns['peaks_nearby'].copy()
        peaks_nearby_orig = peaks_nearby.copy()
    else:
        peaks_nearby = peaks_nearby_orig.copy()

    peaks_nearby = peaks_nearby.loc[peaks_nearby['genes'].isin(feature_pass),:]
    peaks_nearby = peaks_nearby.loc[peaks_nearby['peaks'].isin(feature_pass),:]
    peaks_nearby_orig['Local_L'] = (peaks_nearby_orig['peaks'].isin(feature_pass)) & (peaks_nearby_orig['genes'].isin(feature_pass))
    print("%s pairs left for graph-based correlation L calculation."%(peaks_nearby.shape[0]))
    print()

    print("Calculating local graph-based correlation value...")
    
    L_mtx, L_mtx_names = _calculate_LL(anndat_multiome, peaks_nearby,
                                       dropout_rm=dropout_rm,
                                       seed=1, max_RAM=16)

    if dropout_rm:
        print("Set L to 0 for cells with no expression on either feature of a certain pair...")
        GP_names = L_mtx_names
        Dropout_mtx = dropout_filter(anndat_multiome, GP_names)
        L_mtx = L_mtx * Dropout_mtx

    anndat_multiome.uns['Local_L'] = L_mtx
    anndat_multiome.uns['Local_L_names'] = L_mtx_names

    anndat_multiome.uns['peaks_nearby'] = peaks_nearby_orig.copy()
    print("Following changes made to the AnnData object:")
    print("\tGraph-based Pearson Correlation results saved in uns['Local_L']")
    print("\tuns['peaks_nearby']['Local_L'] added indicates feature selected for local L calculation or not.")

    #print("%.3fs past"%(time.time()-start))
    return anndat_multiome



def _calculate_LL(anndat_multiome, peaks_nearby,
                  dropout_rm=True,
                  seed=1, max_RAM=16):
    random.seed(seed)
    np.random.seed(seed)

    anndat_multiome.var['index'] = np.arange(anndat_multiome.n_vars)
    gene = peaks_nearby['genes']
    peak = peaks_nearby['peaks']
    pair_names = gene + '_' + peak
    gene_index = anndat_multiome.var.loc[gene,'index'].to_numpy()
    peak_index = anndat_multiome.var.loc[peak,'index'].to_numpy()
    gene_X = anndat_multiome.X[:,gene_index]
    peak_X = anndat_multiome.X[:,peak_index]

    cong_mtx = anndat_multiome.obsp['connectivities'].toarray()

    eSP2 = lee_vec.Spatial_Pearson_Local(cong_mtx, permutations=0)
    eSP2 = eSP2.fit(gene_X,peak_X,seed=seed, max_RAM=max_RAM)
    local_L_df = pd.DataFrame(eSP2.associations_)
    local_L_df.columns = pair_names

    '''
    if permutations:
        local_p_df = pd.DataFrame(eSP2.significance_)
        local_p_df.columns = pair_names
    '''

    L_mtx = local_L_df.to_numpy()

    return L_mtx, local_L_df.columns.to_numpy()


def Local_L_bygroup(anndat_multiome, groupby=None,
                    groups_list = None,
                    peaks_nearby_orig=None,
                    I_cutoff=0.25, pct_cutoff=0.05,
                    dropout_rm=True,
                    seed=1, max_RAM=16):
    """ 
    Function to calculate the local L index for all the pairs in anndat_multiome BY GROUP
    This is needed in cluster marker discovery

    Arguments
    --------------

        anndat_multiome: multiome AnnData object

        groupby: column name in anndat_multiome.obs to be used for group cells

        groups_list: label list specifying the clusters to include

        peaks_nearby_orig: peaks_nearby DataFrame containing pairs to be calculated
                           if not provided, use the anndat_multiome.uns['peaks_nearby']

        I_cutoff: Moran's I cutoff for pairs to be measured

        pct_cutoff: sparsity cutoff for pairs to be measured

        dropout_rm: make local L index to be 0 if feature value is 0 (dropout)

        seed: random seed to make the results reproducible

        max_RAM: maximum limitation of memory (Gb)


    Returns
    --------------
        anndat added with
            Global L results and QC metrics (sparsity) updated in uns['peaks_nearby']

    """

    random.seed(seed)
    np.random.seed(seed)

    # Moran's I as pre-filtering
    print("Filtering features by Moran's I and feature sparsity in single-cell data...")
    if 'Morans.I' in anndat_multiome.var.columns:
        print("Previously calculated Moran's I found.")
    else:
        print("Calculating Moran's I for all features...")
        anndat_multiome = Morans_I(anndat_multiome, seed=seed)

    print("Keep features with Moran's I > %s and at least %s%% expressed in single-cell data..."%(I_cutoff,pct_cutoff*100))
    feature_pass = anndat_multiome.var_names[(anndat_multiome.var['Morans.I']>=I_cutoff) & (anndat_multiome.var['Frac.all']>=pct_cutoff)]

    if peaks_nearby_orig is None:
        peaks_nearby = anndat_multiome.uns['peaks_nearby'].copy()
        peaks_nearby_orig = peaks_nearby.copy()
    else:
        peaks_nearby = peaks_nearby_orig.copy()

    peaks_nearby = peaks_nearby.loc[peaks_nearby['genes'].isin(feature_pass),:]
    peaks_nearby = peaks_nearby.loc[peaks_nearby['peaks'].isin(feature_pass),:]
    peaks_nearby_orig['Local_L_by_%s'%groupby] = (peaks_nearby_orig['peaks'].isin(feature_pass)) & (peaks_nearby_orig['genes'].isin(feature_pass))
    print("%s pairs left for graph-based correlation L calculation."%(peaks_nearby.shape[0]))
    print()

    print("Calculating local graph-based correlation value...")

    if groups_list is None:
        groups = anndat_multiome.obs[groupby].value_counts() > 1
        groups = groups.loc[groups].index.to_list()
    else:
        groups = groups_list

    L_mtx = np.ones((anndat_multiome.n_obs, peaks_nearby.shape[0]))
    for group in groups:
        anndat_tmp = anndat_multiome[anndat_multiome.obs[groupby]==group].copy()
        L_mtx_tmp, L_mtx_names = _calculate_LL(anndat_tmp, peaks_nearby,
                                               dropout_rm=True,
                                               seed=1, max_RAM=16)
        L_mtx[(anndat_multiome.obs[groupby]==group).tolist(),:] = L_mtx_tmp

    if dropout_rm:
        print("Set L to 0 for cells with no expression on either feature of a certain pair...")
        GP_names = L_mtx_names
        Dropout_mtx = dropout_filter(anndat_multiome, GP_names)
        L_mtx = L_mtx * Dropout_mtx


    anndat_multiome.uns['peaks_nearby'] = peaks_nearby_orig.copy()

    if groups_list is None:
        anndat_multiome.uns['Local_L_by_%s'%groupby] = L_mtx
        anndat_multiome.uns['Local_L_by_%s_names'%groupby] = L_mtx_names

        print("Following changes made to the AnnData object:")
        print("\tGraph-based Pearson Correlation results saved in uns['Local_L_by_%s']"%groupby)
        print("\tuns['peaks_nearby']['Local_L_by_%s'] added indicates feature selected for per group local L calculation or not."%groupby)

    else:
        anndat_multiome.uns['Local_L_by_%s_pair'%groupby] = L_mtx
        anndat_multiome.uns['Local_L_by_%s_pair_names'%groupby] = L_mtx_names

        print("Following changes made to the AnnData object:")
        print("\tGraph-based Pearson Correlation results saved in uns['Local_L_by_%s_pair']"%groupby)
        print("\tuns['peaks_nearby']['Local_L_by_%s'] added indicates feature selected for per group local L calculation or not."%groupby)

    #print("%.3fs past"%(time.time()-start))
    return anndat_multiome




def dropout_filter(anndat_multiome, GP_names):
    index_df = pd.DataFrame({'index':np.arange(len(anndat_multiome.var_names))})
    index_df.index = anndat_multiome.var_names
    GP_G = [gp.split('_')[0] for gp in GP_names]
    GP_P = [gp.split('_')[1] for gp in GP_names]
    GP_genes_index = index_df.loc[GP_G,:]['index'].to_numpy()
    GP_peaks_index = index_df.loc[GP_P,:]['index'].to_numpy()
    
    E_mtx = anndat_multiome.X
    #E_mtx_dropout_value = E_mtx.min(axis=0)
    #E_mtx_dropout_value = E_mtx_dropout_value[np.newaxis,:][np.zeros(anndat_multiome.n_obs).astype(int)] #.shape
    E_mtx_dropout_value = np.zeros(E_mtx.shape)
    #Dropout_mtx = (E_mtx_dropout_value != E_mtx).astype(int)
    Dropout_mtx = (~np.isclose(E_mtx_dropout_value,E_mtx, 1e-3)).astype(int)
    #print(Dropout_mtx)
    Dropout_mtx_G = Dropout_mtx[:,GP_genes_index]
    Dropout_mtx_P = Dropout_mtx[:,GP_peaks_index]
    
    Dropout_mtx = Dropout_mtx_G * Dropout_mtx_P
    
    return Dropout_mtx




#=========================================================================================
# Pearson correlation function for benchmark purpose
#=========================================================================================

def Global_Pearson(anndat_multiome, peaks_nearby_orig=None, p_value=False, seed=1):
    random.seed(seed)
    np.random.seed(seed)

    print("Calculating Pearson's correlation coefficient...")
    index_df = pd.DataFrame({'index':np.arange(anndat_multiome.n_vars)})
    index_df.index = anndat_multiome.var.index

    if peaks_nearby_orig is None:
        peaks_nearby = anndat_multiome.uns['peaks_nearby'].copy()
    else:
        peaks_nearby = peaks_nearby_orig.copy()
    gene = peaks_nearby['genes']
    peak = peaks_nearby['peaks']
    gene_index = index_df.loc[gene,'index'].to_numpy()
    peak_index = index_df.loc[peak,'index'].to_numpy()

    gene_X = anndat_multiome.X[:,gene_index]
    peak_X = anndat_multiome.X[:,peak_index]

    gene_X = pd.DataFrame(gene_X, index=np.arange(gene_X.shape[0]), columns=np.arange(gene_X.shape[1]))
    peak_X = pd.DataFrame(peak_X, index=np.arange(peak_X.shape[0]), columns=np.arange(gene_X.shape[1]))

    global_P_df = pd.DataFrame(peak_X.corrwith(gene_X, method='pearson', axis=0))
    global_P_df.index = peaks_nearby['genes'] + '_' + peaks_nearby['peaks']
    global_P_df.columns = ['r']

    if p_value:
        print("Calculating p-values...")
        p_list = []
        for i in range(global_P_df.shape[0]):
            p_list.append(stats.pearsonr(gene_X.iloc[:,i].to_numpy(), peak_X.iloc[:,i].to_numpy())[1])
        global_P_df['r.p_value'] = p_list
        print("Calculating FDR...")
        _, global_P_df['r.FDR'] = fdrcorrection(global_P_df['r.p_value'],
                                                alpha=0.05, method='indep')
    else:
        global_P_df['r.p_value'] = 1.0
        global_P_df['r.FDR'] = 1.0

    peaks_nearby['r'] = global_P_df['r'].to_numpy()
    peaks_nearby['r.p_value'] = global_P_df['r.p_value'].to_numpy()
    peaks_nearby['r.FDR'] = global_P_df['r.FDR'].to_numpy()

    anndat_multiome.uns['peaks_nearby'] = peaks_nearby.copy()

    print("Following changes made to the AnnData object:")
    print("\tPearson Correlation results saved in uns['peaks_nearby']")

    return anndat_multiome






