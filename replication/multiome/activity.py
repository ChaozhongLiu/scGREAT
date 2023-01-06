import numpy as np
import pandas as pd
import anndata as ad
import time
import scipy as sp

import multiome.graph_measure as graph_measure





def calculate_weights(anndat, upstream=None, downstream=None):

    peaks_nearby = anndat.uns['peaks_nearby'].copy()

    if 'peaks_nearby' not in anndat.uns.keys():
        print("Genes' nearby peaks information missed. Please add 'peaks_nearby' DataFrame in the AnnData object uns.")
        return

    if 'L' in anndat.uns['peaks_nearby']:
        print("Previously computated global L found in anndat.uns['peaks_nearby'].\nWill skip the computation again.")

    else:
        print("No global L score found in anndat.uns['peaks_nearby'].\nWill compute now.")
        #start = time.time()

        #global_L_df = graph_measure.Global_L(anndat, peaks_nearby, permutations=0, seed=1)
        #peaks_nearby['L'] = global_L_df['L'].to_numpy()

    peaks_nearby['weight'] = peaks_nearby['L'].copy()
    
    return peaks_nearby





def calculate_weights_pearson(anndat, upstream=None, downstream=None):

    if 'peaks_nearby' not in anndat.uns.keys():
        print("Genes' nearby peaks information missed. Please add 'peaks_nearby' DataFrame in the AnnData object uns.")
        return

    if 'r' in anndat.uns['peaks_nearby']:
        print("Previously computated global P found in anndat.uns['peaks_nearby'].\nWill skip the computation again.")

    else:
        print("No Pearson coefficient score found in anndat.uns['peaks_nearby'].\nWill compute now.")
        #start = time.time()

        #peaks_nearby = anndat.uns['peaks_nearby'].copy()
        #global_P_df = graph_measure.common_Pearson(anndat, peaks_nearby, seed=1)
        #peaks_nearby['P'] = global_P_df[0].to_numpy()

    peaks_nearby['weight'] = peaks_nearby['r'].copy()
    
    return peaks_nearby





def GeneActivity(anndat, peaks_nearby_orig, L_cutoff, weight='all', distance_decay=True, non_neg=False):
    """ plans
    weight: all, distal, none
            all: the weight of promoter and gene body peaks will also be used
            distal: the weight of promoter and gene body peaks also 1
            none: no L weight used

    distance_decay: wether to use distance decay

    what about genes don't have related peaks?

    after construction, make negative value zero?

    """

    """ Strategy applied from ArchR
    genes with correlated peaks
       - promoter
       - gene boday
       - distal ones

    genes without correlated peaks
       - mean
       - 

    """
    #if 'peaks_nearby' not in anndat.uns.keys():
    #    print("Genes' nearby peaks information missed. Please add 'peaks_nearby' DataFrame in the AnnData object uns and run calculate_weights( ).")
    #    return

    if 'L' not in peaks_nearby_orig.columns:
        print("Global L not found in peaks_nearby.\nPlease run calculate_weights( ) first.")
        return

    #elif np.any(anndat.X < 0):
    #    print("Negative values in X detected. Please make sure the AnnData.X is counts matrix.\n")
    #    return
    else:
        print("No negative values in X. But please make sure the AnnData.X is counts matrix.\nContinue generating gene activity matrix...")

    print("Drop peaks that are not in TSS regions or gene body with L less than %s."%L_cutoff)
    peaks_nearby = peaks_nearby_orig.copy()
    peaks_nearby.loc[(np.abs(peaks_nearby['L'])<L_cutoff) & (peaks_nearby['pRegion']==0) & (peaks_nearby['gBody']==0),'weight'] = 0.0
    peaks_nearby = peaks_nearby.loc[peaks_nearby['weight']!=0.0,:]

    ## distance decay weight calculation
    if distance_decay:
        peaks_nearby['dist_w'] = np.exp(-np.abs(peaks_nearby['tss_dist']/5000)) + np.exp(-1)
        peaks_nearby.loc[peaks_nearby['tts_dist']>0, 'dist_w'] = np.exp(-np.abs(peaks_nearby['tts_dist']/5000)) + np.exp(-1)
        peaks_nearby.loc[(peaks_nearby['pRegion']==1) | (peaks_nearby['gBody']==1), 'dist_w'] = 1 + np.exp(-1)
    else:
        peaks_nearby['dist_w'] = 1.0


    ## final weight calculation
    if weight == 'all':
        peaks_nearby['final_w'] = peaks_nearby['dist_w'] * peaks_nearby['weight']
    elif weight == 'none':
        peaks_nearby['final_w'] = peaks_nearby['dist_w']
    elif weight == 'distal':
        partial_L_w = peaks_nearby['weight'].to_numpy()
        partial_L_w[((peaks_nearby['pRegion']==1) | (peaks_nearby['gBody']==1)).to_numpy()] = 1.0
        peaks_nearby['final_w'] = peaks_nearby['dist_w'] * partial_L_w

    peaks_nearby_2 = peaks_nearby.loc[peaks_nearby['final_w']!=0,:].copy()

    ## weight matrix construction
    index_df_peaks = pd.DataFrame({'index':np.arange(anndat.n_vars)},
                                  index=anndat.var_names)
    index_df_genes = pd.DataFrame({'index':np.arange(len(peaks_nearby_2['genes'].unique()))},
                                  index=peaks_nearby_2['genes'].unique())

    weight_mtx = sp.sparse.coo_matrix((peaks_nearby_2['final_w'].to_numpy(),
                                      (
                                        index_df_peaks.loc[peaks_nearby_2['peaks'].to_numpy(),'index'].to_numpy(),
                                        index_df_genes.loc[peaks_nearby_2['genes'].to_numpy(),'index'].to_numpy()
                                      )),
                                      shape=(index_df_peaks.shape[0], index_df_genes.shape[0])
                                     ).tocsr()

    ## calculate activity matrix utilizing sparse matrix
    X_raw = sp.sparse.csr_matrix(anndat.X)
    act_mtx = X_raw @ weight_mtx

    if non_neg:
        act_mtx = act_mtx.toarray()
        act_mtx[act_mtx<0] = 0
        act_mtx = sp.sparse.csr_matrix(act_mtx)


    return act_mtx, index_df_genes.index.to_numpy()




