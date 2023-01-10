import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import random
import time

from sklearn.neighbors import KNeighborsRegressor


"""

Functions to extract neighborhood information from the files user provided.

    nn_map_Seurat
    --------------
    Given the n_cells x n_neighbors Data Frame from Seurat (cell index start from 1), 
    generate the nn_map needed for pseudo-bulk generatiion

    nn_map_py
    --------------
    Given the n_cells x n_neighbors Data Frame from python packages like scglue (cell index start from 0),
    generate the nn_map needed for pseudo-bulk generatiion

    nn_map_embed
    --------------
    Given the n_cells x n_dims array-like data,
    calculate the nearest neighbors, 
    then generate the nn_map

    nn_map_scanpy
    --------------
    Given the AnnData object processed with Scanpy,
    generate the nn_map from distance matrix saved in obsp['distances']

Arguments
--------------
    K: default 20
       number of neighbors (including self) to generate the nn_map

"""
def nn_map_Seurat(seurat_wnn, K=20):
    nn_map = seurat_wnn.iloc[:,0:(K-1)].to_numpy()
    nn_map = nn_map - 1
    nn_map = np.append(np.arange(nn_map.shape[0]).reshape(nn_map.shape[0],-1), nn_map, axis = 1)
    return nn_map


def nn_map_py(nn_map, K=20):
    nn_map = nn_map.iloc[:,0:(K-1)].to_numpy()
    nn_map = np.append(np.arange(nn_map.shape[0]).reshape(nn_map.shape[0],-1), nn_map, axis = 1)
    return nn_map


def nn_map_embed(embed_mtx, K=20):
    ann_reduct = ad.AnnData(
        X = embed_mtx
    )
    sc.pp.neighbors(ann_reduct, use_rep="X", n_neighbors = K)
    nn_map = ann_reduct.obsp['distances']
    nn_map = nn_map.nonzero()[1].reshape(-1,int(K-1))
    nn_map = np.append(nn_map, np.arange(nn_map.shape[0]).reshape(nn_map.shape[0],-1), axis = 1)
    return nn_map


def nn_map_scanpy(anndat, K=20):

    dist_mtx = sp.sparse.coo_matrix((
            anndat.obsp['distances'].data, (
                anndat.obsp['distances'].nonzero()[0],
                np.tile(np.arange(K-1), anndat.n_obs)
            )), shape=(anndat.n_obs, K-1)
        ).tocsr().toarray()

    ind_mtx = sp.sparse.coo_matrix((
                anndat.obsp['distances'].nonzero()[1], (
                    anndat.obsp['distances'].nonzero()[0],
                    np.tile(np.arange(K-1), anndat.n_obs)
                )), shape=(anndat.n_obs, K-1)
            ).tocsr().toarray()

    ind_mtx_new = ind_mtx[np.repeat(np.arange(ind_mtx.shape[0]), K-1),
                      dist_mtx.argsort(axis=1).flatten()].reshape(dist_mtx.shape[0],-1)

    ind_mtx_new = np.concatenate([np.arange(ind_mtx_new.shape[0]).reshape(-1,1), ind_mtx_new], axis=1)

    return ind_mtx_new



def _nnmap_by_group(nn_map, group_labels):
    """
    Function to make the neighbors of each cell be within the same label group
    
    Arguments
    --------------
        nn_map: (n_cells, n_neighbors) numpy array containing neighbors from different label groups

        group_labels: (n_cells,) array-like object for cell labels
    
    Returns
    --------------
        nn_map containing neighborhoods with cells from the same group

        group label for each neighborhood


    """
    nn_labels = group_labels[nn_map]
    mode_row = stats.mode(nn_labels, axis=1, keepdims=True)[0]
    mixed_row = np.arange(nn_labels.shape[0])[~np.all(nn_labels == mode_row, axis=1)]

    for ri in mixed_row:
        col_to_replace = nn_labels[ri,:] != mode_row[ri]
        num_replace = int(np.sum(col_to_replace))
        nn_map[ri,col_to_replace] = np.random.choice(nn_map[ri,~col_to_replace], 
                                                     size=num_replace, replace=True)

    return nn_map, mode_row.reshape(-1)





def subset(anndat_mod1, anndat_mod2, nn_map, by, groups, seed=1):
    """
    Function to subset the dataset to spcific groups

    Arguments
    --------------
        anndat_mod1: data module 1, i.e., scRNA-seq data

        anndat_mod2: data module 2, i.e., scATAC-seq data,
                     cells should be in the same order and index as anndat_mod1

        nn_map: (n_cells, n_neighbors) numpy array

        by: label name in anndat_mod1.obs

        groups: list of labels to be included

        seed: random seed for re-sampling neighbors

    Return
    --------------
        Subseted
            anndat_mod1
            anndat_mod2
            nn_map

    """
    random.seed(seed)
    np.random.seed(seed)
    K = nn_map.shape[1]

    assert np.all(anndat_mod1.obs_names == anndat_mod2.obs_names)
    mask = anndat_mod1.obs[by].isin(groups)
    
    nn_map = nn_map[:,1:]
    con_mtx = sp.sparse.coo_matrix((
            np.ones(int(nn_map.shape[0]*nn_map.shape[1])), (
                np.repeat(np.arange(nn_map.shape[0]), nn_map.shape[1]),
                nn_map.reshape(-1)
            )), shape=(nn_map.shape[0], nn_map.shape[0])
        ).tocsr()
    
    anndat_mod1.obsp['connectivity'] = con_mtx
    anndat_mod1 = anndat_mod1[mask]
    anndat_mod2 = anndat_mod2[mask]
    
    nn_long = pd.DataFrame(anndat_mod1.obsp['connectivity'].nonzero()).T
    nn_map = np.empty((anndat_mod1.n_obs, K-1))
    for i in range(nn_map.shape[0]):
        num_nb = len(nn_long.loc[nn_long[0]==i,1])
        if num_nb >= K-1:
            nn_map[i] = np.random.choice(nn_long.loc[nn_long[0]==i,1].to_numpy(), K-1, 
                                         replace=False)
        elif num_nb <= 2:
            nn_map[i] = -np.ones(K-1)
        else:
            nn_map[i] = np.random.choice(nn_long.loc[nn_long[0]==i,1].to_numpy(), K-1,
                                         replace=True)
    nn_map = np.append(np.arange(nn_map.shape[0]).reshape(nn_map.shape[0],-1), nn_map, axis = 1)

    
    return anndat_mod1, anndat_mod2, nn_map.astype(int)



def fill_miss_index(df, n):
    miss_ind = np.setdiff1d(np.arange(n), df['ind'].to_numpy())
    miss_pd = pd.DataFrame({'ind':miss_ind,'size':0})
    df = pd.concat([df,miss_pd], axis=0).sort_values('ind')
    return df


def add_group_sparsity(anndat, group_by=None):
    """
    Function to add sparisity info in AnnData.var

    Arguments
    --------------
        anndat: scRNA-seq or scATAC-seq data

        group_by: if provided, calculate per group feature sparsity

    Return
    --------------
        anndat with
            var['Frac.all']
            var['Frac.GroupName']

    """

    spsmtx = sp.sparse.csr_matrix(anndat.X)
    non_zero = spsmtx.nonzero()
    non_zero_row = non_zero[0]
    non_zero_ind = non_zero[1]

    ## All cell sparsity
    non_zero_pd = pd.DataFrame(non_zero_ind)
    non_zero_pd = non_zero_pd.groupby(by=0, as_index=False).size()
    non_zero_pd = pd.DataFrame(non_zero_pd)
    non_zero_pd.columns = ['ind','size']
    non_zero_pd = fill_miss_index(non_zero_pd, anndat.n_vars)
    non_zero_pd['sparsity'] = non_zero_pd['size'] / anndat.n_obs

    anndat.var['Frac.all'] = non_zero_pd['sparsity'].to_numpy()

    ## Per group sparsity
    if group_by is not None:
        non_zero_pd = pd.DataFrame({'ind':non_zero_ind,
                                    'group':anndat.obs[group_by].to_numpy()[non_zero_row]})
        non_zero_pd = non_zero_pd.groupby(by=['ind','group'], as_index=False).size().sort_values('group')
        non_zero_pd['n_cells'] = anndat.obs[group_by].value_counts()[non_zero_pd['group']].to_numpy()

        for ident in anndat.obs[group_by].unique():
            df = non_zero_pd.loc[non_zero_pd['group']==ident,:].copy()
            n_cells = df['n_cells'].tolist()[0]
            df = df.loc[:,['ind','size']]
            df = fill_miss_index(df, anndat.n_vars)
            df['sparsity'] = df['size'] / n_cells
            anndat.var['Frac.%s'%ident] = df['sparsity'].to_numpy()


    return anndat





def PsedoBulk(anndat_mod1, anndat_mod2, nn_map, 
              N=2000, K=20,
              group_name=None, seed=1):
    """
    Function to generate pseudo-bulk data

    Arguments
    --------------
        anndat_mod1: data module 1, i.e., scRNA-seq data

        anndat_mod2: data module 2, i.e., scATAC-seq data

        nn_map: (n_cells, n_neighbors) numpy array containing neighborhoods to merge

        group_by: if provided, calculate per group feature sparsity

        N: maximum number of pseudo-bulks

        K: number of nearest neighbors to be aggregated

        group_name: annotation saved in anndat_mod1.obs. If provided, bulk will be generated within clusters.

        seed: random seed to make the results reproducible

    Return
    --------------
        pseudo-bulk data for
            anndat_mod1
            anndat_mod2

    """


    random.seed(seed)
    np.random.seed(seed)

    if group_name is not None:
        print("Cluster annotation provided, will generate the pseudo-bulk within each cluster defined...")
        group_labels = anndat_mod1.obs[group_name].to_numpy()
        nn_map, group_labels_knn = _nnmap_by_group(nn_map, group_labels)
    else:
        print("No cluster annotation provided, will generate the pseudo-bulk by KNN...")


    nn_map = nn_map[:,0:K]
    good_choices = np.arange(nn_map.shape[0])

    choice = np.random.choice(len(good_choices))
    chosen = np.array([good_choices[choice]])
    good_choices = good_choices[good_choices != good_choices[choice]]
    it = 0
    k2 = K * 2 # Compute once


    while len(good_choices) > 0 and len(chosen) < N:
        it += 1
        choice = np.random.choice(len(good_choices))
        new_chosen = np.append(chosen, good_choices[choice])
        good_choices = good_choices[good_choices != good_choices[choice]]
        cell_sample = nn_map[new_chosen,:]
        if np.any(cell_sample==-1.):
            continue
        #others = np.arange(cell_sample.shape[0] - 1)
        #this_choice = cell_sample[-1]
        shared = pd.DataFrame(cell_sample[0:-1]).isin(cell_sample[-1]).sum(axis=1).max()
        shared = shared / K
        if shared < 0.9:
            chosen = new_chosen

    cell_sample = nn_map[chosen,:]
    if group_name is not None:
        bulk_labels = group_labels_knn[chosen]
        obs_df = pd.DataFrame({group_name:bulk_labels})
        obs_df[group_name] = obs_df[group_name].astype('category')

    agg_sum = sp.sparse.coo_matrix((
        np.ones(cell_sample.size), (
            np.repeat(np.arange(cell_sample.shape[0]), K),
            cell_sample.reshape(-1)
        )), shape=(cell_sample.shape[0], anndat_mod1.n_obs)
    ).tocsr()

    bulk_mod1_X = agg_sum @ anndat_mod1.X
    bulk_mod2_X = agg_sum @ anndat_mod2.X

    obsm = {'NN':cell_sample}
    #obsm.index = obsm.index.astype(str)
    #obsm.columns = 'NN_' + pd.Index(np.arange(K).astype(str))
    print("%s pseudo-bulk generated"%(bulk_mod1_X.shape[0]))

    return ad.AnnData(
        X=bulk_mod1_X, var=anndat_mod1.var, obsm=obsm,
        obs=obs_df if group_name is not None else None,
        uns={'group':group_name},
        ), ad.AnnData(
        X=bulk_mod2_X, var=anndat_mod2.var, obsm=obsm,
        obs=obs_df if group_name is not None else None,
        uns={'group':group_name},
        )





def PseudoBulk_Summary(anndat):
    """
    Function to give user a quality summary of pseudo-bulk generation

    Arguments
    --------------
        anndat: Pseudo-bulk AnnData object 1

    Returns
    --------------
        print summary information on screen
        frequency plot - number of shared cells among pseudo-bulks

    """

    cell_sample = anndat.obsm['NN']
    shared_all = np.array([])
    for i in range(len(cell_sample)):
        shared = pd.DataFrame(cell_sample).isin(cell_sample[i]).sum(axis=1).to_numpy()
        shared = np.delete(shared, i)
        shared_all = np.append(shared_all, shared)

    print("Overlap QC metrics:\nNumber of cells included:", len(np.unique(cell_sample)),
          "\nCells per bin:", cell_sample.shape[1],
          "\nMaximum shared cells bin-bin:", np.max(shared_all),
          "\nMean shared cells bin-bin:", np.mean(shared_all),
          "\nMedian shared cells bin-bin:", np.median(shared_all))

    plot = plt.hist(shared_all)
    plt.xlabel('Number of shared cells')
    plt.ylabel('Frequency')







def RNA_preprocessing_in_one(anndat, HVGs=None,
                             sparse_cutoff=0.1,
                             n_neighbors=20, n_pcs=30):
    """
    Funcion to preprocess the input RNA-seq data all in one function

    Arguments
    --------------
        anndat: scRNA-seq AnnData object

        HVGs: gene list as Highly-Variable-Genes for standard scRNA-seq data processing
              if None, HVGs will be calculated by Scanpy

        sparse_cutoff: sparsity cutoff as a quality control for all genes

        n_neighbors: number of neighbors in scanpy.pp.neighbors()

        n_pcs: number of PCs to use in scanpy.pp.neighbors()

    Returns
    --------------
        Preprocessed scRNA-seq AnnData object with
            HVGs as features
            normalized and log-transformed matrix
            PCA, nearest neighbor graph, and UMAP


    """
    print("---- scRNA-seq pre-processing -----")
    print("Filter genes expressed in less than %s%% cells..."%(100*sparse_cutoff))
    sc.pp.filter_genes(anndat, min_cells=int(sparse_cutoff*anndat.n_obs))
    anndat.var['mt'] = anndat.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(anndat, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    print("Do library size correction and log-transformation...")
    sc.pp.normalize_total(anndat, target_sum=1e6)

    #print("Library size normalized counts saved in uns['counts']")
    #anndat.uns['counts'] = anndat.X.copy()

    sc.pp.log1p(anndat)

    sc.pp.highly_variable_genes(anndat)

    if HVGs is not None:
        print("Highly-variable genes provided, will use it for dimension reduction.")
        gene_index = pd.DataFrame(HVGs)
        gene_index = gene_index[0][gene_index[0].isin(anndat.var_names)]
    else:
        print("Highly-variable genes not provided, will use scanpy.pp.highly_variable_genes defined HVGs.")
        gene_index = anndat.var_names[anndat.var.highly_variable]

    print("Perform dimension reduction with HVGs")
    anndat.var['variances_norm'] = np.sqrt(anndat.var['dispersions'])
    #sc.pp.regress_out(anndat, ['total_counts', 'pct_counts_mt'])

    anndat.raw = anndat
    anndat = anndat[:, gene_index]
    sc.pp.scale(anndat, max_value=10)

    sc.tl.pca(anndat, svd_solver='arpack', use_highly_variable=False)
    sc.pp.neighbors(anndat, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(anndat)
    print("Preprocessing done!")

    return anndat


def ATAC_preprocessing_in_one(anndat,
                              sparse_cutoff=0.05):
    """
    Funcion to preprocess the input ATAC-seq data all in one function

    Arguments
    --------------
        anndat: ATAC-seq AnnData object

        sparse_cutoff: sparsity cutoff as a quality control for all peaks


    Returns
    --------------
        Preprocessed ATAC-seq AnnData object with
            normalized and log-transformed matrix


    """
    ## AnnData Mod2 processing
    print("---- scATAC-seq pre-processing -----")
    print("Filter peaks expressed in less than %s%% cells"%(100*sparse_cutoff))
    sc.pp.filter_genes(anndat, min_cells=int(sparse_cutoff*anndat.n_obs))
    print("Do library size correction, log-transformation, and scaling...")
    sc.pp.normalize_total(anndat)

    #print("Library size normalized counts saved in layers['counts']")
    #anndat.layers['counts'] = anndat.X.copy()

    sc.pp.log1p(anndat)
    if sp.sparse.issparse(anndat.X):
        anndat.var['mean'] = anndat.X.toarray().mean(axis=0)
        anndat.var['std'] = anndat.X.toarray().std(axis=0)
    else:
        anndat.var['mean'] = anndat.X.mean(axis=0)
        anndat.var['std'] = anndat.X.std(axis=0)
    #X = anndat.X
    #sc.pp.scale(anndat, max_value=10)
    #anndat.X = X

    print("Preprocessing done!")

    return anndat





def multiome_data(anndat_mod1, anndat_mod2, peaks_nearby):
    """ 
    Function to generate the final multiome AnnData object

    Arguments
    --------------
        anndat_mod1: AnnData object 1, normalized and log-transformed, KNN graph required

        anndat_mod2: AnnData object 2, normalized and log-transformed

        peaks_nearby: pandas DataFrame of gene and nearby peaks information
                      get this from peaks_within_distance() function

    Returns
    --------------
        Multiome AnnData object with
            normalized and log-transformed gene and peak matrix
            metadata from scRNA-seq data
            dimensional reduction results from scRNA-seq data
            connectivities and distances matrix among cells

    """
    gene_mask = anndat_mod1.raw.var_names.isin(peaks_nearby['genes'].unique())
    peak_mask = anndat_mod2.var_names.isin(peaks_nearby['peaks'].unique())

    print("AnnData object genertated with log-transformed data as X.")
    genes_X = anndat_mod1.raw.X.copy()[:, gene_mask]

    if sp.sparse.issparse(anndat_mod2.X):
        peaks_X = anndat_mod2.X.toarray()[:, peak_mask]
    else:
        peaks_X = anndat_mod2.X.copy()[:, peak_mask]

    weights = anndat_mod1.obsp['connectivities']
    distance = anndat_mod1.obsp['distances']
    anndat_multiome = ad.AnnData(
        X = np.concatenate((genes_X, peaks_X), axis=1),
        obs = anndat_mod1.obs,
        obsm = anndat_mod1.obsm,
        obsp = {'connectivities': weights, 'distances':distance},
        var = pd.concat([anndat_mod1.raw.var.loc[gene_mask,:],anndat_mod2.var.loc[peak_mask,:]]),
    )
    anndat_multiome.var_names = np.concatenate((anndat_mod1.raw.var_names.to_numpy()[gene_mask], 
                                          anndat_mod2.var_names.to_numpy()[peak_mask]))

    if np.any(anndat_multiome.var_names.str.contains('_')):
        print("'_' detected in some features, will change it to '.' to avoid conflicting.")
        anndat_multiome.var_names = anndat_multiome.var_names.str.replace('_','.')

    anndat_multiome.obs_names = anndat_mod1.obs_names

    #re-index of DataFrame
    peaks_nearby.index = np.arange(len(peaks_nearby['genes']))
    anndat_multiome.uns['peaks_nearby'] = peaks_nearby
    anndat_multiome.var = anndat_multiome.var.drop(columns=['mt','highly_variable'])

    return anndat_multiome



'''
def citeseq_data(anndat_mod1, anndat_mod2, genes, prtns):
    """ Functions to generate the multiome AnnData object

    anndat_mod1: AnnData object 1, scaling and KNN graph required

    anndat_mod2: AnnData object 2, scaling required

    peaks_nearby: pandas DataFrame of gene and nearby peaks information
    """
    #gene_mask = np.isin(anndat_mod1.raw.var_names, peaks_nearby['genes'].to_numpy())
    gene_mask = anndat_mod1.raw.var_names.isin(genes)
    #peak_mask = np.isin(anndat_mod2.var_names, peaks_nearby['peaks'].to_numpy())
    prtn_mask = anndat_mod2.var_names.isin(prtns)

    genes_X = anndat_mod1.raw.X.copy()[:, gene_mask]
    prtns_X = anndat_mod2.X.copy()[:, prtn_mask]

    weights = anndat_mod1.obsp['connectivities']
    distance = anndat_mod1.obsp['distances']
    anndat_cite = ad.AnnData(
        X = np.concatenate((genes_X, prtns_X), axis=1),
        obs = anndat_mod1.obs,
        obsm = anndat_mod1.obsm,
        obsp = {'connectivities': weights, 'distances':distance},
        var = pd.concat([anndat_mod1.raw.var.loc[gene_mask,:],anndat_mod2.var.loc[prtn_mask,:]]),
    )
    anndat_cite.var_names = np.concatenate((anndat_mod1.raw.var_names.to_numpy()[gene_mask], 
                                            anndat_mod2.var_names.to_numpy()[prtn_mask]))
    anndat_cite.obs_names = anndat_mod1.obs_names

    # Pair DataFrame
    genes = anndat_mod1.raw.var_names.to_numpy()[gene_mask]
    prtns = anndat_mod2.var_names.to_numpy()[prtn_mask]
    cite_pair = pd.DataFrame({'genes': np.repeat(genes, len(prtns)),
                              'peaks': np.tile(prtns, len(genes))})
    cite_pair.index = np.arange(len(cite_pair['genes']))
    anndat_cite.uns['peaks_nearby'] = cite_pair
    #anndat_cite.var = anndat_cite.var.drop(columns=['mt','highly_variable'])

    return anndat_cite
'''



def map_back_sc(anndat, anndat_multiome, label, embed, KNN_n=5):
    """ 
    Function to map results in pseudo-bulk data back to original single-cell data
    according to the cells involved in each pseudo-bulk

    Arguments
    --------------
        anndat: original single-cell AnnData object

        anndat_multiome: multiome AnnData object

        label: column name in anndat_multiome.obs to be mapped back to anndat
               the data can be either categorical or numerical

        embed: embedding space saved in anndat.obsm to use for compensating missing cells by its neighbors

        KNN_n: number of neighbors to use for compensating missing cells

    Returns
    --------------
        anndat added with
            obs[label] mapped back from anndat_multiome.obs[label]

    """

    print('Mapping back from mini-bulk to single-cell data...')
    cell_sample = anndat_multiome.obsm['NN']
    K = cell_sample.shape[1]
    agg_sum = sp.sparse.coo_matrix((
            np.ones(cell_sample.size), (
                np.repeat(np.arange(cell_sample.shape[0]), K),
                cell_sample.reshape(-1)
            )), shape=(cell_sample.shape[0], anndat.n_obs)
        ).tocsr()

    n_bulk_appeared = agg_sum.sum(axis=0).T

    is_cate = True
    if pd.api.types.is_categorical_dtype(anndat_multiome.obs[label]):
        cluster_labels = anndat_multiome.obs[label]

        label_mtx = sp.sparse.coo_matrix((
            np.ones(cluster_labels.size), (
                np.arange(cluster_labels.shape[0]),
                cluster_labels
            )), shape=(cluster_labels.size, len(np.unique(cluster_labels)))
        ).tocsr()

    elif pd.api.types.is_numeric_dtype(anndat_multiome.obs[label]):
        label_mtx = anndat_multiome.obs[label].to_numpy()[:,np.newaxis]
        is_cate = False
    else:
        print("Please specify the label as either numeric or category using astype().")
        return

    
    sc_label_mtx = agg_sum.T @ label_mtx
    print()
    #ident_mtx = agg_sum.T @ anndat_multiome.obsm['region_ident_n']

    print("Compensating missing cells by its %s neighbors in %s..."%(KNN_n,embed))
    mask = np.ravel((sc_label_mtx.sum(axis=1) != 0))
    embed_mtx = anndat.obsm[embed][mask]
    sc_label_mtx_nonzero = sc_label_mtx[mask]
    n_bulk_appeared = n_bulk_appeared[mask]
    #ident_mtx_nonzero = ident_mtx[mask]

    neigh = KNeighborsRegressor(n_neighbors=KNN_n, weights='distance')

    if is_cate:
        sc_label_mtx_nonzero = sc_label_mtx_nonzero/sc_label_mtx_nonzero.sum(axis=1)
    else:
        sc_label_mtx_nonzero = sc_label_mtx_nonzero / n_bulk_appeared

    neigh.fit(embed_mtx, sc_label_mtx_nonzero)
    sc_label_mtx[~mask] = neigh.predict(anndat.obsm[embed][~mask])

    if is_cate:
        sc_label_mtx = sc_label_mtx/sc_label_mtx.sum(axis=1)
        final_labels = sc_label_mtx.argmax(axis=1)
        anndat.obs[label] = final_labels
        anndat.obs[label] = anndat.obs[label].astype('category')
        anndat.obsm['%s_prob'%(label)] = sc_label_mtx
        print()
        print('Following infos added to the AnnData:')
        print("\t%s labels\tobs['%s']"%(label,label))
        print("\t%s label probability matrix\tobsm['%s_prob']"%(label,label))
    else:
        sc_label_mtx[mask] = sc_label_mtx_nonzero
        final_labels = sc_label_mtx.reshape(-1)
        anndat.obs.loc[:,label] = final_labels
        print()
        print('Following infos added to the AnnData:')
        print("\t%s labels\tobs['%s']"%(label,label))

    return anndat

