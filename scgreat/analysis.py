import numpy as np
import pandas as pd
import anndata as ad
import time

import scgreat.graph_measure as graph_measure

from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

import random
import scanpy as sc

import os
import sys
from minisom import MiniSom
from sklearn import preprocessing

#=========================================================================================
# Group marker identification
#=========================================================================================

def group_stat(local_L, group_1, group_2=None):
    groups_label = local_L['clus'].to_numpy()
    mask_1 = groups_label == group_1
    obsn1 = np.sum(mask_1)
    if group_2 is None:
        group_2 = 'others'
        mask_2 = groups_label != group_1
        obsn2 = len(mask_2) - obsn1
    else:
        mask_2 = groups_label == group_2
        obsn2 = np.sum(mask_2)

    local_L_df = local_L.iloc[:,:-1].copy()
    local_L_df['clus'] = 'others'
    local_L_df.loc[mask_1, 'clus'] = group_1
    local_L_df.loc[mask_2, 'clus'] = group_2

    mean_df = local_L_df.groupby('clus').mean() #.reset_index()
    mean_df = mean_df.loc[[group_1,group_2],:]
    std_df = local_L_df.groupby('clus').std()
    std_df = std_df.loc[[group_1,group_2],:]

    mean_df = mean_df.T
    mean_df.columns = ['Mean.1','Mean.2']

    std_df = std_df.T
    std_df.columns = ['Std.1','Std.2']

    stat_df = pd.concat([mean_df, std_df], axis=1)
    stat_df.insert(0,'name',stat_df.index)
    stat_df = stat_df.reset_index(drop=True)
    stat_df.insert(0,'group',group_1)
    stat_df['obsn.1'] = obsn1
    stat_df['obsn.2'] = obsn2
    
    return stat_df


def marker_test(local_L_df, group_1, group_2=None, corrct_method='bonferroni'):

    #stat_df.columns = ['group','name','Mean.1','Mean.2','Std.1',
    #                       'Std.2','obsn.1','obsn.2','score','p.value','p.adj']

    stat_df = group_stat(local_L_df, group_1=group_1, group_2=group_2)
    stat_df['score'] = np.NaN
    stat_df['p.value'] = np.NaN
    for i in range(stat_df.shape[0]):
        ttest = stats.ttest_ind_from_stats(
                mean1=stat_df['Mean.1'][i],
                std1=stat_df['Std.1'][i],
                nobs1=stat_df['obsn.1'][i],
                mean2=stat_df['Mean.2'][i],
                std2=stat_df['Std.2'][i],
                nobs2=stat_df['obsn.2'][i],
                equal_var=True,  # Welch's
                )
        stat_df.loc[stat_df.index[i],'score'] = ttest[0]
        stat_df.loc[stat_df.index[i],'p.value'] = ttest[1]
    if corrct_method == 'fdr':
        _, stat_df['p.adj'] = fdrcorrection(
                    stat_df['p.value'], alpha=0.05, method='indep'
                )
    elif corrct_method == 'bonferroni':
        stat_df['p.adj'] = np.minimum(stat_df['p.value'] * stat_df.shape[0], 1.0)
    else:
        print("Please select correction methods from ['fdr', 'bonferroni']!")
        stat_df['p.adj'] = np.NaN

    #stat_df['Frac.gene'], stat_df['Frac.peak'] = _add_clus_sparsity(stat_df, anndat_multiome, group)

    #stat_df_all = pd.concat([stat_df_all,stat_df])

    return stat_df



def add_feature_sparsity(stat_df, anndat_multiome, group):
    GP_G = [gp.split('_')[0] for gp in stat_df['name']]
    GP_P = [gp.split('_')[1] for gp in stat_df['name']]
    
    return anndat_multiome.var.loc[GP_G,:]['Frac.%s'%group].to_numpy(),\
           anndat_multiome.var.loc[GP_P,:]['Frac.%s'%group].to_numpy()




def FindAllMarkers(anndat_multiome, ident, corrct_method='bonferroni', seed=1):
    """
    Function to discover regulatory markers in all groups

    Arguments
    ---------
        anndat_multiome: multiome AnnData object
        
        ident: column name in anndat_multiome.obs containing group labels

        corrct_method: multi-test correction method, one of ['bonferroni', 'fdr']

        seed: random seed to make the results reproducible

    Returns
    ---------
        anndat_multiome added with
            uns['Local_L_by_IDENT']
            uns['Local_L_by_IDENT_names']

        DataFrame - Differentially regulated pairs statistical test results


    """

    if 'Local_L_by_%s'%ident in anndat_multiome.uns.keys():
        print('Previously calculated per group L matrix found, will use it for marker discovery.')
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_names'%ident]
    else:
        print("No previously calculated per group L matrix found, will calculate the matrix with default parameters...")
        print("It's highly recommended to run Local_L_bygroup() first with fine-tuned parameters.")
        anndat_multiome = graph_measure.Local_L_bygroup(anndat_multiome, ident,
                                                        I_cutoff=0.4, pct_cutoff=0.05,
                                                        dropout_rm=True,
                                                        seed=1)
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_names'%ident]
        print("========= Finished ========")


    # cluster comparison
    print("Performing statistical test for correlation differences among clusters...")
    start = time.time()
    # get clusters have at least 2 mini-bulk
    groups = anndat_multiome.obs[ident].value_counts() > 1
    groups = groups.loc[groups].index.to_list()

    local_L_df['clus'] = anndat_multiome.obs[ident].to_numpy()

    stat_df_all = pd.DataFrame(np.empty((0,13)))
    stat_df_all.columns = ['group','name','Mean.1','Mean.2','Std.1','Std.2',
                           'obsn.1','obsn.2','score','p.value','p.adj','Frac.gene.1','Frac.peak.1']

    for group in groups:
        stat_df = marker_test(local_L_df, group_1=group, group_2=None, corrct_method=corrct_method)
        try:
            stat_df['Frac.gene.1'], stat_df['Frac.peak.1'] = add_feature_sparsity(stat_df, anndat_multiome, group)
        except:
            print('Cluster %s sparsity information not found. Skip.'%group)
            stat_df['Frac.gene.1'] = 1
            stat_df['Frac.peak.1'] = 1

        stat_df_all = pd.concat([stat_df_all,stat_df])
    
    print("Completed! %.2fs past."%(time.time()-start))

    return anndat_multiome, stat_df_all





def FindMarkers(anndat_multiome, ident, group_1, group_2, corrct_method='bonferroni',seed=1):
    """
    Function to compare regulatory pairs between two group

    Arguments
    ---------
        anndat_multiome: multiome AnnData object
        
        ident: column name in anndat_multiome.obs containing group labels

        group_1: first group name in ident to compare with the second

        group_2: second group name in ident to compare with the first

        corrct_method: multi-test correction method, one of ['bonferroni', 'fdr']

        seed: random seed to make the results reproducible

    Returns
    ---------
        anndat_multiome added with
            uns['Local_L_by_IDENT_pair']
            uns['Local_L_by_IDENT_pair_names']

        DataFrame - Differentially regulated pairs statistical test results


    """

    if 'Local_L_by_%s_pair'%ident in anndat_multiome.uns.keys():
        print('Previously calculated per group L matrix found, will use it for marker discovery.')
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s_pair'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_pair_names'%ident]
    else:
        print("No previously calculated per group L matrix found, will calculate the matrix with default parameters...")
        print("It's highly recommended to run Local_L_bygroup() first with fine-tuned parameters.")
        anndat_multiome = graph_measure.Local_L_bygroup(anndat_multiome, 
                                                        groupby=ident,
                                                        groups_list=[group_1,group_2],
                                                        I_cutoff=0.4, pct_cutoff=0.05,
                                                        dropout_rm=True,
                                                        seed=1)
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s_pair'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_pair_names'%ident]
        print("========= Finished ========")


    # cluster comparison
    print("Perform statistical test for correlation differences between selected two group...")
    start = time.time()

    local_L_df['clus'] = anndat_multiome.obs[ident].to_numpy()
    stat_df = marker_test(local_L_df, group_1=group_1, group_2=group_2, corrct_method=corrct_method)
    stat_df['Frac.gene.1'], stat_df['Frac.peak.1'] = add_feature_sparsity(stat_df, anndat_multiome, group_1)
    stat_df['Frac.gene.2'], stat_df['Frac.peak.2'] = add_feature_sparsity(stat_df, anndat_multiome, group_2)

    print("Completed! %.2fs past."%(time.time()-start))

    return anndat_multiome, stat_df




def MarkerFilter(stat_df, min_pct_rna=0.1, min_pct_atac=0.05, mean_diff=1.0, p_cutoff=1e-12, plot=True):
    """
    Function to filter markers from statistical test results by sparsity, correlation difference, and p-value

    Arguments
    ---------
        stat_df: Differentially regulated pairs statistical test results
        
        min_pct_rna: percentage of cells that express the gene as sparsity cutoff

        min_pct_atac: percentage of cells that have the peak as sparsity cutoff

        mean_diff: mean correlation difference between the group and background (all other groups)

        p_cutoff: adjusted p-value cutoff

        plot: if True, return volcano plot

    Returns
    ---------
        Filtered marker list with the same columns as stat_df

        if plot==True, return volcano plot


    """

    pd.options.mode.chained_assignment = None
    stat_df.loc[:,'score.abs'] = np.abs(stat_df['score'])
    stat_df = stat_df.sort_values(by=['group','score.abs'], ascending=False)

    mask = (stat_df['Frac.gene.1']>min_pct_rna) & (stat_df['Frac.peak.1']>min_pct_atac)
    stat_df = stat_df.loc[mask,:]

    filt = ((stat_df['p.adj']<p_cutoff) &
        (np.abs(stat_df['Mean.1']-stat_df['Mean.2']) > mean_diff))

    if plot:
        plt.scatter(stat_df['Mean.1'][~filt]-stat_df['Mean.2'][~filt],
                    -np.log10(stat_df['p.adj'][~filt]), s=2, marker='o', c='grey')
        plt.scatter(stat_df['Mean.1'][filt]-stat_df['Mean.2'][filt],
                    -np.log10(stat_df['p.adj'][filt]), s=2, marker='o', c='red')
        plt.xlabel('mean_1 - mean_2')
        plt.ylabel('-log10(p.adj)')

    return stat_df.loc[filt,:]





#=========================================================================================
# Unlabeled analysis
#=========================================================================================

def RegulatoryAnalysis(anndat_multiome, binary=True, weighted=False,
                       trajectory=True,
                       leiden_resol=0.5, root=None):
    
    """
    Function for unlabeled analysis, including subclustering and trajectory inferring

    Arguments
    ---------
        anndat_multiome: multiome AnnData object with Local L matrix saved in uns['Local_L']

        binary: when performing all the analysis,
                whether to use the binarized local L matrix or not

        weighted: when performing all the analysis,
                  whether assign weights by feature variance to the local L matrix or not

        trajectory: if True, do the trajectory analysis after sub-clustering

        leiden_resol: resolution in clustering

        root: if trajectory is True,
              the first time calling the function will return a PAGA graph for users to choose the root cluster
              Then run the function again with root number to complete the trajectory analysis

    Returns
    ---------
        anndat_multiome added with
            obs['leiden']: sub-clustering labels
            obs['dpt_pseudotime']: pseudo-time points from the trajectory inferred


    """

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





#=========================================================================================
# Feature module by pseudo-time
#=========================================================================================

# SOM for gene/peak module discovery
# Ref: https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
#      https://github.com/JustGlowing/minisom/blob/master/minisom.py
def Time_Module(anndat, features, pseudotime_label, bins=20, num_iteration=5000,
                som_shape=(4,4), sigma=0.5, learning_rate=.1, module_name=None, random_seed=1):
    
    """
    Function to discover feature modules by Self-Organizing Map


    Arguments
    ---------
        anndat: AnnData object with feature matrix and trajectory information

        features: feature list of interests

        pseudotime_label: column name in anndat.obs containing pseudo-time labels
    
        bins: split the pseudo-timeline into N bins, use the average value within each bin for module discovery

        num_iteration: maximum number of iteration to optimize the SOM
        
        som_shape: (M, N) shape of the map, defines number and similarity structure of modules

        sigma: the radius of the different neighbors in the SOM

        learning_rate: optimization speed, how much weights are adjusted during each iteration

        module_name: SOM name by the user, if not provided, 1, 2, 3, 4, etc., will be used

        random_seed: random seed to make the results reproducible


    Returns
    ---------
        anndat updated with
            var['Module_index_NAME'] - module label for all features
            uns['win_map_NAME'] - fitted SOM model


    """
    features = list(set(features))
    nf = len(features)
    print("Generating pseudotime feature matrix for self-organizing map construction...")
    som_mtx = np.array(anndat[:,features].X)
    som_mtx = preprocessing.StandardScaler().fit_transform(som_mtx)
    som_mtx[som_mtx>10] = 10.0
    som_mtx[som_mtx<-10] = -10.0
    som_mtx = pd.DataFrame(som_mtx)
    som_mtx[nf] = anndat.obs[pseudotime_label].to_numpy()

    #cell_order = np.argsort(anndat.obs[pseudotime_label])
    #som_mtx = som_mtx.to_numpy()[cell_order]

    som_mtx = som_mtx.sort_values(by=nf)
    time_range = (som_mtx[nf].min(), som_mtx[nf].max())
    N_interval = bins
    interv_range = (time_range[1] - time_range[0]) / N_interval

    group_intev = pd.cut(som_mtx[nf], np.arange(time_range[0]-1e-3, 
                                          time_range[1]+interv_range,
                                          interv_range))

    data = som_mtx.groupby(group_intev).mean().dropna()
    data = data.to_numpy()[:,:-1].T
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    #som_shape = (4, 4)
    print("Start training...")
    som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', random_seed=random_seed)

    som.train_batch(data, num_iteration=num_iteration, verbose=True)

    # record module index
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    #anndat.var['Module_index'] = anndat.var['Module_index'].fillna(-1)
    # save mappings of winner nodes
    if module_name is None:
        num_exists_module = sum([i.startswith('win_map') for i in anndat.uns.keys()])
        module_name = int(num_exists_module+1)
        print("No module name given, will use integer %s as name"%module_name)

    anndat.var.loc[features, 'Module_index_%s'%module_name] = cluster_index
    anndat.uns['win_map_%s'%module_name] = som.win_map(data)
    anndat.uns['som_shape_%s'%module_name] = som_shape

    print("Module index saved in var['Module_index_%s']"%module_name)
    print("Mappings of winner nodes saved in uns['win_map_%s']"%module_name)

    return anndat



#=========================================================================================
# Motif enrichment
#=========================================================================================

def run_HOMER_motif(peaks, out_dir, prefix, ref_genome, 
                    homer_path=None, split_symbol=['-','-'], size=200):

    """
    Function to run Homer from Python script.
    It will prepare Homer required input file
    and output results in the directory specified

    Arguments
    ---------
        peaks: list or array-like, peaks of interests

        out_dir: output directory to save the results
    
        prefix: prefix of all files, folder called out_dir/homer_prefix will be created to save all the results

        ref_genome: str, reference genome name, e.g., 'hg19', 'hg38'

        homer_path: path to Homer software, if Homer already added to the PATH, argument can be ignored

        split_symbol: how peak location ID is merged
                      'chr1-12345-23456' - split_symbol=['-','-']
                      'chr1:12345-23456' - split_symbol=[':','-']
        
        size: Homer paramter, the size of the region used for motif finding


    Returns
    ---------
        Homer results saved in out_dir/homer_prefix
        Peak list DataFrame in BED format

    """

    homer_df = pd.DataFrame({'chrom': [i.split(split_symbol[0],1)[0] for i in peaks],
                             'start': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[0] for i in peaks],
                             'end': [i.split(split_symbol[0],1)[1].split(split_symbol[1])[1] for i in peaks],
                             'index': np.arange(len(peaks)),
                             'unused': '.',
                             'strand':'.'})
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, '%s.peaks.bed'%(prefix))
    print("Save peaks list in BED file format at %s"%out_file)
    homer_df.to_csv(out_file, index=False, header=False, sep='\t')

    homer_dir = os.path.join(out_dir, 'homer_%s'%(prefix))
    if not os.path.exists(homer_dir):
        print("Creat HOMER output folder at %s"%homer_dir)
        os.mkdir(homer_dir)


    if homer_path is None:
        try:
            os.system('findMotifsGenome.pl %s %s %s -size %s'%(out_file, ref_genome, homer_dir, size))
        except:
            print("HOMER run failed. The HOMER peak format DataFrame is returned.")
            return homer_df
    else:
        homer_cmd_path = os.path.join(homer_path,'bin/findMotifsGenome.pl')
        try:
            os.system('%s %s %s %s -size %s'%(homer_cmd_path, out_file, ref_genome, homer_dir, size))
        except:
            print("HOMER run failed. The HOMER peak format DataFrame is returned.")
            return homer_df

    print("HOMER finished successfully! Please check the HTML report for interesting motifs.")
    print("motif_summary can be run with the motif index for further analysis.")
    return homer_df




def motif_summary(peak_file, homer_dir, motif_index, ref_genome,
                  homer_path=None, size=200):

    """
    Function to extract related peaks from motifs of interests.


    Arguments
    ---------
        peak_file: out_dir/prefix.peaks.bed file generated in run_HOMER_motif()

        homer_dir: out_dir/homer_prefix in run_HOMER_motif() | Homer output folder
    
        motif_index: motif of interests index in homer_dir/knownResults.html

        ref_genome: str, reference genome name, e.g., 'hg19', 'hg38'

        homer_path: path to Homer software, if Homer already added to the PATH, argument can be ignored
        
        size: Homer paramter, the size of the region used for motif finding
              Keep the same as in run_HOMER_motif()


    Returns
    ---------
        Motif related peaks information saved in homer_dir/
        DataFrame containing peak list and motif matching quality information

    """

    motif_file = os.path.join(homer_dir, 'knownResults/known%s.motif'%motif_index)
    motif_out = os.path.join(homer_dir, 'motif%s.peaks.txt'%motif_index)
    if homer_path is None:
        try:
            os.system('findMotifsGenome.pl %s %s %s -find %s -size %s > %s'%(
                peak_file, ref_genome, homer_dir, motif_file, size, motif_out))
        except:
            print("HOMER run failed. Make sure homer_path is set and all file paths are right.")
            return
    else:
        homer_cmd_path = os.path.join(homer_path,'bin/findMotifsGenome.pl')
        try:
            os.system('%s %s %s %s -find %s -size %s > %s'%(
                homer_cmd_path, peak_file, ref_genome, homer_dir, motif_file, size, motif_out))
        except:
            print("HOMER run failed. Make sure homer_path is set and all file paths are right.")
            return

    print("HOMER finished successfully! Motif related peaks will be loaded and returned.")
    motif_peaks = pd.read_csv(motif_out,sep='\t')
    peak_df = pd.read_csv(peak_file, sep='\t',header=None)

    peaks = peak_df.loc[motif_peaks['PositionID'].tolist()]
    peaks = peaks[0] + '-' + peaks[1].astype(str) + '-' + peaks[2].astype(str)
    motif_peaks.insert(0, 'peaks', peaks.to_numpy())

    return motif_peaks








