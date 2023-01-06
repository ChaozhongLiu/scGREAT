import numpy as np
import pandas as pd
import anndata as ad
import time
import multiome.graph_measure as graph_measure
from scipy import stats
#from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection


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




def FindAllMarkers(anndat_multiome, ident, permutations=0, corrct_method='bonferroni', seed=1):

    if 'Local_L_by_%s'%ident in anndat_multiome.uns.keys():
        print('Previously calculated per group L matrix found, will use it for marker discovery.')
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_names'%ident]
    else:
        print("No previously calculated per group L matrix found, will calculate the matrix with default parameters...")
        print("It's highly recommended to run the per group Local L calculation first with fine-tuned parameters.")
        anndat_multiome = graph_measure.Local_L_bygroup(anndat_multiome, ident,
                                                        I_cutoff=0.4, pct_cutoff=0.05,
                                                        dropout_rm=True,
                                                        permutations=0, seed=1)
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_names'%ident]
        print("========= Finished ========")

        '''
        print("Calculate Moran's I for each feature as a filter...")
        # calculate global moran's I
        anndat_multiome = graph_measure.Morans_I(anndat_multiome, seed=seed)

        # filter pairs by Moran's I
        print("Keep features with spatial enrichment Moran's I > 0.5")
        feature_pass = anndat_multiome.var_names[anndat_multiome.var['Morans.I']>=0.5]
        peaks_nearby = anndat_multiome.uns['peaks_nearby']
        peaks_nearby = peaks_nearby.loc[peaks_nearby['genes'].isin(feature_pass),:]
        peaks_nearby = peaks_nearby.loc[peaks_nearby['peaks'].isin(feature_pass),:]
        print("%s features left for spatial correlation calculation."%(peaks_nearby.shape[0]))

        # calculate local Lee
        print("Calculate Spatial Pearson L for each pair...")
        local_L_df, local_p_df = graph_measure.Local_L(anndat_multiome, peaks_nearby, permutations=permutations, seed=seed)

        anndat_multiome.uns['Local_L_names'] = local_L_df.columns.to_numpy()
        anndat_multiome.uns['Local_L'] = local_L_df.to_numpy()
        print("Spatial Pearson Correlation results saved in uns['Local_L']")
        anndat_multiome.uns['peaks_nearby'] = peaks_nearby
        print("uns['peaks_nearby'] updated with Moran's I.")
        '''

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
        stat_df['Frac.gene.1'], stat_df['Frac.peak.1'] = add_feature_sparsity(stat_df, anndat_multiome, group)
        stat_df_all = pd.concat([stat_df_all,stat_df])
    
    print("Completed! %.2fs past."%(time.time()-start))

    return anndat_multiome, stat_df_all





def FindMarkers(anndat_multiome, ident, group_1, group_2, corrct_method='bonferroni',seed=1):

    if 'Local_L_by_%s'%ident in anndat_multiome.uns.keys():
        print('Previously calculated per group L matrix found, will use it for marker discovery.')
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_names'%ident]
    else:
        print("No previously calculated per group L matrix found, will calculate the matrix with default parameters...")
        print("It's highly recommended to run the per group Local L calculation first with fine-tuned parameters.")
        anndat_multiome = graph_measure.Local_L_bygroup(anndat_multiome, 
                                                        groupby=ident,
                                                        groups_list=[group_1,group_2],
                                                        I_cutoff=0.4, pct_cutoff=0.05,
                                                        dropout_rm=True,
                                                        permutations=0, seed=1)
        local_L_df = pd.DataFrame(anndat_multiome.uns['Local_L_by_%s'%ident])
        local_L_df.columns = anndat_multiome.uns['Local_L_by_%s_names'%ident]
        print("========= Finished ========")

        '''
        print("Calculate Moran's I for each feature as a filter...")
        # calculate global moran's I
        anndat_multiome = graph_measure.Morans_I(anndat_multiome, seed=seed)

        # filter pairs by Moran's I
        print("Keep features with spatial enrichment Moran's I > 0.5")
        feature_pass = anndat_multiome.var_names[anndat_multiome.var['Morans.I']>=0.5]
        peaks_nearby = anndat_multiome.uns['peaks_nearby']
        peaks_nearby = peaks_nearby.loc[peaks_nearby['genes'].isin(feature_pass),:]
        peaks_nearby = peaks_nearby.loc[peaks_nearby['peaks'].isin(feature_pass),:]
        print("%s features left for spatial correlation calculation."%(peaks_nearby.shape[0]))

        # calculate local Lee
        print("Calculate Spatial Pearson L for each pair...")
        local_L_df, local_p_df= graph_measure.Local_L(anndat_multiome, peaks_nearby, permutations=permutations, seed=seed)

        anndat_multiome.uns['Local_L_names'] = local_L_df.columns.to_numpy()
        anndat_multiome.uns['Local_L'] = local_L_df.to_numpy()
        '''

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








