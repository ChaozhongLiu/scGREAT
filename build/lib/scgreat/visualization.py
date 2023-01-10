import numpy as np
import pandas as pd
import anndata as ad
import time

import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

import os
import sys
from minisom import MiniSom
from sklearn import preprocessing


# Heatmap
def LocalL_Heatmap(anndat, pairs, ident=None, groupby=None, save=None, **kwds):
    """
    Function to visualize the local L matrix by heatmap, and cluster features

    Arguments
    ---------
        anndat: AnnData object with local L matrix saved in uns

        pairs: gene-peak pair lists to visualize

        ident: if local L matrix is from Local_L_bygroup(), specify the group name to get the matrix
               if not, ignore it

        groupby: if provided, cells / pseudo-bulks will be grouped by the label name

        save: if provided, heatmap will be saved in the file path

        **kwds: other arguments for sc.pl.heatmap()


    Returns
    ---------
        Local L index heatmap with features clustered


    """
    if ident is None:
        L_mtx = anndat.uns['Local_L']
        L_mtx_name = anndat.uns['Local_L_names']
    else:
        L_mtx = anndat.uns['Local_L_by_%s'%ident]
        L_mtx_name = anndat.uns['Local_L_by_%s_names'%ident]

    if not np.all(np.isin(np.asarray(pairs), L_mtx_name)):
        print("Not all pairs listed are in the local L matrix")
        return

    index_df = pd.DataFrame({'index':np.arange(L_mtx_name.shape[0])})
    index_df.index = L_mtx_name
    idx = index_df.loc[pairs,'index'].to_numpy()

    anndat_L = ad.AnnData(
        X = L_mtx[:,idx],
        obs = anndat.obs
    )
    anndat_L.var_names = pairs

    # Determine features order in heatmap
    model = AgglomerativeClustering(n_clusters=15, affinity='euclidean', 
                                  linkage='ward', compute_distances=True)
    model = model.fit(anndat_L.X.T)
    cluster_labels = model.labels_
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    R = dendrogram(linkage_matrix, no_plot=True)
    feature_order = R['leaves']

    anndat_L = anndat_L[:,feature_order]
    if save is None:
        sc.pl.heatmap(anndat_L, anndat_L.var_names,
                      groupby=groupby, **kwds)
                      #vmin=-1.0, vmax=1.0, dendrogram=True, cmap='vlag', figsize=(10,6)
    else:
        with plt.rc_context():  # Use this to set figure params like size and dpi
            sc.pl.heatmap(anndat_L, anndat_L.var_names,
                          show=False,
                          groupby=groupby, **kwds)
            plt.savefig(save)



def LocalL_Heatmap_nocluster(anndat, pairs, ident=None, groupby=None, save=False, **kwds):

    """
    Function to visualize the local L matrix by heatmap, without clustering features

    Arguments
    ---------
        anndat: AnnData object with local L matrix saved in uns

        pairs: gene-peak pair lists to visualize

        ident: if local L matrix is from Local_L_bygroup(), specify the group name to get the matrix
               if not, ignore it

        groupby: if provided, cells / pseudo-bulks will be grouped by the label name

        save: if provided, heatmap will be saved in the file path

        **kwds: other arguments for sc.pl.heatmap()


    Returns
    ---------
        heatmap without feature clustering


    """

    if not np.all(np.isin(np.asarray(pairs), anndat.uns['Local_L_by_%s_names'%ident])):
        print("Not all pairs listed are in uns['Local_L_by_%s_names']"%ident)
        return

    index_df = pd.DataFrame({'index':np.arange(anndat.uns['Local_L_by_%s_names'%ident].shape[0])})
    index_df.index = anndat.uns['Local_L_by_%s_names'%ident]
    idx = index_df.loc[pairs,'index'].to_numpy()

    anndat_L = ad.AnnData(
        X = anndat.uns['Local_L_by_%s'%ident][:,idx],
        obs = anndat.obs
    )
    anndat_L.var_names = pairs
    anndat_L.var_names_make_unique()

    if save is None:
        sc.pl.heatmap(anndat_L, anndat_L.var_names,
                      groupby=groupby, **kwds)
                      #vmin=-1.0, vmax=1.0, dendrogram=True, cmap='vlag', figsize=(10,6)
    else:
        with plt.rc_context():  # Use this to set figure params like size and dpi
            sc.pl.heatmap(anndat_L, anndat_L.var_names,
                          show=False,
                          groupby=groupby, **kwds)
            plt.savefig(save)






# Feature Plot in UMAP
def visualize_marker(anndat_multiome, gene, peak, vmin=None, vmax=None, size=None):

    """
    Function to visualize the gene-peak pair correlation in UMAP

    Arguments
    ---------
        anndat: AnnData object with umap coordinates saved in obsm, local L matrix saved in uns

        gene: gene name

        peak: peak name

        vmin: minimum value to show

        vmax: maximum value to show

        size: dot size in UMAP


    Returns
    ---------
        UMAP colored by the correlation between gene and peak


    """
    peaks_nearby = anndat_multiome.uns['peaks_nearby']
    feature_index = np.arange(len(peaks_nearby))[(peaks_nearby['genes']==gene) & 
                                                 (peaks_nearby['peaks']==peak)][0]
    
    _, n_col = anndat_multiome.obs.shape
    anndat_sp_L = pd.DataFrame(anndat_multiome.uns['Local_L'])
    anndat_sp_L.columns = anndat_multiome.uns['Local_L_names']
    anndat_multiome.obs['%s_%s'%(gene,peak)] = anndat_sp_L['%s_%s'%(gene,peak)].to_numpy()
    #anndat_sp.obs['SpPr.%s.p_sim'%(feature_index)] = anndat_sp_L['SpPr.%s.p_sim'%(feature_index)].to_numpy()

    print('%s and %s'%(gene, peak))

    sc.pl.umap(anndat_multiome, color_map='bwr', vmin=vmin, vmax=vmax, size=size,
               color=[gene,peak,'%s_%s'%(gene,peak)])
    anndat_multiome.obs = anndat_multiome.obs.iloc[:,0:n_col]



# SOM visualization
def visualize_module(anndat, module_name, figsize=(20,20), save=None):
    """
    Function to visualize feature modules discovered with SOM

    Arguments
    ---------
        anndat: AnnData object with fitted SOM saved in uns

        module_name: name of the SOM to be plotted

        figsize: figure size (width, heihgt)

        save: provide the output file path to save the plot


    Returns
    ---------
        SOM plot showing how features are changing along the pseudo-timeline in each module
        if save is not None, a plot will be saved


    """
    som_x, som_y = anndat.uns['som_shape_%s'%module_name]
    win_map = anndat.uns['win_map_%s'%module_name]
    num_features_clus = anndat.var['Module_index_%s'%module_name].value_counts().sort_index() #.to_numpy()
    empty_ind = np.setdiff1d(np.arange(som_x*som_y),num_features_clus.index)
    empty_df = pd.DataFrame({'Module_index_%s'%module_name:0},index=empty_ind)
    num_features_clus = pd.concat([pd.DataFrame(num_features_clus), empty_df]).sort_index()

    fig, axs = plt.subplots(som_x,som_y,figsize=figsize)
    fig.suptitle('Module Discovery - %s Summary'%module_name)
    if som_x == 1:
        for y in range(som_y):
            cluster = (0,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[y].plot(series,c="gray",alpha=0.5) 
                axs[y].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = y #x*som_y+y
            axs[y].set_title(f"Cluster {cluster_number} ({num_features_clus.loc[(cluster_number),'Module_index_%s'%module_name]})")

    else:
        for x in range(som_x):
            for y in range(som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    for series in win_map[cluster]:
                        axs[cluster].plot(series,c="gray",alpha=0.5) 
                    axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                cluster_number = x*som_y+y
                axs[cluster].set_title(f"Cluster {cluster_number} ({num_features_clus.loc[(cluster_number),'Module_index_%s'%module_name]})")

    #plt.show()
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()








