import numpy as np
import pandas as pd
import anndata as ad
import time

import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# Heatmap
def LocalL_Heatmap(anndat, pairs, ident=None, groupby=None, save=None, **kwds):

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
    



def LocalL_Heatmap_nocluster(anndat, pairs, ident=None, groupby=None, **kwds):

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

    sc.pl.heatmap(anndat_L, anndat_L.var_names,
                  groupby=groupby, **kwds)
                  #vmin=-1.0, vmax=1.0, dendrogram=True, cmap='vlag', figsize=(10,6)




# Feature Plot in UMAP
def visualize_marker(anndat_multiome, gene, peak, vmin=None, vmax=None, size=None):
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




#### Visualization ####
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def CorPairSummary(anndat_multiome, cor_cutoff=0.125):
    fig = plt.figure(figsize=(7,6), dpi=80)

    # Generate data...
    x = anndat_multiome.var.loc[anndat_multiome.uns['peaks_nearby']['genes'].to_numpy(),'std']
    y = anndat_multiome.var.loc[anndat_multiome.uns['peaks_nearby']['peaks'].to_numpy(),'std']
    s = np.abs(anndat_multiome.uns['peaks_nearby']['L'].to_numpy())
    #s[s<cor_cutoff] = 0
    c = -np.log10(anndat_multiome.uns['peaks_nearby']['L.FDR'].to_numpy())
    # Plot...
    inset = fig.add_subplot(111)
    axis = plt.scatter(x[s<cor_cutoff], y[s<cor_cutoff], s=50*s[s<cor_cutoff], c='grey', edgecolor='None', alpha=0.3)
    axis = plt.scatter(x[s>=cor_cutoff], y[s>=cor_cutoff], s=50*s[s>=cor_cutoff], c=c[s>=cor_cutoff], cmap='bwr', alpha=0.5)


    cbar = plt.colorbar(orientation="vertical", extend="both",
                       pad=0.05, shrink=1, aspect=20, format="%.3f")
    cbar.set_label(label="-log10(q)", size=15)
    #plt.legend(*plot.legend_elements("sizes", num=6))
    plt.clim(0,3)

    plt.xlabel("Gene Std")
    plt.ylabel("Peak Std")
    plt.show()





# Genome track




















