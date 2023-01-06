import numpy as np
import pandas as pd
import os
import sys
from minisom import MiniSom
from sklearn import preprocessing



def run_HOMER_motif(peaks, out_dir, prefix, ref_genome, 
                    homer_path=None, split_symbol=['-','-'], size=200):

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
    #os.system('')

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




#=========================================================================================
# Feature module by pseudo-time
#=========================================================================================

# SOM for gene/peak module discovery
# Ref: https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
#      https://github.com/JustGlowing/minisom/blob/master/minisom.py
def Time_Module(anndat, features, pseudotime_label, bins=20, num_iteration=5000,
                som_shape=(4,4), sigma=0.5, learning_rate=.1, module_name=None, random_seed=1):
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






##### Visualization #####
import matplotlib.pyplot as plt


def visualize_module(anndat, module_name, figsize=(20,20), save=None):
    som_x, som_y = anndat.uns['som_shape_%s'%module_name]
    win_map = anndat.uns['win_map_%s'%module_name]
    num_features_clus = anndat.var['Module_index_%s'%module_name].value_counts().sort_index() #.to_numpy()
    empty_ind = np.setdiff1d(np.arange(som_x*som_y),num_features_clus.index)
    empty_df = pd.DataFrame({'Module_index_%s'%module_name:0},index=empty_ind)
    num_features_clus = pd.concat([pd.DataFrame(num_features_clus), empty_df]).sort_index()

    fig, axs = plt.subplots(som_x,som_y,figsize=figsize)
    fig.suptitle('Module Discovery - %s Summary'%module_name)
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
