import numpy as np
import pandas as pd


def get_gloc_from_atac_data(peaks, split_symbol):
    """
    Method to get the genomic locations (including the middle point)of peaks
    Author: Linhua Wang Linhua.Wang@bcm.edu
    https://github.com/LiuzLab/Neurips2021/blob/master/task1_utils1.py
    """
    glocs = peaks.tolist()
    glocs = [c for c in glocs if 'chr' in c]
    chrms, ranges, sts, ends, midpoints = [], [], [], [], []
    for gl in glocs:
        chrms.append(gl.split(split_symbol[0],1)[0])
        st, end = int(gl.split(split_symbol[0],1)[1].split(split_symbol[1],1)[0]), int(gl.split(split_symbol[0],1)[1].split(split_symbol[1],1)[1])
        sts.append(st)
        ends.append(end)
        #midpoints.append(int((st + end)/2))
        #ranges.append("_".join(gl.split(split_symbol[0])[1].split(split_symbol[1])))
    gloc_df = pd.DataFrame({'chrm': chrms, #'grange': ranges,
                        'start': sts, 'end': ends}, index=glocs)
                        #'midpoint': midpoints}, index=glocs)
    return gloc_df




def nearby_peaks(g_array):
    g_chr = str(g_array[2])
    g_array_n = g_array[0:2].copy().astype(float)
    op = np.minimum(p_array[:,1].squeeze(),g_array_n[1]) - np.maximum(p_array[:,0].squeeze(),g_array_n[0])
    filter_bool = (op>0) & (chr_list==g_chr)
    op_index = np.arange(op.shape[0])[filter_bool]
    return np.array(','.join(list(plist[op_index])), dtype=object)
    




def peaks_within_distance(genes, peaks, upstream, downstream, ref_gtf_fn,
                          no_intersect=True, id_type='Symbol', split_symbol=['-','-']):
    gloc_df = get_gloc_from_atac_data(peaks, split_symbol=split_symbol)
    
    ref_gtf = pd.read_csv(ref_gtf_fn, sep='\t')
    if id_type=='Symbol':
        ref_gtf = ref_gtf.loc[ref_gtf['GeneSymbol'].isin(genes)]
    elif id_type=='Ensembl':
        ref_gtf = ref_gtf.loc[ref_gtf['gene_id'].isin(genes)]
        ref_gtf['GeneSymbol'] = ref_gtf['gene_id'].copy()
    #upstream, downstream = 100000, 100000
    ref_gtf['start_exp'] = 0
    ref_gtf['end_exp'] = 0

    ref_gtf.loc[ref_gtf['Strand']=='+','start_exp'] = ref_gtf.loc[ref_gtf['Strand']=='+','start'] - upstream
    ref_gtf.loc[ref_gtf['Strand']=='-','start_exp'] = ref_gtf.loc[ref_gtf['Strand']=='-','start'] - downstream

    ref_gtf.loc[ref_gtf['Strand']=='+','end_exp'] = ref_gtf.loc[ref_gtf['Strand']=='+','end'] + downstream
    ref_gtf.loc[ref_gtf['Strand']=='-','end_exp'] = ref_gtf.loc[ref_gtf['Strand']=='-','end'] + upstream
    
    global p_array,plist,chr_list
    p_array = gloc_df[['start','end']].to_numpy()
    plist = gloc_df.index.to_numpy()
    chr_list = gloc_df['chrm'].to_numpy().astype(object)

    genes_loc = ref_gtf[['start_exp','end_exp','chr']].to_numpy()
    selected_peaks = np.apply_along_axis(nearby_peaks, axis=1, arr=genes_loc)
    
    ref_gtf['nearby_peaks'] = selected_peaks
    ref_gtf = ref_gtf.loc[~(ref_gtf['nearby_peaks']=='')]
    ref_gtf['nearby_peaks'] = ref_gtf['nearby_peaks'].str.split(',')
    
    peaks_nearby_new = pd.DataFrame(np.ones((0,6)), columns=ref_gtf.columns[[1,4,5,6,7,10]])
    for i in range(ref_gtf.shape[0]):
        n_lines = len(ref_gtf.loc[ref_gtf.index[i],'nearby_peaks'])
        peaks_nearby_new_tmp = ref_gtf.iloc[np.repeat(i,n_lines),[1,4,5,6,7,10]]
        peaks_nearby_new_tmp['nearby_peaks'] = ref_gtf.loc[ref_gtf.index[i],'nearby_peaks']

        peaks_nearby_new = pd.concat([peaks_nearby_new, peaks_nearby_new_tmp], axis=0)
        peaks_nearby_new.index = np.arange(peaks_nearby_new.shape[0])
    
    #return peaks_nearby_new

    peaks_nearby_new['midp'] = peaks_nearby_new['nearby_peaks'].apply(lambda x: \
                               int((int(x.split(split_symbol[0],1)[1].split(split_symbol[1],1)[0]) + int(x.split(split_symbol[0],1)[1].split(split_symbol[1],1)[1]))/2) )

    plus_strand = peaks_nearby_new['Strand']=='+'

    peaks_nearby_new['tss_dist'] = np.NaN
    peaks_nearby_new.loc[plus_strand,'tss_dist'] = peaks_nearby_new.loc[plus_strand,'midp'] - peaks_nearby_new.loc[plus_strand,'start']
    peaks_nearby_new.loc[~plus_strand,'tss_dist'] = peaks_nearby_new.loc[~plus_strand,'end'] - peaks_nearby_new.loc[~plus_strand,'midp']

    peaks_nearby_new['tts_dist'] = np.NaN
    peaks_nearby_new.loc[plus_strand,'tts_dist'] = peaks_nearby_new.loc[plus_strand,'midp'] - peaks_nearby_new.loc[plus_strand,'end']
    peaks_nearby_new.loc[~plus_strand,'tts_dist'] = peaks_nearby_new.loc[~plus_strand,'start'] - peaks_nearby_new.loc[~plus_strand,'midp']

    promoters = ((peaks_nearby_new['tss_dist'] >= -2000) & (peaks_nearby_new['tss_dist'] <= 0)).astype(int)
    genebodys = ((peaks_nearby_new['tss_dist'] > 0) & (peaks_nearby_new['tts_dist'] <= 0)).astype(int)

    peaks_nearby_new['pRegion'] = promoters
    peaks_nearby_new['gBody'] = genebodys
    
    peaks_nearby_final = peaks_nearby_new.loc[(peaks_nearby_new['tss_dist']>=-upstream)&
                                              (peaks_nearby_new['tts_dist']<=downstream)].copy()
    
    df = peaks_nearby_final.copy()
    df.columns = ['genes','Strand','chr','start','end','peaks','midp','tss_dist','tts_dist','pRegion','gBody']
    df = df.loc[~df.iloc[:,[0,5]].duplicated(),:].copy()
    
    if no_intersect:
        print("Remove nearby peaks if it lies on the gene body or promoter regions of other genes.")
        duplicated_peaks = df['peaks'].duplicated(keep=False)
        dup_df = df.loc[duplicated_peaks,:].copy()
        dup_df = dup_df.groupby(by='peaks')[['pRegion','gBody']].max()
        body_peaks = dup_df.index[(dup_df['pRegion']==1) | (dup_df['gBody']==1)]

        peaks_rm = (df['peaks'].isin(body_peaks)) & (df['pRegion']==0) & (df['gBody']==0)

        df = df.loc[~peaks_rm,:]
    
    df = df.iloc[:,[0,5,7,9,8,10]].copy()
    df.index = np.arange(df.shape[0])
    
    return df


    
    
