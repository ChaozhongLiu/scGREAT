# scGREAT main result replication
This folder stores codes and related files to replicate the scGREAT paper results.

## Data Availability
- 10X Multiome PBMC: [10X website](https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k). Downnload the files into ``10XMultiome/raw``
- SHARE-seq mouse brain: [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203). Download all files into ``share_seq/raw``
- SNARE-seq mouse skin: [GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074). Download all files into ``snare_seq/raw``

## Data Preprocessing
### 10X Multiome
1. Run scripts in ``10XMultiome/seurat2H5AD/`` to get:
  - H5AD files of scRNA-seq and scATAC-seq data
  - UMAP coordinates
  - Weighted nearest neighbor graph
2. Run ``data_preparation.ipynb`` to generate all data needed for replicating the results.


### SHARE-seq and SNARE-seq
1. Run scripts in ``sh(n)are_seq/data2h5ad.ipynb`` to get:
  - Preprocessed H5AD files of scRNA-seq and scATAC-seq data

2. Run ``data_preparation.ipynb`` to generate all data needed for replicating the results.

## Replication of each figure and supplementary materials
Please refer to the codes and Jupyter notebooks in each folder to replicate the results.
- Figure 1:  
  - ``time.*.py`` for estimating the computational time
  - ``time_plot.R`` for ploting the time consumption summary
  - You can also refer to the raw results saved in ``supplementary/``
  
  
- Figure 2:  
  - ``10X.ipynb`` for results in 10X Multiome PBMC dataset.  
  - ``SHARE_seq.ipynb`` for results in SHARE-seq mouse brain dataset
  - ``SNARE_seq.ipynb`` for results in SNARE-seq mouse skin dataset
  - ``plots.R`` to produce the summary plots shown in the manuscript
  - ``Supplementary_examples.ipynb`` to replicate the supplementary examples shown in the manuscript
  
- Figure 3:
  - ``fig3.unlabeled_analysis.ipynb`` to perform the trajectory analysis on CD4+ T cells
  - ``plots.R`` to replicate the plots in manuscript
  - you can also refer to the homer results saved in ``supplementary/``
  
  
* package name related code modification
