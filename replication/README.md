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


* package name related code modification
