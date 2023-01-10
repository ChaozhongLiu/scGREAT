# scGREAT
[![License](https://img.shields.io/github/license/ChaozhongLiu/DyberPet.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)
![scgreat Version](https://img.shields.io/badge/scgreat-v1.0.0-green.svg)  

Single-cell Graph-based Regulatory Element Analysis Toolkit (scGREAT) takes as input the single-cell multiome data for various kinds of analyses:  
- gene-peak correlation measurement
- regulatory pair marker discovery
- regulation-based sub-clustering
- trajectory inferring
- feature module discovery
- motif enrichment analysis
- other useful functions in single-cell multiome data analysis
- ...
  
  
The core of the package is graph-based correlation measurement L adapted from geographical researches. For details, please refer to our manuscript on bioRxiv.

## Installation
scGREAT is written in Python and is available from PyPI. Note that Python version should be 3.x.

### Install from PyPI
Please run the following command. Creating new environment using conda is recommended.
```
pip3 install scgreat

# For all the functions, please also install leidenalg and minisom
pip3 install leidenalg
pip3 install minisom
```
If the above method failed, please try create a new environment using conda first then re-install the three packages above.  
If things still doesn't work, try the method below.

### Use the package locally
If there is conflicts with other packages or something unexpected happened, try download the package and usee it locally in a new conda environment.
```
# Create a new conda environment
conda create -n scgreat numpy=1.22
conda activate scgreat

# Setting up kernels for Jupyter Notebook if needed
conda install ipykernel
ipython kernel install --user --name=scgreat

# Install dependencies
conda install -c conda-forge scanpy python-igraph leidenalg
conda install -c conda-forge libpysal

# Option 1 to install minisom
git clone https://github.com/JustGlowing/minisom.git
cd minisom
python setup.py install

# Option 2 to install minisom
pip3 install minisom

# install Homer if needed
```

  
  
  
## Quick start
- To test whether the package is working without error and go through important functions in scGREAT:
  - Download the toy example data ``example/data/toy_test.h5ad``
  - Follow this [Jupyter Notebook tutorial](example/quick_start_test.ipynb)  to get familiar with the package

- To prepare your own single-cell multiome data for scGREAT, see tutorial here (incomplete).


## Manuscript-related information
- For supplementary materials, please check the ``supplementary`` folder
- For manuscript results replication, please check the ``replication`` folder
