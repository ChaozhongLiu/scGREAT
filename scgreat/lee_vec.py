import numpy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn import utils
import random
from sys import getsizeof
import scipy.stats


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory / 1024


class Spatial_Pearson(BaseEstimator):
    """Global Spatial Pearson Statistic"""

    def __init__(self, connectivity=None, permutations=999):
        """
        Initialize a spatial pearson estimator

        Arguments
        ---------
            connectivity:   scipy.sparse matrix object
                            the connectivity structure describing the relationships
                            between observed units. Will be row-standardized. 

            permutations:   int
                            the number of permutations to conduct for inference.
                            if < 1, no permutational inference will be conducted. 

        Attributes
        ----------
            association_: numpy.ndarray (p,)
                          array containg the estimated Lee spatial pearson correlation
                          coefficients for all gene-peak pairs

            reference_distribution_: numpy.ndarray (n_permutations, p)
                          distribution of correlation matrices for randomly-shuffled
                          maps. 

            significance_: numpy.ndarray (p,)
                           permutation-based z-scores (p-values) for the probability that
                           observed correlation was more extreme than the simulated 
                           correlations.
        """
        self.connectivity = connectivity
        self.permutations = permutations
        if self.connectivity is None:
            self.connectivity = sparse.eye(Z.shape[0])

        self.standard_connectivity = sparse.csc_matrix(
            self.connectivity / (self.connectivity.sum(axis=1)[:, numpy.newaxis])
        )

        self.ctc = self.connectivity.T @ self.connectivity
        ones = numpy.ones(self.ctc.shape[0])
        self.denom = (ones.T @ self.ctc @ ones)


    def fit(self, X, Y, percent=0.1, max_RAM=16, seed=1):
        """
        bivariate spatial pearson's R based on Eq. 18 of :cite:`Lee2001`.

        Arguments
        ---------
            X       :   numpy.ndarray [n x p]
                        array containing continuous data
            Y       :   numpy.ndarray [n x p]
                        array containing continuous data

            percent: percentage of cells to shuffle during permutation.
                     For most of the time, default 0.1 is alread a good choice.
            
            seed: random seed to make the results reproducible

            max_RAM: maximum limitation of memory (Gb)


        Returns
        -------
            the fitted estimator.

        Notes
        -----
            Technical details and derivations can be found in :cite:`Lee2001`.

        """
        random.seed(seed)
        numpy.random.seed(seed)
        X = utils.check_array(X)
        Y = utils.check_array(Y)
        X = preprocessing.StandardScaler().fit_transform(X)
        Y = preprocessing.StandardScaler().fit_transform(Y)
        X[X>10] = 10.0
        X[X<-10] = -10.0
        Y[Y>10] = 10.0
        Y[Y<-10] = -10.0
        

        self.association_ = self._statistic(X, Y, self.ctc, self.denom, max_RAM)


        if self.permutations is None:
            return self
        elif self.permutations < 1:
            return self

        if self.permutations:
            self.reference_distribution_ = numpy.zeros((self.permutations, X.shape[1]))

            for i in range(self.permutations):
                itp = numpy.random.choice(numpy.arange(X.shape[0]), int(X.shape[0]*percent), replace=False)
                rnd_index = numpy.arange(X.shape[0])
                rnd_index[itp] = numpy.random.permutation(itp)
                self.reference_distribution_[i] = self._statistic(Y[rnd_index], X[rnd_index], self.ctc, self.denom, max_RAM)

            self.z_sim = (self.association_ - numpy.mean(self.reference_distribution_, axis=0)) / numpy.std(self.reference_distribution_, axis=0)

            #above = self.reference_distribution_ >= numpy.repeat(self.association_[numpy.newaxis,:],self.permutations,axis=0)  #self.association_
            #larger = above.sum(axis=0)
            #extreme = numpy.minimum(self.permutations - larger, larger)
            #self.significance_ = (extreme + 1.0) / (self.permutations + 1.0)
            self.significance_ = scipy.stats.norm.sf(numpy.abs(self.z_sim))
        return self

    @staticmethod
    def _statistic(X, Y, ctc, denom, max_RAM):

        '''
        Memory taken (Mb): 0.000008 * (2 * p * n + p)
        '''
        free_memory = min(get_memory(), max_RAM*1024)
        if (free_memory*0.8) > (0.000008 * (2 * X.shape[1] * X.shape[0] + X.shape[1])):
            out = ((Y.T @ ctc) * X.T).sum(-1) / denom
        else:
            nt = (0.000008 * (2 * X.shape[1] * X.shape[0] + X.shape[1])) / (free_memory*0.8) + 1
            nt = int(nt)
            nf = X.shape[1] // nt
            out = numpy.zeros(X.shape[1])
            for i in range(nt-1):
                out[(nf*i):(nf*(i+1))] = ((Y[:,(nf*i):(nf*(i+1))].T @ ctc) * X[:,(nf*i):(nf*(i+1))].T).sum(-1) / denom

            out[(nf*(nt-1)):] = ((Y[:,(nf*(nt-1)):].T @ ctc) * X[:,(nf*(nt-1)):].T).sum(-1) / denom

        return out


class Spatial_Pearson_Local(BaseEstimator):
    """Local Spatial Pearson Statistic"""

    def __init__(self, connectivity=None, permutations=999):
        """
        Initialize a spatial local pearson estimator

        Arguments
        ---------
            connectivity:   scipy.sparse matrix object
                            the connectivity structure describing the relationships
                            between observed units. Will be row-standardized. 

            permutations:   int
                            the number of permutations to conduct for inference.
                            if < 1, no permutational inference will be conducted.


        Attributes
        ----------
            associations_: numpy.ndarray (n_samples,)
                          array containg the estimated Lee spatial pearson correlation
                          coefficients, where element [0,1] is the spatial correlation
                          coefficient, and elements [0,0] and [1,1] are the "spatial
                          smoothing factor"

            significance_: numpy.ndarray (n,p)
                           permutation-based z-scores (p-values) for the probability that
                           observed correlation was more extreme than the simulated 
                           correlations.


        Notes
        -----
        Technical details and derivations can be found in :cite:`Lee2001`.
        """
        self.connectivity = connectivity
        self.permutations = permutations

        self.standard_connectivity = sparse.csc_matrix(
                                        self.connectivity / (self.connectivity.sum(axis=1)[:, numpy.newaxis])
                                        )

    def fit(self, X, Y, max_RAM=16, seed=1):
        """
        bivariate local pearson's R based on Eq. 22 in Lee (2001)

        Arguments
        ---------
            X       :   numpy.ndarray [n x p]
                        array containing continuous data
            Y       :   numpy.ndarray [n x p]
                        array containing continuous data
            
            seed: random seed to make the results reproducible

            max_RAM: maximum limitation of memory (Gb)

        Returns
        -------
            the fitted estimator.
        """
        random.seed(seed)
        numpy.random.seed(seed)
        X = utils.check_array(X)
        X = preprocessing.StandardScaler().fit_transform(X)

        Y = utils.check_array(Y)
        Y = preprocessing.StandardScaler().fit_transform(Y)

        X[X>10] = 10.0
        X[X<-10] = -10.0
        Y[Y>10] = 10.0
        Y[Y<-10] = -10.0

        #Z = numpy.column_stack((x, y))

        n, _ = X.shape

        self.associations_ = self._statistic(X, Y, self.standard_connectivity, max_RAM)

        if self.permutations:

            self.significance_ = numpy.empty(X.shape)

            #perm_X = numpy.empty((self.permutations, X.shape[0], X.shape[1]))
            #perm_Y = numpy.empty((self.permutations, Y.shape[1], Y.shape[0]))

            for i in range(X.shape[1]):
                perm_X = numpy.empty((self.permutations, X.shape[0]))
                perm_Y = numpy.empty((self.permutations, Y.shape[0]))
                for j in range(self.permutations):
                    rnd_index = numpy.random.permutation(numpy.arange(X.shape[0]))
                    perm_X[j] = X[rnd_index,i]
                    perm_Y[j] = Y[rnd_index,i]

                reference_distribution = numpy.transpose(
                    (numpy.transpose(perm_Y[:,:,numpy.newaxis], (0,2,1)) @ self.standard_connectivity.T), (0, 2, 1)
                    ) * (self.standard_connectivity @ perm_X[:,:,numpy.newaxis])

                above = reference_distribution.squeeze(-1) >= numpy.repeat(self.associations_[:,i][numpy.newaxis,:],self.permutations,axis=0)
                larger = above.sum(axis=0)
                extreme = numpy.minimum(larger, self.permutations - larger)
                self.significance_[:,i] = (extreme + 1.0) / (self.permutations + 1.0)


        else:
            self.reference_distribution_ = None
        return self

    @staticmethod
    def _statistic(X, Y, W, max_RAM):

        '''
        Memory taken (Mb): 0.000008 * (3 * p * n)
        '''
        free_memory = min(get_memory(), max_RAM*1024)
        if (free_memory*0.8) > (0.000008 * (3 * X.shape[1] * X.shape[0])):
            return (Y.T @ W.T).T * (W @ X)
        else:
            nt = (0.000008 * (3 * X.shape[1] * X.shape[0])) / (free_memory*0.8) + 1
            nt = int(nt)
            nf = X.shape[1] // nt
            out = numpy.zeros(X.shape)
            for i in range(nt-1):
                out[:,(nf*i):(nf*(i+1))] = (Y[:,(nf*i):(nf*(i+1))].T @ W.T).T * (W @ X[:,(nf*i):(nf*(i+1))])

            out[:,(nf*(nt-1)):] = (Y[:,(nf*(nt-1)):].T @ W.T).T * (W @ X[:,(nf*(nt-1)):])

            return out


