from libpysal.weights.spatial_lag import lag_spatial as slag
import numpy as np
import random
from sklearn import preprocessing

PERMUTATIONS = 0


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory / 1024


class Moran(object):
    """Moran's I Global Autocorrelation Statistic

    Modification from https://github.com/pysal/esda

    Parameters
    ----------

        n               : int
                          number of observations

        w               : W
                          spatial weights instance

        transformation  : string
                          weights transformation,  default is row-standardized "r".
                          Other options include "B": binary,  "D":
                          doubly-standardized,  "U": untransformed
                          (general weights), "V": variance-stabilizing.

        permutations    : int
                          number of random permutations for calculation of
                          pseudo-p_values

    Attributes
    ----------
        w            : W
                       original w object

        permutations : int
                       number of permutations

        I            : array
                       value of Moran's I

        sim          : array
                       (if permutations>0)
                       vector of I values for permuted samples

        p_sim        : array
                       (if permutations>0)
                       p-value based on permutations (one-tailed)
                       null: spatial randomness
                       alternative: the observed I is extreme if
                       it is either extremely greater or extremely lower
                       than the values obtained based on permutations

    Notes
    -----
    Technical details and derivations can be found in :cite:`cliff81`.

    """

    def __init__(
        self, n, w, transformation="r", permutations=PERMUTATIONS
    ):
        #y = np.asarray(y).flatten()
        #self.y = y
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        self.n = n
        self.__moments()


    def calc_i(self, X, seed=1, max_RAM=16):
        """
        Function to calculate Moran's I of all features

        Arguments
        ---------
            X: n x p array
               log-transformed feature matrix

            seed: random seed to make the results reproducible

            max_RAM: maximum limitation of memory (Gb)

        Returns
        ---------
            the fitted estimator.

        """
        random.seed(seed)
        np.random.seed(seed)
        self.Z = preprocessing.StandardScaler().fit_transform(X)
        #self.Z = np.asarray(self.Z).flatten()
        self.Z[self.Z>10] = 10.0
        self.Z[self.Z<-10] = -10.0
        self.I = self.__calc(self.Z, max_RAM)

        if self.permutations:
            permutations = self.permutations
            sim = [
                self.__calc(np.random.permutation(self.Z), max_RAM) for i in range(permutations)
            ]

            self.sim = sim = np.array(sim)

            above = self.sim >= np.repeat(self.I[np.newaxis,:],self.permutations,axis=0)  #self.association_
            larger = above.sum(axis=0)
            extreme = np.minimum(self.permutations - larger, larger)
            self.p_sim = (extreme + 1.0) / (self.permutations + 1.0)

            #above = sim >= self.I
            #larger = above.sum()
            #if (self.permutations - larger) < larger:
            #    larger = self.permutations - larger
            #self.p_sim = (larger + 1.0) / (permutations + 1.0)


    def __moments(self):

        n = self.n
        n2 = n * n
        s1 = self.w.s1
        s0 = self.w.s0
        s2 = self.w.s2
        s02 = s0 * s0


    def __calc(self, Z, max_RAM):
        '''
        memory taken: 0.000008 * (3*n*p + 3*p)
        '''
        free_memory = min(get_memory(), max_RAM*1024)
        if (free_memory*0.8) > (0.000008 * (3 * Z.shape[1] * Z.shape[0] + 3*Z.shape[1])):
            z2ss = np.square(Z).sum(axis=0)
            zl = slag(self.w, Z)
            inum = (Z * zl).sum(axis=0)
            return self.n / self.w.s0 * (inum / z2ss)
        else:
            out = np.zeros(Z.shape[1])
            nt = (0.000008 * (3 * Z.shape[1] * Z.shape[0] + 3*Z.shape[1])) / (free_memory*0.8) + 1
            nt = int(nt)
            nf = Z.shape[1] // nt

            for i in range(nt-1):
                z2ss = np.square(self.Z[:,(nf*i):(nf*(i+1))]).sum(axis=0)
                zl = slag(self.w, Z[:,(nf*i):(nf*(i+1))])
                inum = (Z[:,(nf*i):(nf*(i+1))] * zl).sum(axis=0)
                out[(nf*i):(nf*(i+1))] = self.n / self.w.s0 * (inum / z2ss)

            z2ss = np.square(self.Z[:,(nf*(nt-1)):]).sum(axis=0)
            zl = slag(self.w, Z[:,(nf*(nt-1)):])
            inum = (Z[:,(nf*(nt-1)):] * zl).sum(axis=0)
            out[(nf*(nt-1)):] = self.n / self.w.s0 * (inum / z2ss)

            return out

        #z2ss = np.square(self.Z).sum(axis=0)
        #zl = slag(self.w, Z)
        #inum = (Z * zl).sum(axis=0)
        


