import numpy as np
import scipy.stats as stats
from pysal.weights.Distance import DistanceBand

PERMUTATIONS = 999999

class VecMoran:
    """Moran's I Global Autocorrelation Statistic For Vectors
    Parameters
    ----------
    y               : array
                      variable measured across n origin-destination vectors
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
    two_tailed      : boolean
                      If True (default) analytical p-values for Moran are two
                      tailed, otherwise if False, they are one-tailed.
    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    permutations : int
                   number of permutations
    I            : float
                   value of vector-based Moran's I
    EI           : float
                   expected value under normality assumption
    VI_norm      : float
                   variance of I under normality assumption
    seI_norm     : float
                   standard deviation of I under normality assumption
    z_norm       : float
                   z-value of I under normality assumption
    p_norm       : float
                   p-value of I under normality assumption
    VI_rand      : float
                   variance of I under randomization assumption
    seI_rand     : float
                   standard deviation of I under randomization assumption
    z_rand       : float
                   z-value of I under randomization assumption
    p_rand       : float
                   p-value of I under randomization assumption
    two_tailed   : boolean
                   If True p_norm and p_rand are two-tailed, otherwise they
                   are one-tailed.
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
    EI_sim       : float
                   (if permutations>0)
                   average value of I from permutations
    VI_sim       : float
                   (if permutations>0)
                   variance of I from permutations
    seI_sim      : float
                   (if permutations>0)
                   standard deviation of I under permutations.
    z_sim        : float
                   (if permutations>0)
                   standardized I based on permutations
    p_z_sim      : float
                   (if permutations>0)
                   p-value based on standard normal approximation from
                   permutations
    """


    def __init__(self, y, w, transformation="U", permutations=PERMUTATIONS,
        two_tailed=True):
        self.y = y
        self.o = y[:, 1:3]
        self.d = y[:, 3:5]
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        self.__moments()
        self.I = self.__calc(self.z)

        self.z_norm = (self.I - self.EI) / self.seI_norm
        self.z_rand = (self.I - self.EI) / self.seI_rand

        if self.z_norm > 0:
            self.p_norm = 1 - stats.norm.cdf(self.z_norm)
            self.p_rand = 1 - stats.norm.cdf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)

        if two_tailed:
            self.p_norm *= 2.
            self.p_rand *= 2.

        if permutations:
            sim = [self.__calc(np.random.permutation(self.z))
                   for i in xrange(permutations)]
            self.sim = sim = np.array(sim)
            above = sim >= self.I
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.EI_sim = sim.sum() / permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim ** 2
            self.z_sim = (self.I - self.EI_sim) / self.seI_sim
            if self.z_sim > 0:
                self.p_z_sim = 1 - stats.norm.cdf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

    def __moments(self):
        self.n = len(self.y)
        xObar = self.o[:,0].mean()
        yObar = self.o[:,1].mean()
        xDbar = self.d[:,0].mean()
        yDbar = self.d[:,1].mean()
        u = (self.y[:,3] - self.y[:,1]) - (xDbar - xObar)
        v = (self.y[:,4] - self.y[:,2]) - (yDbar - yObar)
        z = np.outer(u, u) + np.outer(v,v)
        self.z = z
        self.uv2ss = np.sum(np.dot(u,u) + np.dot(v,v))
        self.EI = -1. / (self.n - 1)
        n = self.n
        s1 = self.w.s1
        W = self.w.s0
        s2 = self.w.s2

        v_num = n * n * s1 - n * s2 + 3 * W * W
        v_den = (n - 1) * (n + 1) * W * W
        self.VI_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
        self.seI_norm = self.VI_norm ** (1 / 2.)

        a2 = np.sum(np.dot(u, u))/n
        b2 = np.sum(np.dot(v, v))/n
        m2 = a2 + b2
        a4 = np.sum(np.dot(np.dot(u, u), np.dot(u, u)))/n
        b4 = np.sum(np.dot(np.dot(v, u), np.dot(v, v)))/n
        n1 = a2**2*((n**2 - 3*n + 3)*s1-n*s2 + 3*W**2)
        n2 = a4*((n**2 - n)*s1 - 2*n*s2 + 6*W**2)
        n3 = b2**2*((n**2 - 3*n + 3)*s1 - n*s2 + 3*W**2)
        n4 = b4*((n**2 - n)*s1 - 2*n*s2 + 6*W**2)
        d = (n - 1)*(n - 2)*(n - 3)
        self.VI_rand = 1/(W**2*m2**2) * \
                  ((n1 - n2)/d + (n3 - n4)/d) + \
                  ((a2*b2) - m2**2)/(m2**2*(n - 1)**2)
        self.seI_rand = self.VI_rand ** (1 / 2.)

    def __calc(self, z):
        zl = self.slag(self.w, z)
        inum = np.sum(zl)
        return self.n / self.w.s0 * inum / self.uv2ss

    def slag(self, w, y):
        """
        Spatial lag operator.
        If w is row standardized, returns the average of each observation's neighbors;
        if not, returns the weighted sum of each observation's neighbors.
        Parameters
        ----------
        w                   : W
                              object
        y                   : array
                              numpy array with dimensionality conforming to w (see examples)
        Returns
        -------
        wy                  : array
                              array of numeric values for the spatial lag
        """
        return np.array(w.sparse.todense()) * y

if __name__ == '__main__':
    vecs = np.array([[1, 55, 60, 100, 500], [2, 60, 55, 105, 501], [3, 500, 55, 155, 500], [4, 505, 60, 160, 500], [5, 105, 950, 105, 500], [6, 155, 950, 155, 499]])
    origins = vecs[:, 1:3]
    dests = vecs[:, 3:5]
    wo = DistanceBand(origins, threshold=9999, alpha=-1.5, binary=False)
    wd = DistanceBand(dests, threshold=9999, alpha=-1.5, binary=False)
    vmo = VecMoran(vecs, wo)
    vmd = VecMoran(vecs, wd)
    print vmo.I
    print vmo.p_rand
    print vmo.p_z_sim
    print vmd.I
    print vmd.p_rand
    print vmd.p_z_sim


'''
    def w(self, vectors, beta = -1.5):
        w = dist.squareform(dist.pdist(vectors)) ** (beta)
        np.fill_diagonal(w, 0)
        return w
'''
