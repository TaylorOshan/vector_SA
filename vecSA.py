class VecMoran:
    """Moran's I Global Autocorrelation Statistic
    Parameters
    ----------
    y               : array
                      variable measured across n spatial units
    w               : W
                      spatial weights instance
    transformation  : string
                      weights transformation,  default is row-standardized "r".
                      Other options include "B": binary,  "D":
                      doubly-standardized,  "U": untransformed
                      (general weights), "V": variance-stabilizing.
    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    I            : float
                   value of Moran's I
    """


    def __init__(self, origins, destinations, transformation="r"):
        #Y would become either origins or destination locations (x,y)   
        self.origins = origins
        self.dests = destinations
        
        self.n = len(origins)
        xObar = self.origins[:,0].mean()
        yObar = self.origins[:,1].mean()
        xDbar = self.dests[:,0].mean()
        yDbar = self.dests[:,1].mean()
        u = (y[:,3] - y[:,1]) - (xDbar - xObar)
        v = (y[:,4] - y[:,2]) - (yDbar - yObar)
        z = np.outer(u, u) + np.outer(v,v)
        wo = self.wO(vectors = y)
        wd = self.wD(vectors = y)
        
        self.IO = n / np.sum(wo) * np.sum(wo*z) / sum(u**2 +v**2)
        self.ID = n / np.sum(wd) * np.sum(wd*z) / sum(u**2 +v**2)
        
        #print IO, ID, u, v, z
    
    #def __moments(self):
    #    zO = self.dests - self.dests.mean()
    #    zD = self.origins - self.origins.mean()
    #    self.zO = zO
    #    self.zD = zD
    #    self.zO2ss = sum(zO * zO)
    #    self.zD2ss = sum(
    #    self.EI = -1. / (self.n - 1)
    #    s1 = self.w.s1
    #    s0 = self.w.s0
    #    s2 = self.w.s2
    #    s02 = s0 * s0

        
    def wO(self, vectors, beta = -1.5): 
        if vectors == None:
            vectors = self.y
        origin_W = dist.squareform(dist.pdist(vectors[:,1:3])) ** (beta)
        np.fill_diagonal(origin_W, 0)
        return origin_W
    
    def wD(self, vectors, beta = -1.5):
        dest_W = dist.squareform(dist.pdist(vectors[:,3:5])) ** (beta)
        np.fill_diagonal(dest_W, 0)
        return dest_W

if __name__ == '__main__':
    vecs_A = np.array([[1, 55, 60, 100, 500], [2, 60, 55, 105, 501], [3, 500, 55, 155, 500], [4, 505, 60, 160, 500], [5, 105, 950, 105, 500], [6, 155, 950, 155, 499]])
    origins = np.array([vecs_A[:,1], vecs_A[:,2]]).transpose()
    dests = np.array([vecs_A[:,3], vecs_A[:,4]]).transpose()
    V = VecMoran(origins, dests)
