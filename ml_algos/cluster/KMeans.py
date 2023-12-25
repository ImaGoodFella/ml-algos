from sklearn.base import BaseEstimator, ClusterMixin
import torch

from ml_algos.utils.Similarity import euc_sim, cos_sim

# Adapted from https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
class KMeans(BaseEstimator, ClusterMixin):
    """Pytorch KMeans implementation"""

    def __init__(
            self, 
            n_cluster: int, 
            max_iter: int = 100, 
            tol: float = 0.0001, 
            verbose: int = 0, 
            mode: str ='euclidean', 
            init_method: str ='random', 
        ):

        """
        Constructor for `KMeans`

        Parameter
        ---------
        n_cluster: int
            Number of cluster
        max_iter: int
            Maximum number of iterations
        tol: float
            Tolerance
        verbose: int
            Verbosity
        mode: {'euclidean', 'cosine'}
            Type of similarity measure
        init_method: {'random', 'point', '++'}
            Type of initialization
        """

        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.init_method = init_method

        if mode == 'euclidean':
            self.sim_func = euc_sim
        elif mode == 'cosine':
            self.sim_func = cos_sim
        else:
            raise NotImplementedError(f"Unsupported similarity measure: {mode}")
        
    
    def fit(self, X : torch.tensor, y : torch.tensor = None):
        pass
    
    def predict(self, X : torch.tensor):
        pass

    def fit_predict(self, X : torch.tensor, y : torch.tensor = None):
        pass