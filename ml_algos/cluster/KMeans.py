from sklearn.base import BaseEstimator, ClusterMixin
import math
import torch
import numpy as np
from typing import Union

from ml_algos.utils.Similarity import euc_sim, cos_sim
from ml_algos.utils.Memory import find_optimal_splits

# Adapted from https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
class KMeans(BaseEstimator, ClusterMixin):
    """Pytorch KMeans implementation"""

    def __init__(
            self, 
            n_clusters: int = 8, 
            init: Union[str, torch.Tensor] ='k-means++', 
            n_init: Union[str, int] = 10,
            max_iter: int = 300, 
            tol: float = 1e-4, 
            verbose: int = 0, 
            random_state: Union[int, None, np.random.RandomState] = None,
            copy_x: bool = True,
            algorithm: str = "llyod"
        ):

        """
        Constructor for `KMeans`

        Parameters
        ---------
        n_clusters: int = 8, 
            Number of cluster
        init: Union(str, torch.Tensor) ='k-means++', 
            Initialization method
        n_init: Union(str, int) = 10,
            Number of initializations
        max_iter: int = 300, 
            Maximum number of iterations
        tol: float = 1e-4, 
            Tolerance
        verbose: int = 0, 
            Verbosity
        random_state: Union(int, None, np.random.RandomState) = None,
            random state
        copy_x: bool = True,
            Copy x before modifying
        algorithm: str = "llyod"
            algorithm
        """

        self.n_clusters = n_clusters
        
        if init == "random":
            self.init = self.init_random
        elif init == "k-means++":
            self.init = self.init_pp
        elif isinstance(init, torch.Tensor):
            self.init = lambda x: init
        else:
            raise NotImplementedError(f"Init method not implemented!")
        
        if isinstance(n_init, int):
            self.n_init = n_init
        elif n_init == "auto":
            self.n_init = 1 if init == "k-means++" else 10
        else:
            raise NotImplementedError(f"N_init strategy not implemented!")
        
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        if isinstance(random_state, int):
            self.rng = np.random.default_rng(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            raise NotImplementedError(f"random_state argument not implemented for type {type(random_state)}")

        self.copy_x = copy_x

        if algorithm != "llyod":
            raise NotImplemented("Algorithm not implemented!")
        
        self.sim_func = euc_sim
        
    def init_random(self, X: torch.Tensor):
        return X[self.rng.choice(X.shape[0], size=self.n_clusters, replace=False).tolist()]
    
    def init_pp(self, X: torch.Tensor):
        raise NotImplementedError(f"Init method not implemented!")
    
    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        self.fit_predict(X, y)
    
    def predict(self, X : torch.Tensor) -> torch.Tensor:
        return self.max_sim(a=X, b=self.centroids)
    
    def max_sim(self, a : torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        
        n_samples = a.shape[0]
        max_sim_i = torch.empty(n_samples, device=a.device, dtype=torch.int64)

        def get_required_memory(chunk_size):
            return chunk_size * a.shape[1] * b.shape[0] * a.element_size() + n_samples * 2 * 4

        splits = find_optimal_splits(n_samples, get_required_memory,device=a.device, safe_mode=True)
        chunk_size = math.ceil(n_samples / splits)

        for i in range(splits):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            sub_x = a[start:end]
            sub_sim = self.sim_func(sub_x, b)
            _, sub_max_sim_i = sub_sim.max(dim=-1)
            max_sim_i[start:end] = sub_max_sim_i

        return max_sim_i

    def fit_predict(self, X: torch.Tensor, y: torch.Tensor = None):
        
        device = X.device

        if y is None:
            self.centroids = self.init(X)
        else:
            self.centroids = y

        closest = None
        prev_err = None
        arranged_mask = torch.arange(self.n_clusters, device=device)[:,None]
        for i in range(self.max_iter):

            closest = self.max_sim(a=X, b=self.centroids)

            expanded_closest = closest.repeat(self.n_clusters, 1)
            mask = (expanded_closest==arranged_mask).to(X.dtype)
            c_grad = mask @ X / mask.sum(-1)[:,None]

            error = (c_grad - self.centroids).square().sum()
            if prev_err is not None and abs(error - prev_err): break
            prev_err = error

            self.centroids = c_grad

        return closest
    

def main():

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans as KMeans_sk
    import time
    
    num_runs = 100
    device = 'cuda'
    m = 10000
    num_features = 2

    n_clusters = 100
    X = torch.rand(size=(m, num_features), device=device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='random', n_init=1, max_iter=100, copy_x=False)
    kmeans_sk = KMeans_sk(n_clusters=n_clusters, random_state=42, init='random', n_init=1, max_iter=100, copy_x=False)

    start = time.time()
    for i in range(num_runs):
        kmeans_sk.fit_predict(X=X.cpu().numpy())
    duration = time.time() - start
    print(duration)

    start = time.time()
    for i in range(num_runs):
        kmeans.fit_predict(X=X)
    duration = time.time() - start
    print(duration)

    centroids = kmeans.centroids.cpu().numpy()
    data = X.cpu().numpy()

    #sns.scatterplot(x=data[:,0], y=data[:,1], hue=y.cpu().numpy(), legend=False, palette=sns.color_palette('Set3'))
    #sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color='red')
    #plt.savefig("fig.png")

# ========================
# SCRIPT ENTRY
# ========================
if __name__ == '__main__':
    
    # run script
    main()