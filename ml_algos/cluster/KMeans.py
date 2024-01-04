from sklearn.base import BaseEstimator, ClusterMixin
import math
import torch
import numpy as np
from typing import Union

from ml_algos.utils.Similarity import euc_sim, euc_dist
from ml_algos.utils.Memory import find_optimal_splits

from tqdm import tqdm

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
            algorithm: str = "llyod",
            device: Union[None, str] = None,
            safe_mode: bool = False
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
        device: str = None
            device to do computation on
        safe_mode: bool = False
            whether to allocate memory before computing chunk_size
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
        self.random_state = random_state

        self.copy_x = copy_x

        if algorithm != "llyod":
            raise NotImplemented("Algorithm not implemented!")
        
        self.device = device
        self.safe_mode = safe_mode

        self.sim_func = euc_sim
        self.dist_func = euc_dist
        
    def init_random(self, X: torch.Tensor):
        return X[self.rng.choice(X.shape[0], size=self.n_clusters, replace=False).tolist()]
    
    def init_pp(self, X: torch.Tensor):

        init = torch.empty((self.n_clusters, X.shape[1]), device=self.compute_device)
        init[0,:] = X[torch.randint(X.shape[0], [1]),:]

        r = torch.distributions.uniform.Uniform(0, 1)
        for i in tqdm(range(1, self.n_clusters), disable=self.verbose<1):
            D2 = self.dist_func(init[:i,:], X).amin(dim=0)
            probs = D2 / torch.sum(D2)
            # https://github.com/pytorch/pytorch/issues/30968 and why I do not just use torch.multinomial(probabs, 1)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = X[torch.searchsorted(cumprobs, r.sample([1]).to(self.compute_device))]

        return init
    
    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        self.fit_predict(X, y)
    
    def predict(self, X : torch.Tensor) -> torch.Tensor:
        return self.get_closest_clusters(X=X, centroids=self.centroids)
        
    def get_closest_clusters(self, X : torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        
        centroids = centroids.to(self.compute_device)
        n_samples = X.shape[0]
        max_sim_i = torch.empty(n_samples, device=self.compute_device, dtype=torch.int64)

        def get_required_memory(chunk_size):
            return chunk_size * X.shape[1] * centroids.shape[0] * X.element_size() + n_samples * 2 * 4

        splits = find_optimal_splits(n_samples, get_required_memory, device=self.compute_device, safe_mode=False)
        chunk_size = math.ceil(n_samples / splits)

        for i in tqdm(range(splits), disable=self.verbose<1):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            sub_x = X[start:end].to(self.compute_device)
            sub_sim = self.sim_func(sub_x, centroids)
            _, sub_max_sim_i = sub_sim.max(dim=-1)
            max_sim_i[start:end] = sub_max_sim_i

        return max_sim_i.to(X.device)

    def score(self, X: torch.Tensor):
        return - self.compute_error(X, self.centroids, self.predict(X)).to(X.device)
    
    def compute_error(self, X: torch.Tensor, centroids: torch.Tensor, closest: torch.Tensor):
        return (X - centroids[self.get_closest_clusters(X=X, centroids=centroids)]).square().sum()
    
    def compute_new_clusters(self, X: torch.Tensor, closest: torch.Tensor, arranged_mask: torch.Tensor):
        
        n_samples = X.shape[0]

        new_centroids = torch.zeros((self.n_clusters, X.shape[1]), device=self.compute_device, dtype=X.dtype)
        cluster_size = torch.zeros(self.n_clusters, device=self.compute_device, dtype=X.dtype)

        def get_required_memory(chunk_size):
            return chunk_size * X.shape[1] * self.n_clusters * closest.element_size() * 8

        splits = find_optimal_splits(n_samples, get_required_memory, device=self.compute_device, safe_mode=False)
        chunk_size = math.ceil(n_samples / splits)
        
        for i in tqdm(range(splits), disable=self.verbose<1):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            expanded_closest = closest[start:end].repeat(self.n_clusters, 1)
            mask = (expanded_closest==arranged_mask).to(X.dtype)
            new_centroids += mask @ X[start:end]
            cluster_size += mask.sum(-1)

        new_centroids /= cluster_size[:,None]
        return new_centroids

    def fit_predict(self, X: torch.Tensor, y: torch.Tensor = None):
        self.rng = np.random.default_rng(self.random_state)
        
        self.compute_device = X.device

        self.mean = X.mean(dim=0)
        if self.copy_x:
            X = X.clone() - self.mean
        else:
            X -= self.mean

        best_error = torch.tensor(float("inf"), device=X.device)
        best_centroids = None
        best_closest = None

        for i in range(self.n_init):
            if y is None:
                centroids = self.init(X)
            else:
                centroids = y

            closest = None
            prev_err = None
            arranged_mask = torch.arange(self.n_clusters, device=X.device)[:,None]
            for i in range(self.max_iter):

                closest = self.get_closest_clusters(X=X, centroids=centroids)
                c_grad = self.compute_new_clusters(X=X, closest=closest, arranged_mask=arranged_mask)               
                expanded_closest = closest.repeat(self.n_clusters, 1)
                mask = (expanded_closest==arranged_mask).to(X.dtype)
                c_grad = mask @ X / mask.sum(-1)[:,None]
                
                error = (c_grad - centroids).square().sum()
                if prev_err is not None and abs(error - prev_err) < self.tol: break
                prev_err = error

                centroids = c_grad

            error = self.compute_error(X, centroids, closest)
            if error < best_error:
                best_error = error
                best_centroids = centroids
                best_closest = closest

        self.centroids = best_centroids + self.mean
        if not self.copy_x: X += self.mean

        return best_closest
    

def main():

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans as KMeans_sk
    import time
    
    num_runs = 1
    device = 'cuda'
    compute_device = 'cuda'
    m = 100000
    num_features = 10000

    n_clusters = 1000
    X = torch.rand(size=(m, num_features), device=device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='random', n_init=1, max_iter=100, copy_x=False, verbose=0, device=compute_device)
    kmeans_sk = KMeans_sk(n_clusters=n_clusters, random_state=42, init='random', n_init=1, max_iter=100, copy_x=False)

    start = time.time()
    for i in range(num_runs):
        y = kmeans.fit_predict(X=X)
    duration = time.time() - start
    print(kmeans.score(X=X))
    print(duration)

    X_numpy = X.cpu().numpy()
    start = time.time()
    for i in range(num_runs):
        kmeans_sk.fit(X=X_numpy)
    duration = time.time() - start
    print(kmeans_sk.score(X=X_numpy))
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