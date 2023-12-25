import torch
from torch.nn.functional import normalize

# Adapted from: https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
def cos_sim(a: torch.tensor, b: torch.tensor):  
    """
    Compute cosine similarity of 2 sets of vectors

    Parameters:
    a: torch.tensor, shape: [m, num_features]
    b: torch.tensor, shape: [n, num_features]

    Returns: torch.tensor, shape: [m, n]
    """
    
    return normalize(a, dim=1) @ normalize(b, dim=1).transpose(0, 1)

# Adapted from: https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
def euc_sim(a: torch.tensor, b: torch.tensor):  
    """
    Compute euclidean similarity of 2 sets of vectors

    Parameters:
    a: torch.tensor, shape: [m, num_features]
    b: torch.tensor, shape: [n, num_features]

    Returns: torch.tensor, shape: [m, n]
    """
    
    return - (a**2).sum(dim=1)[:, None] - (b**2).sum(dim=1)[None, :] + 2 * a @ b.transpose(0, 1)

def euc_sim_2(a: torch.tensor, b: torch.tensor):  
    """
    Compute euclidean similarity of 2 sets of vectors

    Parameters:
    a: torch.tensor, shape: [m, num_features]
    b: torch.tensor, shape: [n, num_features]

    Returns: torch.tensor, shape: [m, n]
    """
    
    return - (a[:, None] - b[None, :]).square().sum(dim=-1)

def main():
    from math import isclose
    import time

    device = 'cuda'
    m = 10000
    n = m // 5
    num_features = 150

    a = torch.rand(size=(m, num_features), device=device)
    b = torch.rand(size=(n, num_features), device=device)

    start_time = time.time()
    res_cos = cos_sim(a, b)
    print(-start_time + time.time())

    start_time = time.time()
    res_euc = euc_sim(a, b)
    print(-start_time + time.time())
    
    start_time = time.time()
    res_euc_2 = euc_sim_2(a, b)
    print(-start_time + time.time())

    for i in range(m):
        for j in range(n):
            assert isclose((normalize(a[i,:], dim=0) * normalize(b[j,:], dim=0)).sum(), res_cos[i, j], rel_tol=1e-5)
            assert isclose(-(a[i,:] - b[j,:]).square().sum(), res_euc[i, j], rel_tol=1e-4)
            assert isclose(res_euc[i, j], res_euc_2[i, j], rel_tol=1e-4)

# ========================
# SCRIPT ENTRY
# ========================
if __name__ == '__main__':
    
    # run script
    main()