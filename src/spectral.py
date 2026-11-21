import time
import numpy as np


def triu_indices(atlas_size=200):
    return np.triu_indices(atlas_size, k=1)


def precompute_eig(X, atlas_size=200):
    """Eigendecompose every subject's FC matrix (built from the upper triangle)."""
    N = len(X)
    tri = triu_indices(atlas_size)
    vals = np.zeros((N, atlas_size), dtype=np.float32)
    vecs = np.zeros((N, atlas_size, atlas_size), dtype=np.float32)
    for i in range(N):
        M = np.zeros((atlas_size, atlas_size), dtype=np.float64)
        M[tri] = X[i]
        M = M + M.T
        w, v = np.linalg.eigh(M)
        vals[i] = np.clip(w, 1e-6, None).astype(np.float32)
        vecs[i] = v.astype(np.float32)
    return vals, vecs


def spectral_features(vals, vecs, alpha, idx=None, atlas_size=200):
    """FC^alpha = V diag(lambda^alpha) V^T, returned as upper-triangle edge features."""
    tri = triu_indices(atlas_size)
    idx = np.arange(len(vals)) if idx is None else idx
    feats = np.zeros((len(idx), len(tri[0])), dtype=np.float32)
    for j, i in enumerate(idx):
        M = (vecs[i] * (vals[i] ** alpha)) @ vecs[i].T
        feats[j] = M[tri]
    return feats


def build_cache(X, alphas, atlas_size=200, verbose=True):
    t0 = time.time()
    vals, vecs = precompute_eig(X, atlas_size)
    if verbose:
        print(f"eigendecompositions done ({time.time() - t0:.1f}s)")
    cache = {}
    for a in alphas:
        cache[a] = spectral_features(vals, vecs, a, atlas_size=atlas_size)
        if verbose:
            print(f"  alpha={a:.1f}: {cache[a].shape} ({time.time() - t0:.1f}s)")
    return cache
