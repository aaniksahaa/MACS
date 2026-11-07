"""MK-MMD source weighting.

alpha = argmin over the simplex of
    || sum_i alpha_i mu(S_i) - mu(T) ||^2_H  +  lam * sum_i alpha_i log alpha_i
with mu the kernel mean embedding in one common embedding (every domain encoded by every
teacher, blocks L2-normalised and concatenated) under a multi-bandwidth RBF kernel. Writing
G_ij = <mu(S_i), mu(S_j)>, h_i = <mu(S_i), mu(T)>, c = <mu(T), mu(T)> the objective is
alpha^T G alpha - 2 alpha^T h + c (+ entropy): a convex QP in the m source weights.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

from .utils import weight_stats


def common_embedding(blocks):
    """blocks: list over encoders of [n, C_e]; L2-normalise each block, concatenate, scale by 1/sqrt(m)."""
    parts = []
    for block in blocks:
        block = np.asarray(block, dtype=np.float64)
        parts.append(block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12))
    return np.concatenate(parts, axis=1) / np.sqrt(len(parts))


def median_squared_distance(arrays, cap=2000, seed=0):
    pooled = np.concatenate(arrays, axis=0)
    if len(pooled) > cap:
        pooled = pooled[np.random.RandomState(seed).choice(len(pooled), cap, replace=False)]
    sq = euclidean_distances(pooled, pooled, squared=True)
    upper = sq[np.triu_indices_from(sq, k=1)]
    upper = upper[upper > 1e-12]
    return float(np.median(upper)) if upper.size else 1.0


def multi_kernel(x, y, gammas):
    k = np.zeros((len(x), len(y)))
    for g in gammas:
        k += rbf_kernel(x, y, gamma=g)
    return k / len(gammas)


def unbiased_mean(k):
    n = k.shape[0]
    return float((k.sum() - np.trace(k)) / (n * (n - 1)))


def gram_terms(sources, target, gammas):
    """G (m x m), h (m), c for the mixture objective; unbiased within-domain means."""
    m = len(sources)
    G = np.zeros((m, m))
    h = np.zeros(m)
    for i in range(m):
        for j in range(i, m):
            k = multi_kernel(sources[i], sources[j], gammas)
            G[i, j] = G[j, i] = unbiased_mean(k) if i == j else float(k.mean())
        h[i] = float(multi_kernel(sources[i], target, gammas).mean())
    c = unbiased_mean(multi_kernel(target, target, gammas))
    return G, h, c


def mmd2_of_mixture(alpha, G, h, c):
    alpha = np.asarray(alpha, dtype=np.float64)
    return float(alpha @ G @ alpha - 2 * alpha @ h + c)


def solve_mixture(G, h, c, lam=0.01):
    m = len(h)
    eps = 1e-12

    def objective(a):
        ent = float(np.sum(a * np.log(np.maximum(a, eps))))
        return mmd2_of_mixture(a, G, h, c) + lam * ent

    def gradient(a):
        return 2 * G @ a - 2 * h + lam * (np.log(np.maximum(a, eps)) + 1)

    best = None
    starts = [np.full(m, 1.0 / m)] + [np.eye(m)[i] * 0.9 + 0.1 / m for i in range(m)]
    for a0 in starts:
        res = minimize(objective, a0, jac=gradient, method="SLSQP",
                       bounds=[(0.0, 1.0)] * m,
                       constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1.0, "jac": lambda a: np.ones(m)}],
                       options={"maxiter": 500, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    alpha = np.clip(best.x, 0, None)
    alpha /= alpha.sum()
    return alpha, float(best.fun)


def mixture_weights(sources_common, target_common, scales, lam, seed=0):
    """sources_common: list of [n_i, D] arrays in the common embedding; returns weights + diagnostics."""
    med = median_squared_distance(sources_common + [target_common], seed=seed)
    gammas = [s / med for s in scales]
    G, h, c = gram_terms(sources_common, target_common, gammas)
    alpha, value = solve_mixture(G, h, c, lam)
    m = len(h)
    uniform = np.full(m, 1.0 / m)
    vertex = [mmd2_of_mixture(np.eye(m)[i], G, h, c) for i in range(m)]
    return alpha, {
        "objective": value,
        "mmd2_mixture": mmd2_of_mixture(alpha, G, h, c),
        "mmd2_uniform": mmd2_of_mixture(uniform, G, h, c),
        "mmd2_single_source": vertex,
        "median_squared_distance": med,
        "gammas": gammas,
        **weight_stats(alpha),
    }

