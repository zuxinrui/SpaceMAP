import numpy as np
import copy
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# _utils is from sklearn, _utils_xinrui is compiled using AOCC and with -O0 option
# from sklearn.manifold import TSNE  # necessary for _util
# from sklearn_xinrui.manifold import _utils

import numba

import scipy.sparse
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
# from scipy.spatial.distance import pdist
# from scipy.special import gamma
import scipy.special as ss
from scipy.stats import chi
from sklearn import svm

# import cuml
# from sklearn.decomposition import PCA

import umap_utils  # just used for setting Pij to be UMAP-like in some options
# from pynndescent import NNDescent
# from pynndescent.distances import named_distances as pynn_named_distances
# from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances


np.random.seed(71)
NPY_INFINITY = np.inf
SMOOTH_K_TOLERANCE = 1e-5  # 1e-5
MACHINE_EPSILON_NP = np.finfo(np.double).eps
# MACHINE_EPSILON_NP = 1e-14
MACHINE_EPSILON_TORCH = torch.finfo(torch.float32).eps  # used for Q function correction (prevent nan)
MACHINE_EPSILON_SPACE = 0.05  # used for Q function correction (prevent nan)


@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors, k):
    knn_indices = np.empty((X.shape[0], k), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[n_neighbors:(n_neighbors+k)]
        knn_indices[row] = v
    return knn_indices


def nearest_neighbors(
    X,
    n_neighbors,
    k,
    # metric='precomputed',
    verbose=False,
):
    # if metric == 'precomputed':
    # if verbose:
    #     print(ts(), "Finding Nearest Neighbors")

    # Note that this does not support sparse distance matrices yet ...
    # Compute indices of n nearest neighbors
    knn_indices = fast_knn_indices(X, n_neighbors, k)
    # knn_indices = np.argsort(X)[:, :n_neighbors]
    # Compute the nearest neighbor distances
    #   (equivalent to np.sort(X)[:,:n_neighbors])
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    # Prune any nearest neighbours that are infinite distance apart.
    disconnected_index = knn_dists == np.inf
    knn_indices[disconnected_index] = -1
    # if verbose:
    #     print(ts(), "Finished Nearest Neighbor Search")
    # else:
    #     # TODO: Hacked values for now
    #     n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    #     n_iters = max(5, int(round(np.log2(X.shape[0]))))
    #
    #     knn_search_index = NNDescent(
    #         X,
    #         n_neighbors=n_neighbors,
    #         metric=metric,
    #
    #         n_trees=n_trees,
    #         n_iters=n_iters,
    #         max_candidates=60,
    #         low_memory=True,
    #         n_jobs=-1,
    #         verbose=verbose,
    #     )
    #     knn_indices, knn_dists = knn_search_index.neighbor_graph

    return knn_indices, knn_dists


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def binary_search_sigma(distances, k, chi_concern_rate, n_iter=64, bandwidth=1.0):  # n_iter=64
    target = (1-chi_concern_rate) * np.log2(k) * bandwidth
    sigma = np.zeros(distances.shape[0], dtype=np.float32)
    last_nn = np.zeros(distances.shape[0], dtype=np.float32)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        last_nn[i] = np.min(distances[i])
        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - last_nn[i]
                if d >= 0:
                    psum += (1-chi_concern_rate) * np.exp(-(np.power(d, 2) / mid))  # exp2
                    # psum += (1 - chi_concern_rate) * np.exp(-(d / mid))  # exp1
                # else:
                #     psum += 1-chi_concern_rate

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigma[i] = mid
    return sigma, last_nn


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def binary_search_sigma_2(distances, last_nn, k, chi_concern_rate, n_iter=64, bandwidth=1.0):  # n_iter=64
    target = (1-chi_concern_rate) * np.log2(k) * bandwidth
    sigma = np.zeros(distances.shape[0], dtype=np.float32)
    # last_nn = np.zeros(distances.shape[0], dtype=np.float32)
    trans = np.sqrt(-np.log(1-chi_concern_rate))

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        # last_nn[i] = np.min(distances[i])
        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - last_nn[i]
                if d >= 0:
                    psum += np.exp(-(np.power(d+trans, 2) / mid))  # exp2
                    # psum += (1 - chi_concern_rate) * np.exp(-(d / mid))  # exp1
                # else:
                #     psum += 1-chi_concern_rate

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigma[i] = mid
    return sigma


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos, chi_concern_rate, return_dists=False, bipartite=False,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1-chi_concern_rate
            else:
                val = (1-chi_concern_rate) * np.exp(-(np.power((knn_dists[i, j] - rhos[i]), 2) / (sigmas[i])))  # exp2
                # val = (1-chi_concern_rate) * np.exp(-(knn_dists[i, j] - rhos[i]) / (sigmas[i]))  # exp1

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


@numba.njit(
    locals={
        # "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        # "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths_p2(
    knn_indices, knn_dists, sigmas, rhos, chi_concern_rate, return_dists=False, bipartite=False,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    trans = np.sqrt(-np.log(1-chi_concern_rate))

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1-chi_concern_rate
            else:
                val = np.exp(-(np.power((knn_dists[i, j] - rhos[i]) + trans, 2) / (sigmas[i])))  # exp2
                # val = (1-chi_concern_rate) * np.exp(-(knn_dists[i, j] - rhos[i]) / (sigmas[i]))  # exp1

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


'''
=============================
Overall P calculation method:
=============================
'''


def hierarchical_probability_p(x, dof, y_rate, chi_concern_rate, knn, knn2, cdf_range, start_point, verbose=False):
    # rho = np.sort(x)[:, 1]  # the nearest neighbor
    # last_nn = np.sort(x)[:, knn]  # the kth nearest neighbor
    # x_rate = cdf_range / (last_nn - rho)

    near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    rho = near_field_dists[:, 0]
    last_nn = near_field_dists[:, -1]
    x_rate = cdf_range / (last_nn - rho)
    # sigma = -(last_nn - rho) / np.log(1-chi_concern_rate)
    # sigma = -np.power((last_nn - rho)/alpha, 1/beta) / np.log(1-chi_concern_rate)

    rho_mat = np.tile(np.array([rho]).T, (1, knn))
    x_rate_mat = np.tile(np.array([x_rate]).T, (1, knn))
    # sigma_mat = np.tile(np.array([sigma]).T, (1, knn))

    # P1 (knn)
    p1 = 1 - chi_concern_rate * y_rate * chi.cdf((near_field_dists - rho_mat)*x_rate_mat + start_point, dof)
    # p1 = np.exp(-(near_field_dists - rho_mat) / sigma_mat)
    # p1 = np.exp(-np.power((near_field_dists-rho_mat)/alpha, 1/beta) / sigma_mat)
    rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')
    # P2 knn-knn2
    global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    sigmas, last_nns = binary_search_sigma(global_nn_dists, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths(global_nn_indices, global_nn_dists,
                                                       sigmas, last_nns, chi_concern_rate)
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')
    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def hierarchical_probability_p_new(x, beta, alpha, chi_concern_rate, knn, knn2, verbose=False):
    # the knn:
    near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    rho = near_field_dists[:, 0]
    last_nn = near_field_dists[:, -1]
    # shaping: (knn, 0.1) ()
    sigma_1 = -np.power((last_nn - rho)/alpha, 1/beta) / np.log(1-chi_concern_rate)

    rho_mat = np.tile(np.array([rho]).T, (1, knn))
    sigma_mat = np.tile(np.array([sigma_1]).T, (1, knn))

    p1 = np.exp(-np.power((near_field_dists-rho_mat)/alpha, 1/beta) / sigma_mat)
    p1[np.where(p1 > 1)] = 1
    rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    sigmas, last_nns = binary_search_sigma(global_nn_dists, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths(global_nn_indices, global_nn_dists,
                                                       sigmas, last_nns, chi_concern_rate)
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')
    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def hierarchical_probability_p_ultimate(x, alpha, beta, knn, knn2, chi_concern_rate, verbose=False):
    # the knn:
    near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    rho = near_field_dists[:, 0]
    last_nn = near_field_dists[:, -1]
    # shaping: (knn, 0.1) ()
    sigma_1 = -(np.power((last_nn-rho)/alpha, 1/beta))**2 / np.log(1-chi_concern_rate)

    rho_mat = np.tile(np.array([rho]).T, (1, knn))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, knn))

    p1 = np.exp(-(np.power((near_field_dists-rho_mat)/alpha, 1/beta))**2 / sigma_1_mat)
    p1[np.where(p1 > 1)] = 1

    rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)  # TODO: attention: maybe overlapped
    sigma_2, last_nns = binary_search_sigma_2(global_nn_dists, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
                                                       sigma_2, last_nns, chi_concern_rate)
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')
    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def hierarchical_probability_p_without_threshold(x, alpha, beta, knn, knn2, chi_concern_rate, verbose=False):
    # the knn:
    near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    # rho = near_field_dists[:, 0]
    last_nn = near_field_dists[:, -1]
    # shaping: (knn, 0.1) ()
    sigma_1 = -(np.power(last_nn/alpha, 1/beta))**2 / np.log(1-chi_concern_rate)

    # rho_mat = np.tile(np.array([rho]).T, (1, knn))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, knn))

    p1 = np.exp(-(np.power(near_field_dists/alpha, 1/beta))**2 / sigma_1_mat)
    p1[np.where(p1 > 1)] = 1

    rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    sigma_2, last_nns = binary_search_sigma_2(global_nn_dists, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
                                                       sigma_2, last_nns, chi_concern_rate)
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')
    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def hierarchical_probability_p_data_specific(x_ind, x_dist, knn, knn2, chi_concern_rate, d_local, alpha, beta,
                                             metric='euclidean',
                                             manual_d_local=False,
                                             verbose=False):
    # the knn:
    # near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    # rho = near_field_dists[:, 0]
    near_field_indices = x_ind[:, 1:knn]
    near_field_dists = x_dist[:, 1:knn]
    last_nn = near_field_dists[:, -1]

    if manual_d_local:
        d_esti = d_local*np.ones(x_ind.shape[0])
    else:
        d_esti = mle_intrinsic_dimension(near_field_dists, knn-1, metric=metric)
        # print('d_esti: ', d_esti.shape)
        d_esti[np.where(d_esti > 50)] = 50

    if metric == 'euclidean':
        beta_near = d_esti / 2
        alpha_near = extension_from_2d_factor_euclidean(d_esti)
    elif metric == 'cosine':
        beta_near = (d_esti - 1) / 2
        alpha_near = extension_from_2d_factor_cosine(d_esti)
    beta_trans = beta / beta_near
    alpha_trans = alpha / (alpha_near ** beta_trans)
    sigma_1 = -np.power(last_nn/alpha_trans, 2.0/beta_trans) / np.log(1-chi_concern_rate)

    # rho_mat = np.tile(np.array([rho]).    T, (1, knn))
    alpha_trans_mat = np.tile(np.array([alpha_trans]).T, (1, knn-1))
    beta_trans_mat = np.tile(np.array([beta_trans]).T, (1, knn-1))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, knn-1))

    p1 = np.exp(-(np.power(near_field_dists/alpha_trans_mat, 1/beta_trans_mat))**2 / sigma_1_mat)
    p1[np.where(p1 > 1)] = 1

    # rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    indptr = np.linspace(0, x_ind.shape[0]*(knn-1), x_ind.shape[0]+1)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    # p1 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    # )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    # global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    global_nn_indices = x_ind[:, knn:knn+knn2]
    global_nn_dists = x_dist[:, knn:knn+knn2]
    sigma_2 = binary_search_sigma_2(global_nn_dists, last_nn, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
                                                          sigma_2, last_nn, chi_concern_rate)
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x_ind.shape[0], x_ind.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')
    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def hierarchical_probability_p_data_specific_variant_knn(x_ind, x_dist, knn, knn2, chi_concern_rate, d_local, alpha, beta,
                                                         local_range, metric='euclidean',
                                             manual_d_local=False,
                                             verbose=False):
    # the knn:
    # near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    # rho = near_field_dists[:, 0]
    near_field_indices = x_ind[:, 1:knn]
    near_field_dists = x_dist[:, 1:knn]
    last_nn = near_field_dists[:, -1]

    if manual_d_local:
        d_local_esti = d_local*np.ones(x_ind.shape[0])
    else:
        d_local_esti = mle_intrinsic_dimension(near_field_dists, knn-1, metric=metric)
        # print('d_esti: ', d_esti.shape)
        d_local_esti[np.where(d_local_esti > 50)] = 50

    if metric == 'euclidean':
        print('[SpaceMAP] using metric euclidean')
        beta_near = d_local_esti / 2
        alpha_near = extension_from_2d_factor_euclidean(d_local_esti)
    elif metric == 'cosine':
        print('[SpaceMAP] using metric cosine')
        beta_near = (d_local_esti-1) / 2
        beta_near[np.where(beta_near < 1)] = 1
        alpha_near = extension_from_2d_factor_cosine(d_local_esti)
    beta_trans = beta / beta_near
    alpha_trans = alpha / (alpha_near ** beta_trans)
    sigma_1 = -np.power(last_nn/alpha_trans, 2.0/beta_trans) / np.log(1-chi_concern_rate)

    # Bayes
    # bayes_exponential_factor = np.power(last_nn / local_range, 2 / beta)  # local_range

    # rho_mat = np.tile(np.array([rho]).    T, (1, knn))
    alpha_trans_mat = np.tile(np.array([alpha_trans]).T, (1, knn-1))
    beta_trans_mat = np.tile(np.array([beta_trans]).T, (1, knn-1))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, knn-1))

    # Bayes
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn-1))

    p1 = np.exp(-(np.power(near_field_dists/alpha_trans_mat, 1/beta_trans_mat))**2 / sigma_1_mat)
    p1[np.where(p1 > 1)] = 1

    # P1 Bayes:
    # p1 = np.power(p1, bayes_exponential_factor_mat)

    # rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    indptr = np.linspace(0, x_ind.shape[0]*(knn-1), x_ind.shape[0]+1)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    # p1 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    # )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    # global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    global_nn_indices = x_ind[:, knn:knn+knn2]
    global_nn_dists = x_dist[:, knn:knn+knn2]

    # post iclr New:
    a = (-np.log(1-chi_concern_rate))**(1/2) / last_nn
    a_mat = np.tile(np.array([a]).T, (1, knn2))
    p2 = np.exp(-(a_mat*global_nn_dists)**2)
    p2[np.where(p2 > 1)] = 1
    indptr = np.linspace(0, x_ind.shape[0] * (knn - 1), x_ind.shape[0] + 1)
    cols = global_nn_indices.flatten()
    vals = p2.flatten()
    p2 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Middle field p complete.')

    # *****previous*****:
    # sigma_2 = binary_search_sigma_2(global_nn_dists, last_nn, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    # rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
    #                                                       sigma_2, last_nn, chi_concern_rate)

    # P2 Bayes:
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn2))
    # bayes_exponential_factor = bayes_exponential_factor_mat.flatten()
    # vals = np.power(vals, bayes_exponential_factor)

    # *****previous*****:
    # p2 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x_ind.shape[0], x_ind.shape[0])
    # )
    # p2.eliminate_zeros()
    # if verbose:
    #     print('[SpaceMAP] Far field p complete.')

    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def hierarchical_probability_p_bayes_varknn(x, x_ind, x_dist, near_outlier_threshold, knn2, chi_concern_rate, d_local, alpha, beta,
                                                         local_range, metric='euclidean',
                                             manual_d_local=False,
                                             verbose=False):

    candidate_indices = x_ind[:, :(x_ind.shape[1]-knn2)]
    candidate_indices_copy = candidate_indices
    candidate_indices_copy[np.where(candidate_indices == 0)] = -1
    candidate_dists = x_dist[:, :(x_ind.shape[1]-knn2)]
    # candidate_last_nn = near_field_dists[:, -1]
    local_num = np.zeros(x_ind.shape[0])
    near_field_indices = np.zeros((candidate_indices.shape[0], candidate_indices.shape[1]-1))
    near_field_dists = np.zeros((candidate_indices.shape[0], candidate_indices.shape[1]-1))
    near_field_dists_copy = np.ones((candidate_indices.shape[0], candidate_indices.shape[1]-1)) * np.inf

    onesvm = svm.OneClassSVM(nu=0.3, kernel="rbf", gamma='scale')
    for i in range(x_ind.shape[0]):
        is_local = onesvm.fit_predict(x[candidate_indices[i], :])
        count = 0
        for j in range(candidate_indices.shape[1]):
            if is_local[j] == -1:
                count += 1
            if count >= near_outlier_threshold:
                break
        local_num[i] = j
        near_field_indices[i, :(j-1)] = candidate_indices_copy[i, 1:j]
        near_field_dists[i, :(j-1)] = candidate_dists[i, 1:j]
        near_field_dists_copy[i] = near_field_dists[i]
        near_field_dists[i, (j-1):] = candidate_dists[i, j-1]
        if i % 10000 == 0 and verbose:
            print('[SpaceMAP] Fitting SVM RBF for n-near: ', i)
    near_field_indices = csr_matrix(near_field_indices)
    # near_field_dists = csr_matrix(near_field_dists)
    near_field_indices.eliminate_zeros()
    # near_field_indices.toarray()
    # near_field_dists.eliminate_zeros()
    near_field_indices[near_field_indices == -1] = 0
    indptr = near_field_indices.indptr  # csr matrix indptr
    near_field_indices = near_field_indices.data  # csr matrix col
    last_nn = near_field_dists[:, -1]
    local_num = np.int16(local_num)
    if verbose:
        print('[SpaceMAP] near field numbers: ', local_num)

    if manual_d_local:
        d_local_esti = d_local*np.ones(x_ind.shape[0])
    else:
        d_local_esti = mle_intrinsic_dimension_varknn(near_field_dists, local_num, metric=metric)
        # print('d_esti: ', d_esti.shape)
        d_local_esti[np.where(d_local_esti > 50)] = 50

    if metric == 'euclidean':
        print('[SpaceMAP] using metric euclidean')
        beta_near = d_local_esti / 2
        alpha_near = extension_from_2d_factor_euclidean(d_local_esti)
    elif metric == 'cosine':
        print('[SpaceMAP] using metric cosine')
        beta_near = (d_local_esti-1) / 2
        beta_near[np.where(beta_near < 1)] = 1
        alpha_near = extension_from_2d_factor_cosine(d_local_esti)
    beta_trans = beta / beta_near
    alpha_trans = alpha / (alpha_near ** beta_trans)
    sigma_1 = -np.power(last_nn/alpha_trans, 2.0/beta_trans) / np.log(1-chi_concern_rate)

    bayes_exponential_factor = np.power(last_nn / local_range, 2 / beta)  # local_range fixed as 3 in default settings

    # rho_mat = np.tile(np.array([rho]).    T, (1, knn))
    alpha_trans_mat = np.tile(np.array([alpha_trans]).T, (1, candidate_indices.shape[1]-1))
    beta_trans_mat = np.tile(np.array([beta_trans]).T, (1, candidate_indices.shape[1]-1))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, candidate_indices.shape[1]-1))
    bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, candidate_indices.shape[1]-1))

    p1 = np.exp(-(np.power(near_field_dists_copy/alpha_trans_mat, 1/beta_trans_mat))**2 / sigma_1_mat)
    # p1[np.where(p1 == 0)] = -1
    p1[np.where(p1 == 1)] = 0

    # P1 Bayes:
    p1 = np.power(p1, bayes_exponential_factor_mat)
    p1 = csr_matrix(p1)
    # p1.eliminate_zeros()
    # p1[np.where(p1 == -1)] = 0
    # P1 Bayes:
    # p1 = np.power(p1, bayes_exponential_factor_mat)

    # rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    # indptr = np.linspace(0, x_ind.shape[0]*(knn-1), x_ind.shape[0]+1)
    # cols = near_field_indices.flatten()
    # vals = p1.flatten()
    # p1 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    # p1 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    # )
    p1.eliminate_zeros()
    p1 = p1.data  # csr matrix val

    if verbose:
        print('p1: ', p1.shape)
        print('near: ', near_field_indices.shape)
        print('indptr: ', indptr.shape)

    p1 = scipy.sparse.csr_matrix((p1, near_field_indices, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))

    if verbose:
        print(time.ctime(time.time()), '[SpaceMAP] Near field p complete.')

    # near_field_dists_copy = csr_matrix(near_field_dists_copy)
    # near_field_dists_copy[scipy.sparse.find(near_field_dists_copy == np.inf)] = 0

    # P2 knn-knn2
    # global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    global_nn_indices = np.zeros((x_ind.shape[0], knn2))
    global_nn_dists = np.zeros((x_ind.shape[0], knn2))
    if verbose:
        print('[SpaceMAP] Constructing middle field knn graph.')
    for i in range(x_ind.shape[0]):
        global_nn_indices[i] = x_ind[i, local_num[i]:local_num[i]+knn2]
        global_nn_dists[i] = x_dist[i, local_num[i]:local_num[i]+knn2]
    if verbose:
        print('[SpaceMAP] Middle field knn graph constructed.')
    sigma_2 = binary_search_sigma_2(global_nn_dists, last_nn, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
                                                          sigma_2, last_nn, chi_concern_rate)

    # P2 Bayes:
    bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn2))
    bayes_exponential_factor = bayes_exponential_factor_mat.flatten()
    vals = np.power(vals, bayes_exponential_factor)

    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x_ind.shape[0], x_ind.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print(time.ctime(time.time()), '[SpaceMAP] Far field p complete.')
    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    print('p1: ', p1.shape)
    print('p2: ', p2.shape)
    return p1 + p2


def hierarchical_probability_p_binarysearch(x_ind, x_dist, knn, knn2, chi_concern_rate, d_local, alpha, beta,
                                                         local_range, metric='euclidean',
                                             manual_d_local=False,
                                             verbose=False):
    # the knn:
    # near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    # rho = near_field_dists[:, 0]
    near_field_indices = x_ind[:, 1:knn]
    near_field_dists = x_dist[:, 1:knn]
    last_nn = near_field_dists[:, -1]

    if manual_d_local:
        print('d-local = manual')
        d_local_esti = d_local*np.ones(x_ind.shape[0])
    else:
        print('d-local = auto')
        d_local_esti = mle_intrinsic_dimension(near_field_dists, knn-1, metric=metric)
        # print('d_esti: ', d_esti.shape)
        d_local_esti[np.where(d_local_esti > 50)] = 50

        print('d-local = ', d_local_esti)

    if metric == 'euclidean':
        print('[SpaceMAP] using metric euclidean')
        beta_near = d_local_esti / 2
        alpha_near = extension_from_2d_factor_euclidean(d_local_esti)
    elif metric == 'cosine':
        print('[SpaceMAP] using metric cosine')
        beta_near = (d_local_esti-1) / 2
        beta_near[np.where(beta_near < 1)] = 1
        alpha_near = extension_from_2d_factor_cosine(d_local_esti)
    beta_trans = beta / beta_near
    alpha_trans = alpha / (alpha_near ** beta_trans)
    sigma_1 = -np.power(last_nn/alpha_trans, 2.0/beta_trans) / np.log(1-chi_concern_rate)

    # Bayes
    # bayes_exponential_factor = np.power(last_nn / local_range, 2 / beta)  # local_range

    # rho_mat = np.tile(np.array([rho]).    T, (1, knn))
    alpha_trans_mat = np.tile(np.array([alpha_trans]).T, (1, knn-1))
    beta_trans_mat = np.tile(np.array([beta_trans]).T, (1, knn-1))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, knn-1))

    # Bayes
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn-1))

    p1 = np.exp(-(np.power(near_field_dists/alpha_trans_mat, 1/beta_trans_mat))**2 / sigma_1_mat)
    p1[np.where(p1 > 1)] = 1

    # P1 Bayes:
    # p1 = np.power(p1, bayes_exponential_factor_mat)

    # rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    indptr = np.linspace(0, x_ind.shape[0]*(knn-1), x_ind.shape[0]+1)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    # p1 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    # )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    # global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    global_nn_indices = x_ind[:, knn:knn+knn2]
    global_nn_dists = x_dist[:, knn:knn+knn2]

    # *****previous*****:
    sigma_2 = binary_search_sigma_2(global_nn_dists, last_nn, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
                                                          sigma_2, last_nn, chi_concern_rate)

    # P2 Bayes:
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn2))
    # bayes_exponential_factor = bayes_exponential_factor_mat.flatten()
    # vals = np.power(vals, bayes_exponential_factor)

    # *****previous*****:
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x_ind.shape[0], x_ind.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')

    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2, d_local_esti


def p_fixed_boundary_gaussian(x_ind, x_dist, knn, knn2, chi_concern_rate, d_local, alpha, beta,
                                                         local_range, metric='euclidean',
                                             manual_d_local=False,
                                             verbose=False):
    # the knn:
    # near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    # rho = near_field_dists[:, 0]
    near_field_indices = x_ind[:, 1:knn+knn2]
    near_field_dists = x_dist[:, 1:knn+knn2]
    last_nn = near_field_dists[:, knn]

    if manual_d_local:
        d_local_esti = d_local*np.ones(x_ind.shape[0])
    else:
        d_local_esti = mle_intrinsic_dimension(near_field_dists, knn-1, metric=metric)
        # print('d_esti: ', d_esti.shape)
        d_local_esti[np.where(d_local_esti > 50)] = 50

    if metric == 'euclidean':
        print('[SpaceMAP] using metric euclidean')
        beta_near = d_local_esti / 2
        alpha_near = extension_from_2d_factor_euclidean(d_local_esti)
    elif metric == 'cosine':
        print('[SpaceMAP] using metric cosine')
        beta_near = (d_local_esti-1) / 2
        beta_near[np.where(beta_near < 1)] = 1
        alpha_near = extension_from_2d_factor_cosine(d_local_esti)
    beta_trans = beta / beta_near
    alpha_trans = alpha / (alpha_near ** beta_trans)
    sigma_1 = -np.power(last_nn/alpha_trans, 2.0/beta_trans) / np.log(1-chi_concern_rate)

    # Bayes
    # bayes_exponential_factor = np.power(last_nn / local_range, 2 / beta)  # local_range

    # rho_mat = np.tile(np.array([rho]).    T, (1, knn))
    alpha_trans_mat = np.tile(np.array([alpha_trans]).T, (1, knn+knn2-1))
    beta_trans_mat = np.tile(np.array([beta_trans]).T, (1, knn+knn2-1))
    sigma_1_mat = np.tile(np.array([sigma_1]).T, (1, knn+knn2-1))

    # Bayes
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn-1))

    p1 = np.exp(-(np.power(near_field_dists/alpha_trans_mat, 1/beta_trans_mat))**2 / sigma_1_mat)
    p1[np.where(p1 > 1)] = 1

    # P1 Bayes:
    # p1 = np.power(p1, bayes_exponential_factor_mat)

    # rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    indptr = np.linspace(0, x_ind.shape[0]*(knn+knn2-1), x_ind.shape[0]+1)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    p1 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    # p1 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    # )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1


def binary_p(x_ind,
             x_dist,
             knn,
             knn2,
             chi_concern_rate,
             d_local,
             alpha,
             beta,
             local_range,
             metric='euclidean',
             manual_d_local=False,
             verbose=False,
             ):
    # the knn:
    # near_field_indices, near_field_dists = nearest_neighbors(x, 0, knn)
    # rho = near_field_dists[:, 0]
    near_field_indices = x_ind[:, 1:knn]
    near_field_dists = x_dist[:, 1:knn]
    last_nn = near_field_dists[:, -1]
    near_field_dists = np.zeros(near_field_dists.shape)

    # Bayes
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn-1))

    p1 = np.exp(-near_field_dists)
    p1[np.where(p1 > 1)] = 1

    # P1 Bayes:
    # p1 = np.power(p1, bayes_exponential_factor_mat)

    # rows = np.repeat(np.int16(np.linspace(0, x.shape[0]-1, x.shape[0])), repeats=knn, axis=0)
    indptr = np.linspace(0, x_ind.shape[0]*(knn-1), x_ind.shape[0]+1)
    cols = near_field_indices.flatten()
    vals = p1.flatten()
    vals = np.ones(vals.shape)
    p1 = scipy.sparse.csr_matrix((vals, cols, indptr), shape=(x_ind.shape[0], x_ind.shape[0]))
    # p1 = scipy.sparse.coo_matrix(
    #     (vals, (rows, cols)), shape=(x.shape[0], x.shape[0])
    # )
    p1.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Near field p complete.')

    # P2 knn-knn2
    # global_nn_indices, global_nn_dists = nearest_neighbors(x, knn, knn2)
    global_nn_indices = x_ind[:, knn:knn+knn2]
    global_nn_dists = x_dist[:, knn:knn+knn2]

    # *****previous*****:
    sigma_2 = binary_search_sigma_2(global_nn_dists, last_nn, knn2, chi_concern_rate, bandwidth=1.0)  # sum = log(k) * bandwidth
    rows, cols, vals, _ = compute_membership_strengths_p2(global_nn_indices, global_nn_dists,
                                                          sigma_2, last_nn, chi_concern_rate)
    vals = np.ones(vals.shape)
    # P2 Bayes:
    # bayes_exponential_factor_mat = np.tile(np.array([bayes_exponential_factor]).T, (1, knn2))
    # bayes_exponential_factor = bayes_exponential_factor_mat.flatten()
    # vals = np.power(vals, bayes_exponential_factor)

    # *****previous*****:
    p2 = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(x_ind.shape[0], x_ind.shape[0])
    )
    p2.eliminate_zeros()
    if verbose:
        print('[SpaceMAP] Far field p complete.')

    # p2 = p2.toarray()
    # p2 = np.exp(-((x-rho_mat)**2 / np.tile(np.array([sigmas]).T, (1, x.shape[0]))))
    # p2_index = np.where(p1 < (1-chi_concern_rate))
    # p1[p2_index] = p2[p2_index]
    return p1 + p2


def p_calculation(x, d_local, knn, semi_knn, n, chi_concern_rate=0.9, power=2, verbose=False):
    if verbose:
        print('[SpaceMAP] calculating p matrix...')
    y_rate = n / knn  # theoretical shape of 1-CDF
    if d_local < 10:
        x_sample = np.linspace(0, 5, 2000)  # sample to detect the ind_min and ind_max
    elif d_local < 1000:
        x_sample = np.linspace(0, 35, 5000)
    elif d_local < 2000:
        x_sample = np.linspace(30, 45, 5000)
    else:
        x_sample = np.linspace(40, 50, 5000)

    p = chi.cdf(x_sample, d_local)
    ind_min = np.where(p > 1e-4)[0][0]
    ind_max = np.where(p > (1 / y_rate))[0][0]
    start_point = x_sample[ind_min]
    cdf_range = x_sample[ind_max] - start_point

    # knn2 = 15  # global knn
    p = hierarchical_probability_p(x, d_local, y_rate, chi_concern_rate, knn, semi_knn, cdf_range, start_point)
    # power = 2
    # for i in range(x.shape[0]):
    #     p[i, :] = chi_probability_parallel(x[i, :], d_local, y_rate, chi_concern_rate,
    #     knn, cdf_range, start_point, power)  # 把peak推到远处

    # symmetrical:
    # p = p + p.T - np.multiply(p, p.T)
    # p = p * p.T
    p = (p + p.T) / 2

    # normalization:
    # non_zeros = np.ones((p.shape[0], p.shape[0])) - np.eye(p.shape[0])
    # p *= non_zeros
    # p = p / np.sum(p)
    print('[SpaceMAP] p matrix calculation complete!')
    return p


def p_calculation_new(x, beta, alpha, knn, semi_knn, chi_concern_rate=0.9, power=2, verbose=False):
    if verbose:
        print('[SpaceMAP] calculating p matrix...')

    # knn2 = 15  # global knn
    p = hierarchical_probability_p_new(x, beta, alpha, chi_concern_rate, knn, semi_knn, verbose=verbose)

    # symmetrical:
    # p = p + p.T - np.multiply(p, p.T)
    # p = p * p.T
    p = (p + p.T) / 2

    # normalization:
    # non_zeros = np.ones((p.shape[0], p.shape[0])) - np.eye(p.shape[0])
    # p *= non_zeros
    # p = p / np.sum(p)
    print('[SpaceMAP] p matrix calculation complete!')
    return p


def p_calculation_ultimate(x,
                           x_ind,
                           x_dists,
                           knn,
                           semi_knn,
                           chi_concern_rate,
                           d_local,
                           alpha,
                           beta,
                           local_range,
                           use_manual_d_local=False,
                           metric='euclidean',
                           verbose=False,
                           symmetric='average',
                           pij='spacemap',
                           ):

    if verbose:
        print('[SpaceMAP] calculating p matrix...')

    if pij == 'spacemap' or pij == 'binary':  # Original SpaceMAP implementation

        p, d_local_esti = hierarchical_probability_p_binarysearch(x_ind, x_dists, knn, semi_knn, chi_concern_rate,
                                                                  d_local, alpha, beta, local_range, metric=metric,
                                                                  manual_d_local=use_manual_d_local,
                                                                  verbose=verbose)

    else:
        raise TypeError('[SpaceMAP] no such pij')

    # average:
    if symmetric == 'average':
        p = (p + p.T) / 2

    elif symmetric == 'union':
        # p = p + p.T - p.dot(p.T)
        # union:
        transpose = p.transpose()
        prod_matrix = p.multiply(transpose)
        p = (
                p + transpose - prod_matrix
        )

    elif symmetric == 'none':
        print('[SpaceMAP] No symmetric method')

    else:
        raise TypeError('[SpaceMAP] no such symmetric method')
    print('[SpaceMAP] p matrix calculation complete!')

    return p, d_local_esti


'''
=============================
Overall Q calculation method:
=============================
'''


def chi_exp2_ext(x, dim, factor, sigma):
    eps = torch.tensor([MACHINE_EPSILON_SPACE]).to('cuda')
    q = torch.exp(-((x+eps)**(1/(dim-1))/factor**(1/(dim-1)))/sigma)
    return q


def chi_exp_ext(x, dim, factor, sigma, power):
    eps = torch.tensor([MACHINE_EPSILON_SPACE]).to('cuda')
    q = torch.exp(-((x+eps)**(power/(2*(dim-1)))/factor**(power/(2*(dim-1))))/sigma)
    return q


def exp_extension(x, dim, exponantial_factor, local_dist):
    eps = torch.tensor([MACHINE_EPSILON_SPACE]).to('cuda')
    # D = torch.sqrt(x).to('cuda')
    q = torch.exp(-(exponantial_factor*(x+eps)**(1/(dim-1))-local_dist))
    return q


def chi_t_dist_ext(x, dim, factor, sigma):
    # nan就是和这个eps的大小有关系！！
    eps = torch.tensor([MACHINE_EPSILON_SPACE]).to('cuda')
    q = torch.pow(1+((x+eps)**(1/(dim-1))/factor**(1/(dim-1)))/sigma, -1)
    return q


def hierarchical_probability_q(D2, d_global, local_range, factor, sigma_ext, chi_concern_rate=0.9, power=2):
    # D = torch.sqrt(D2).to('cuda')
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')
    non_zeros = torch.ones(D2.shape[0], D2.shape[0]).to('cuda') - torch.eye(D2.shape[0], D2.shape[0]).to('cuda')
    # y_rate = n / knn  # theoretical shape of 1-CDF
    # q = torch.zeros(D2.shape).to('cuda')
    # x_rate = cdf_range / local_range

    # q = 1 - chi_concern_rate * y_rate * chi.cdf(D * x_rate, 2)

    # mixed Q function:
    power = 8
    a = chi_concern_rate / local_range ** 2
    q1 = -a*D2 + 1
    # q2 = chi_exp2_ext(D, d_global, factor, sigma_ext)
    # q = torch.where(D <= local_range, q1, q2)
    # q1 = -a * D ** 2 + 1
    # q2:
    # exponential:
    q2 = chi_exp2_ext(D2, d_global, factor, sigma_ext)
    # q2 = exp_extension(D2, d_global, exp_factor, local_range)
    # t dist:
    # q2 = chi_t_dist_ext(D2, d_global, factor, sigma_ext)
    q = torch.where(D2 <= local_range**2, q1, q2)
    # q[np.where(D <= local_range)] = -a*D[np.where(D <= local_range)]**2 + 1
    # q[np.where(D > local_range)] = chi_exp2_ext(D[np.where(D > local_range)], d_global, factor, sigma_ext)

    # normalization:
    q *= non_zeros
    q = torch.maximum(q / torch.sum(q), eps)
    return q


'''
=========================================================================
                    Loss functions:
=========================================================================
'''


def pairwise_distances_torch(x):

    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    x_norm = (x ** 2).sum(1).view(-1, 1)
    # if y is not None:
    #     y_norm = (y**2).sum(1).view(1, -1)
    # else:
    y = x
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def cross_entropy(self, P, Y, q_type='tsne'):
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')
    # sum_Y = torch.sum(torch.square(Y), dim=1)
    D = pairwise_distances_torch(Y)  # 已经平方 unstable function
    D += 2*torch.eye(D.shape[0], D.shape[0]).to('cuda')

    Q = hierarchical_probability_q(D, self.d_global, local_range=self.local_range,
                                   factor=self.d_global_extension_factor,
                                   sigma_ext=self.d_global_extension_sigma,
                                   chi_concern_rate=self.chi_concern_rate)

    C1 = torch.sum(P * torch.log(torch.maximum(P, eps) / Q))
    C2 = torch.sum((1 - P) * torch.log(torch.maximum(1-P, eps) / torch.maximum(1-Q, eps)))
    C = C1 + C2
    # print('C: ', C)
    return C


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def extension_from_2d_factor(dim):
    nom = dim*np.power(np.pi, dim/2-1)
    dinom = 2*(dim-1)*np.math.factorial(dim/2-1)
    return nom / dinom


def extension_from_2d_factor_(dim):
    nom = np.sqrt(dim)*np.power(np.pi, dim/4-1)
    dinom = np.sqrt(ss.gamma(dim/2+1))
    return nom / dinom


def extension_from_2d_factor_euclidean(dim):
    nom = np.power(np.pi, dim/4-1/2)
    dinom = np.sqrt(ss.gamma(dim/2+1))
    return nom / dinom


def extension_from_2d_factor_cosine(dim):
    nom = np.sqrt(dim)*np.power(np.pi, dim/4-1/2)
    dinom = np.sqrt(ss.gamma(dim/2+1))
    return nom / dinom


def mle_intrinsic_dimension(knn_dists, k, metric='euclidean'):
    Tk = np.tile(np.array([knn_dists[:, -1]]).T, (1, k-1))
    # replace the zeros of the nearest neighbors
    knn_dists[np.where(knn_dists == 0)] = knn_dists[:, -1][np.where(knn_dists == 0)[0]]
    if metric == 'euclidean':
        # previous k-1 has problem
        return 1/(1/(k-1)*np.sum(np.log2(Tk/knn_dists[:, :(k-1)]), axis=1))
    else:
        eta = 1/(1/(k-1)*np.sum(np.log2(Tk/knn_dists[:, :(k-1)]), axis=1))
        return eta + 1


def mle_intrinsic_dimension_varknn(knn_dists, k, metric='euclidean'):
    Tk = np.tile(np.array([knn_dists[:, -1]]).T, (1, knn_dists.shape[1]))
    if metric == 'euclidean':
        return 1/(1/(k-1)*np.sum(np.log2(Tk/knn_dists), axis=1))
    else:
        eta = 1/(1/(k-1)*np.sum(np.log2(Tk/knn_dists), axis=1))
        return eta + 1


def find_ab_params(spread, min_dist, factor, beta, fit_range=20.0):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * fit_range, 300)  # fit the curve in (0,20)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(factor*(xv[xv >= min_dist] - min_dist)**(2/beta) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def find_alpha_params(spread, min_dist, factor, beta, fit_range=20.0):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, alpha):
        return 1.0 / ((1.0 + (x ** 2) / alpha) ** alpha)

    xv = np.linspace(0, spread * fit_range, 300)  # fit the curve in (0,20)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(factor*(xv[xv >= min_dist] - min_dist)**(2/beta) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params


def find_dof_params(spread, min_dist, factor, beta, fit_range=20.0):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, v):
        return 1.0 / ((1.0 + (x ** 2) / v) ** ((v+1)/2))

    xv = np.linspace(0, spread * fit_range, 300)  # fit the curve in (0,20)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(factor*(xv[xv >= min_dist] - min_dist)**(2/beta) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params




