import numpy as np
import numba
from umap_utils.utils import tau_rand_int
from _utils import find_ab_params


@numba.njit()
def clip(val):

    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _spacemap_optimization_symmetric_t_dist(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    negative_sampling,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            w = weight[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0

                grad_coeff_repul = 2.0 * gamma * b
                grad_coeff_repul /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                )

                grad_coeff += grad_coeff_repul

            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * w * (current[d] - other[d]))

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * b
                    grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                    )
                    # grad_coeff = -(-2*a*np.sqrt(dist_squared)+1)/a*pow(dist_squared, 1.5)

                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (1-w) * (current[d] - other[d]))
                    else:
                        grad_d = 2.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def _spacemap_optimization_asymmetric_t_dist(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    c_2b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    negative_sampling,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            w = weight[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * c_2b[i] * pow(dist_squared, b - 1.0)
                grad_coeff /= a * c_2b[i] * pow(dist_squared, b) + 1.0

                grad_coeff_repul = 2.0 * gamma * b
                grad_coeff_repul /= (0.001 + dist_squared) * (
                        a * c_2b[i] * pow(dist_squared, b) + 1
                )

                grad_coeff += grad_coeff_repul

            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * w * (current[d] - other[d]))

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * b
                    grad_coeff /= (0.001 + dist_squared) * (
                            a * c_2b[i] * pow(dist_squared, b) + 1
                    )
                    # grad_coeff = -(-2*a*np.sqrt(dist_squared)+1)/a*pow(dist_squared, 1.5)

                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (1-w) * (current[d] - other[d]))
                    else:
                        grad_d = 2.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def _spacemap_optimization_symmetric(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    n_vertices,
    epochs_per_sample,
    factor,
    beta,
    local_rate,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    negative_sampling,
):
    """
    Intro
    ----------
    2021.12
    Qij = exp(-(dij^2/alpha)^(1/beta) / sigma_q)

    Parameters
    ----------
    head_embedding
    tail_embedding
    head
    tail
    weight
    n_vertices
    epochs_per_sample
    factor
    beta
    rng_state
    gamma
    dim
    move_other
    alpha
    epochs_per_negative_sample
    epoch_of_next_negative_sample
    epoch_of_next_sample
    n
    negative_sampling

    Returns
    -------

    """
    # if n < 70:
    #     early_exaggeration = 0.0
    # else:
    #     early_exaggeration = 1.0

    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            w = weight[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:

                grad_coeff = 2.0 / (beta * factor) * pow(dist_squared, (1 / beta - 1))

            else:
                grad_coeff = 0.0

            grad_repulsive = 2.0 * gamma / (beta * factor) * pow(dist_squared, (1 / beta - 1)) * \
                             (1 / (1 - np.exp(factor * pow(dist_squared, (1 / beta))) - 1e-15) - 1)

            for d in range(dim):
                grad_d = clip((grad_coeff * w - grad_repulsive * (1-w)) * (current[d] - other[d]))  # w

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            if negative_sampling:

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices
                    # print(k)
                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        # remove gamma in negative sampling:
                        # grad_coeff = -2.0 * gamma / (beta * factor) * pow(dist_squared, (1 / beta - 1)) \
                        grad_coeff = -2.0 / (beta * factor) * pow(dist_squared, (1 / beta - 1)) \
                                     * (1 / (1 - np.exp(factor * pow(dist_squared, (1 / beta))) - 1e-15) - 1)
                        # grad_coeff = -(-2*a*np.sqrt(dist_squared)+1)/a*pow(dist_squared, 1.5)

                    elif j == k:  # repulsive force can't add to itself
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (1-w) * (current[d] - other[d]))  # (1-w)
                        else:
                            grad_d = 4.0
                        # current[d] += grad_d * alpha * early_exaggeration  # add 0.1
                        current[d] += grad_d * alpha  # add 0.1

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )


# current
def _spacemap_optimization_asymmetric(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    n_vertices,
    epochs_per_sample,
    factor,
    beta,
    local_rate,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    negative_sampling,
):
    """
    Intro
    ----------
    2021.12
    different factors based on the near field range

    Parameters
    ----------
    head_embedding
    tail_embedding
    head
    tail
    weight
    n_vertices
    epochs_per_sample
    factor
    beta
    rng_state
    gamma: repulsive force rate in each positive sampling
    dim
    move_other
    alpha
    epochs_per_negative_sample
    epoch_of_next_negative_sample
    epoch_of_next_sample
    n
    negative_sampling

    Returns
    -------

    """
    # if n < 70:
    #     early_exaggeration = 0.0
    # else:
    #     early_exaggeration = 1.0

    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            w = weight[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:

                grad_coeff = 2.0 / (beta * factor[i]) * pow(dist_squared, (1 / beta - 1)) - \
                             2.0 * gamma / (beta * factor[i]) * pow(dist_squared, (1 / beta - 1)) * \
                             (1 / (1 - np.exp(factor[i] * pow(dist_squared, (1 / beta))) - 1e-15) - 1)  # 1e-15 for stability

            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * w * (current[d] - other[d]))  # weights have been included

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            if negative_sampling:

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices
                    # print(k)
                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        # remove gamma in negative sampling:
                        # grad_coeff = -2.0 * gamma / (beta * factor) * pow(dist_squared, (1 / beta - 1)) \
                        grad_coeff = -2.0 / (beta * factor[i]) * pow(dist_squared, (1 / beta - 1)) \
                                     * (1 / (1 - np.exp(factor[i] * pow(dist_squared, (1 / beta))) - 1e-15) - 1)  # 1e-15 for stability
                        # -1e-8 instead of +1e-8 !!! if it's 1e-8 then the results could be lower than zero
                        # grad_coeff = -(-2*a*np.sqrt(dist_squared)+1)/a*pow(dist_squared, 1.5)

                    elif j == k:  # repulsive force can't add to itself
                        continue
                    else:
                        grad_coeff = 0.0
                        print('d2=', dist_squared)

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (1-w) * (current[d] - other[d]))  # (1-w)
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )


def spacemap_optimization(
    min_dist,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    factor,
    standard_factor,
    beta,  # the exponential element in extension
    local_rate,
    rng_state,
    h_largest_dist=20.0,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    num_plots=200,
    parallel=False,
    verbose=True,
    plot_results=False,
    negative_sampling=True,
    move_other=True,
):

    dim = head_embedding.shape[1]
    # move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    # epochs_per_negative_sample_ee = epochs_per_sample / (negative_sample_rate / 10.0)
    # epoch_of_next_negative_sample_ee = epochs_per_negative_sample_ee.copy()

    embedding_results = np.zeros((num_plots, head_embedding.shape[0], dim))

    if min_dist == 0:  # exponential Qij:

        if isinstance(factor, np.ndarray):  # asymmetric layout:

            print('[SpaceMAP] asymmetric factor in the function Pij: ', factor, '\tfactor shape: ', factor.shape)
            # ========================================= main method =========================================
            fn = numba.njit(
                _spacemap_optimization_asymmetric, fastmath=True, parallel=parallel
            )
            print('[SpaceMAP] asymmetric layout')

        else:  # symmetric layout:
            # factor should be unified in low-dim space
            print('[SpaceMAP] local range fixed')
            print('factor: ', factor)
            fn = numba.njit(
                _spacemap_optimization_symmetric, fastmath=True, parallel=parallel
            )
            print('[SpaceMAP] symmetric layout')

        for n in range(n_epochs):

            fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                weight,
                n_vertices,
                epochs_per_sample,
                factor,
                beta,
                local_rate,
                rng_state,
                gamma,
                dim,
                move_other,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                negative_sampling,
            )

            alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

            if verbose and n % int(n_epochs / 10) == 0:
                print("[SpaceMAP] optimized ", n, " / ", n_epochs, "epochs")

            if plot_results and n % int(n_epochs / num_plots) == 0:
                if verbose:
                    print("\t[SpaceMAP] saved embedding result")
                embedding_results[int(n / int(n_epochs / num_plots)), :, :] = head_embedding

    elif min_dist > 0:  # t-dist like UMAP

        if isinstance(factor, np.ndarray):  # asymmetric

            print('[SpaceMAP] t-dist Bayes. standard factor:', factor)
            print('factor shape: ', factor.shape)

            a, b = find_ab_params(1, min_dist, standard_factor, beta, h_largest_dist)
            print('standard a: ', a)
            print('standard b: ', b)
            c_2b = factor ** (2*b)

            fn = numba.njit(
                _spacemap_optimization_asymmetric_t_dist, fastmath=True, parallel=parallel
            )
            for n in range(n_epochs):
                # without min_dist (exponential)
                fn(
                    head_embedding,
                    tail_embedding,
                    head,
                    tail,
                    weight,
                    n_vertices,
                    epochs_per_sample,
                    a,
                    b,
                    c_2b,
                    rng_state,
                    gamma,
                    dim,
                    move_other,
                    alpha,
                    epochs_per_negative_sample,
                    epoch_of_next_negative_sample,
                    epoch_of_next_sample,
                    n,
                    negative_sampling,
                )

                alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

                if verbose and n % int(n_epochs / 10) == 0:
                    print("[SpaceMAP] optimized ", n, " / ", n_epochs, "epochs")

                if plot_results and n % int(n_epochs / num_plots) == 0:
                    if verbose:
                        print("\t[SpaceMAP] saved embedding result")
                    embedding_results[int(n / int(n_epochs / num_plots)), :, :] = head_embedding

        else:  # symmetric

            a, b = find_ab_params(1, min_dist, factor, beta, h_largest_dist)
            print('a: ', a)
            print('b: ', b)
            print('factor: ', factor)
            fn = numba.njit(
                _spacemap_optimization_symmetric_t_dist, fastmath=True, parallel=parallel
            )

            for n in range(n_epochs):
                fn(
                    head_embedding,
                    tail_embedding,
                    head,
                    tail,
                    weight,
                    n_vertices,
                    epochs_per_sample,
                    a,
                    b,
                    rng_state,
                    gamma,
                    dim,
                    move_other,
                    alpha,
                    epochs_per_negative_sample,
                    epoch_of_next_negative_sample,
                    epoch_of_next_sample,
                    n,
                    negative_sampling,
                )

                alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

                if verbose and n % int(n_epochs / 10) == 0:
                    print("[SpaceMAP] optimized ", n, " / ", n_epochs, "epochs")

                if plot_results and n % int(n_epochs / num_plots) == 0:
                    if verbose:
                        print("\t[SpaceMAP] saved embedding result")
                    embedding_results[int(n / int(n_epochs / num_plots)), :, :] = head_embedding

    else:
        raise TypeError("[SpaceMAP] illegal min-dist setting")

    return head_embedding, embedding_results



