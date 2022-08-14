import locale

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse

from umap_utils.spectral import spectral_layout  # ONLY for spectral initialization
from _layouts import spacemap_optimization
import faiss

from _utils import *

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
DISCONNECTION_DISTANCES = {
    "correlation": 1,
    "cosine": 1,
    "hellinger": 1,
    "jaccard": 1,
    "dice": 1,
}


class SpaceMAP(BaseEstimator):
    """
    Space-based Manifold Approximation and Projection

    Author:     Xinrui Zu

    Date:       01/03/2022

    Version:    0.5-dev

    Paper:      https://proceedings.mlr.press/v162/zu22a.html

    Parameters:
    ----------
    n_components : Embedding dimension

    n_near_field : Number of the nearest neighbors in the near field

    n_middle_field : Number of the nearest neighbors in the middle field

    d_local : Local intrinsic dimension. (Default: 0) for automatically estimation.

    d_global : Global intrinsic dimension. (Default: 0) for automatically estimation.

    eta : (Default: 0.6) hyper-parameter representing the conditional probability differentiating near- and middle-fields.

    n_epochs : Number of optimization epochs.

    init : Initialization method. (Default: Spectral embedding)

    metric : Distance metric. (Default: euclidean)

    verbose : Verbosity level.

    plot_results : Save the intermediate embedding results when optimizing. (Default: False)
    """
    def __init__(self,
                 n_components=2,
                 n_near_field=21,
                 n_middle_field=50,
                 d_local=0,
                 d_global=4.5,
                 eta=0.6,
                 n_epochs=200,
                 init='spectral',
                 metric='euclidean',
                 verbose=True,
                 plot_results=False,
                 num_plots=50,
                 ):

        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.data = 0
        self.random_state = check_random_state(42)
        self.verbose = verbose
        self.metric = metric
        self._disconnection_distance = DISCONNECTION_DISTANCES.get(
            self.metric, np.inf
        )

        self.n_near = n_near_field
        self.n_middle = n_middle_field
        self.knn_bayes = 50
        # Q Chi:

        self.d_local = d_local
        self.d_global = d_global  # 10
        self.manual_d_local = False
        if self.d_local != 0:
            self.manual_d_local = True

        self.local_range = 0
        self.local_rate = eta
        self.min_dist = 0
        self.d_global_extension_factor = 0
        self.d_global_extension_sigma = 0
        self.linear_expansion_rate = 0

        # self.beta = self.d_global - 1
        # self.beta_near = self.d_local - 1

        # set global intrinsic dimension manually:
        if self.d_global != 0:
            if self.metric == 'euclidean':
                self.beta = self.d_global / 2
                # self.beta_near = self.d_local / 2

                # self.beta_trans = self.beta / self.beta_near
                self.alpha = extension_from_2d_factor_euclidean(self.d_global)
                # self.alpha_near = extension_from_2d_factor_(self.d_local)
                # self.alpha_trans = self.alpha / (self.alpha_near ** self.beta_trans)

                # final version of factor in Qij:
                if self.local_range != 0:
                    # fixed near field in 2D
                    self.sigma_q = -(self.local_range**(2/self.beta) /
                                     self.alpha**(2/self.beta))/(np.log(1-self.local_rate))
                    self.factor = -1.0 / (self.sigma_q*np.power(self.alpha, 2/self.beta))
                else:
                    # init:
                    self.sigma_q = 0
                    self.factor = 0
                    self.local_embedding_range = 0

            elif self.metric == 'cosine':
                self.beta = (self.d_global-1) / 2
                self.alpha = extension_from_2d_factor_cosine(self.d_global)
                if self.local_range != 0:
                    # fixed near field in 2D
                    self.sigma_q = -(self.local_range**(2/self.beta) /
                                     self.alpha**(2/self.beta))/(np.log(1-self.local_rate))
                    self.factor = -1.0 / (self.sigma_q*np.power(self.alpha, 2/self.beta))
                else:  # Bayes
                    # init:
                    self.sigma_q = 0
                    self.factor = 0
                    self.local_embedding_range = 0

        # automatically estimate global intrinsic dimension:
        else:  # d_global == 0

            if self.metric == 'euclidean':
                self.beta = self.d_global / 2
                self.alpha = 0
                # final version of factor in Qij:
                if self.local_range != 0:
                    # fixed near field in 2D
                    self.sigma_q = 0
                    self.factor = 0
                else:
                    # init:
                    self.sigma_q = 0
                    self.factor = 0
                    self.local_embedding_range = 0

            elif self.metric == 'cosine':
                self.beta = 0
                self.alpha = 0
                if self.local_range != 0:
                    # fixed near field in 2D
                    self.sigma_q = 0
                    self.factor = 0
                else:  # Bayes
                    # init:
                    self.sigma_q = 0
                    self.factor = 0
                    self.local_embedding_range = 0

        self.P = 0
        self.embedding = 0
        self._input_hash = 0
        self.n_neighbors = self.n_near + self.n_middle
        self._knn_indices = 0
        self._knn_dists = 0
        self.d_local_esti = 0

        self.standard_factor = 0

        self.lr = 1.0  # learning rate

        self.plot_results = plot_results
        self.num_plots = num_plots
        if self.plot_results:
            self.intermedian_embeddings = 0

        self.negative_sampling = True
        self.negative_sample_rate = 5.0  # negative sampling rate in UMAP. (default: 5.0)
        self.symmetric = 'average'  # pij symmetric method. (default: average)
        self.sgd = True  # use SGD optimization
        self.gamma = 0  # factor to generate repulsive force during optimization (not negative sampling) (default: 0s)
        self.eliminate_small_p = False
        self.move_other = True  # move the point pair on both sides when optimizing through attractive force
        self.bayes_type = 'geometry'
        self.return_p = False

        self.pij = 'spacemap'

        self.init_embedding = 0
        self.return_init = False
        self.precompute_data = 0

    def hierarchical_manifold_approximation(self):
        """
        Calculate high-dimensional similarity P in near/middle fields.
        """
        if self.verbose:
            print('[SpaceMAP] calculating knn graph...')

        print('[SpaceMAP] use FAISS library to calculate knn graph')
        gpures = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        index = faiss.GpuIndexFlatL2(gpures, self.data.shape[1], flat_config)
        index.add(self.data)

        self._knn_dists, self._knn_indices = index.search(self.data,
                                                          self.n_neighbors)
        self._knn_dists = np.sqrt(self._knn_dists)

        if self.beta == 0:
            knn_dists = self._knn_dists[:, 1:self.n_neighbors]
            self.d_global = mle_intrinsic_dimension(knn_dists, self.n_neighbors-1, metric=self.metric)

            if self.verbose:
                print('[SpaceMAP] d-global-esti <mk(x)>: ', self.d_global)

            self.d_global[np.where(self.d_global > 50)] = 50
            # self.d_global = np.mean(self.d_global)  # mean
            self.d_global = 1 / np.mean(1 / self.d_global)  # NO mean

            if self.verbose:
                print('[SpaceMAP] d-global is auto-calculated <mk>: ', self.d_global)

            if self.metric == 'euclidean':
                self.beta = self.d_global / 2
                self.alpha = extension_from_2d_factor_euclidean(self.d_global)
                if self.local_range != 0:
                    self.sigma_q = -(self.local_range**(2/self.beta) /
                                     self.alpha**(2/self.beta))/(np.log(1-self.local_rate))
                    self.factor = -1.0 / (self.sigma_q*np.power(self.alpha, 2/self.beta))
                else:
                    self.sigma_q = 0
                    self.factor = 0
                    self.local_embedding_range = 0
            elif self.metric == 'cosine':
                self.beta = (self.d_global-1) / 2
                self.alpha = extension_from_2d_factor_cosine(self.d_global)
                if self.local_range != 0:
                    self.sigma_q = -(self.local_range**(2/self.beta) /
                                     self.alpha**(2/self.beta))/(np.log(1-self.local_rate))
                    self.factor = -1.0 / (self.sigma_q*np.power(self.alpha, 2/self.beta))
                else:  # Bayes
                    self.sigma_q = 0
                    self.factor = 0
                    self.local_embedding_range = 0

        # Disconnect any vertices farther apart than _disconnection_distance
        disconnected_index = self._knn_dists >= self._disconnection_distance
        self._knn_indices[disconnected_index] = -1
        self._knn_dists[disconnected_index] = np.inf

        if self.verbose:
            print('[SpaceMAP] knn graph calculated!')

        # ================================ P calculation ================================

        self.P, self.d_local_esti = p_calculation_ultimate(self.data,
                                                           self._knn_indices,
                                                           self._knn_dists,
                                                           self.n_near,
                                                           self.n_middle,
                                                           self.local_rate,
                                                           self.d_local,
                                                           self.alpha,
                                                           self.beta,
                                                           self.local_range,
                                                           use_manual_d_local=self.manual_d_local,
                                                           metric=self.metric,
                                                           verbose=self.verbose,
                                                           symmetric=self.symmetric,
                                                           pij=self.pij,
                                                           )
        return self.P

    def _fit_embedding(self, random_state):
        graph = self.P.tocoo()
        # graph.sum_duplicates()
        n_vertices = graph.shape[1]

        if self.n_epochs <= 0:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                self.n_epochs = 500
            else:
                self.n_epochs = 200

        # this step eliminates small weights!

        if self.eliminate_small_p:

            graph.data[graph.data < (graph.data.max() / float(self.n_epochs))] = 0.0
            graph.eliminate_zeros()

        # -------------------- Initialization --------------------

        if isinstance(self.init, str) and self.init == 'random':
            embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], self.n_components)
            ).astype(np.float32)
        elif isinstance(self.init, str) and self.init == 'spectral':
            # We add a little noise to avoid local minima for optimization to come
            initialisation = spectral_layout(
                self.data,
                graph,
                self.n_components,
                random_state,
                metric="euclidean",
                metric_kwds={},
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph.shape[0], self.n_components]
            ).astype(
                np.float32
            )
        elif isinstance(self.init, str) and self.init == 'pca':
            print('[SpaceMAP] pca begin!')
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            initialisation = pca.fit_transform(self.data).astype(np.float32, copy=False)
            print('[SpaceMAP] pca complete!')
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph.shape[0], self.n_components]
            ).astype(
                np.float32
            )
        elif isinstance(self.init, str) and self.init == 'umap':
            print('[SpaceMAP] Initialize with umap embedding -- begin')
            import umap
            umap_f = umap.UMAP(n_neighbors=5, n_epochs=200, verbose=self.verbose)
            initialisation = umap_f.fit_transform(self.data).astype(np.float32, copy=False)
            print('[SpaceMAP] Initialize with umap embedding -- complete')
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph.shape[0], self.n_components]
            ).astype(
                np.float32
            )
        elif isinstance(self.init, str) and self.init == 'precompute':
            print('[SpaceMAP] Initialize with precomputed data')
            embedding = self.precompute_data
        else:
            init_data = np.array(self.init)
            if len(init_data.shape) == 2:
                if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                    tree = KDTree(init_data)
                    dist, ind = tree.query(init_data, k=2)
                    nndist = np.mean(dist[:, 1])
                    embedding = init_data + random_state.normal(
                        scale=0.001 * nndist, size=init_data.shape
                    ).astype(np.float32)
                else:
                    embedding = init_data

        self.init_embedding = embedding
        if self.return_init:
            return self.init_embedding

        head = graph.row
        tail = graph.col
        weight = graph.data
        if self.sgd:
            epochs_per_sample = make_epochs_per_sample(graph.data, self.n_epochs)
        else:
            epochs_per_sample = np.ones(head.shape[0])  # * self.negative_sample_rate

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)  # random state of negative sampling

        # variant Qij(Bayes):
        if self.local_range == 0:
            # self.sigma_q = -((self._knn_dists[:, self.n_near]/2)**(2/self.beta) /
            #                self.alpha**(2/self.beta))/(np.log(1-self.local_rate))
            # self.factor = -1.0 / (self.sigma_q*np.power(self.alpha, 2/self.beta))
            if self.bayes_type == 'original':
                self.local_embedding_range = self._knn_dists[:, self.n_near]
            elif self.bayes_type == 'sqrt':
                self.local_embedding_range = self._knn_dists[:, self.n_near] ** (1 / 2)
            elif self.bayes_type == 'geometry':
                self.local_embedding_range = self._knn_dists[:, self.n_near] ** (2 / self.d_global)
            elif self.bayes_type == 'real':
                self.local_embedding_range = self._knn_dists[:, self.n_near] ** (self.d_global / 2)
                self.local_embedding_range /= np.mean(self._knn_dists[:, self.n_near]) ** (self.d_global / 2)
            else:
                raise TypeError('[SpaceMAP] Bayes method unavailable')

            self.sigma_q = -(self.local_embedding_range ** (2 / self.beta) /
                             self.alpha ** (2 / self.beta)) / (np.log(1 - self.local_rate))
            self.factor = -1.0 / (self.sigma_q * np.power(self.alpha, 2 / self.beta))
            _, head_count = np.unique(head, return_counts=True)
            self.factor = np.repeat(self.factor, head_count, axis=0)

            if self.min_dist > 0:  # t-dist curve fitting:

                # 此factor非彼factor: (t-dist )
                self.factor = 1 / self.local_embedding_range
                _, head_count = np.unique(head, return_counts=True)
                self.factor = np.repeat(self.factor, head_count, axis=0)

                self.standard_sigma_q = -(1.0 /
                                          self.alpha ** (2 / self.beta)) / (np.log(1 - self.local_rate))
                self.standard_factor = -1.0 / (self.standard_sigma_q * np.power(self.alpha, 2 / self.beta))

        embedding = (
                10.0
                * (embedding - np.min(embedding, 0))
                / (np.max(embedding, 0) - np.min(embedding, 0))
        ).astype(np.float32, order="C")

        # ================================ Q calculation (optimization) ================================

        embedding, self.intermedian_embeddings = spacemap_optimization(
            self.min_dist,
            embedding,
            embedding,
            head,
            tail,
            weight,
            self.n_epochs,
            n_vertices,
            epochs_per_sample,
            self.factor,
            self.standard_factor,
            self.beta,
            self.local_rate,
            rng_state,
            h_largest_dist=np.max(self._knn_dists[:, -1]),
            gamma=self.gamma,
            initial_alpha=self.lr,
            negative_sample_rate=self.negative_sample_rate,
            num_plots=self.num_plots,
            parallel=False,
            verbose=self.verbose,
            plot_results=self.plot_results,
            negative_sampling=self.negative_sampling,
            move_other=self.move_other,
        )

        return embedding

    def fit_transform(self, X):
        print('[SpaceMAP] start. time:', time.ctime(time.time()))
        self.data = X
        self.P = self.hierarchical_manifold_approximation()
        if self.return_p:
            return self.P
        self.embedding = self._fit_embedding(check_random_state(42))
        print('[SpaceMAP] complete. time:', time.ctime(time.time()))

        return self.embedding





