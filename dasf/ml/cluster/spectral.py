#!/usr/bin/env python3

from sklearn.cluster import SpectralClustering as SpectralClustering_CPU
from dask_ml.cluster import SpectralClustering as SpectralClustering_MCPU

from dasf.ml.cluster.classifier import ClusterClassifier


class SpectralClustering(ClusterClassifier):
    def __init__(
        self,
        n_clusters=8,
        eigen_solver=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol=0.0,
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        n_components=None,
        persist_embedding=False,
        kmeans_params=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.n_components = n_components
        self.persist_embedding = persist_embedding
        self.kmeans_params = kmeans_params
        self.verbose = verbose

        self.__sc_cpu = SpectralClustering_CPU(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            n_components=n_components,
            verbose=verbose
        )

        # If n_components is set to None, use default
        n_components = 100 if n_components is None else n_components

        self.__sc_mcpu = SpectralClustering_MCPU(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            n_components=n_components,
            persist_embedding=persist_embedding,
            kmeans_params=kmeans_params,
        )

    def _fit_cpu(self, X, y=None, sample_weight=None):
        return self.__sc_cpu.fit(X=X, y=y)

    def _lazy_fit_predict_cpu(self, X, y=None, sample_weight=None):
        return self.__sc_mcpu.fit_predict(X=X)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        return self.__sc_cpu.fit_predict(X)
