#!/usr/bin/env python3

""" Init module for Clustering ML algorithms. """

from dasf.ml.cluster.agglomerative import AgglomerativeClustering  # noqa
from dasf.ml.cluster.dbscan import DBSCAN  # noqa
from dasf.ml.cluster.kmeans import KMeans  # noqa
from dasf.ml.cluster.som import SOM  # noqa
from dasf.ml.cluster.spectral import SpectralClustering  # noqa
from dasf.ml.cluster.hdbscan import HDBSCAN  # noqa

cluster_methods = [
    "AgglomerativeClustering",
    "KMeans",
    "DBSCAN",
    "SOM",
    "SpectralClustering",
    "HDBSCAN"
]

__all__ = cluster_methods
