#!/usr/bin/env python3

from dasf.ml.cluster.agglomerative import AgglomerativeClustering  # noqa
from dasf.ml.cluster.dbscan import DBSCAN  # noqa
from dasf.ml.cluster.kmeans import KMeans  # noqa
from dasf.ml.cluster.som import SOM  # noqa
from dasf.ml.cluster.spectral import SpectralClustering  # noqa

cluster_methods = [
    "AgglomerativeClustering",
    "KMeans",
    "DBSCAN",
    "SOM",
    "SpectralClustering"
]

# XXX: Import due to CVE-2022-21797
import joblib  # noqa
from packaging import version  # noqa

if version.parse(joblib.__version__) < version.parse("1.2.0"): # pragma: no cover
    # Do not include HDBSCAN while it is not safe
    from dasf.ml.cluster.hdbscan import HDBSCAN  # noqa

    cluster_methods.append("HDBSCAN")
# End of workaround

__all__ = cluster_methods
