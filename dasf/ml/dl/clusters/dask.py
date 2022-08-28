#!/usr/bin/env python

import os

from pytorch_lightning.plugins.environments import ClusterEnvironment


class DaskClusterEnvironment(ClusterEnvironment):
    """
    Create a Dask Cluster environment for workers

    metadata -- dictionary containing all data related to workers.
    """
    def __init__(self, metadata=None) -> None:
        super().__init__()

        if isinstance(metadata, dict):
            self.metadata = metadata
        else:
            self.metadata = {k.lower(): v for k, v in os.environ.items()}

        self._master_port = 23456

    def detect(self) -> bool:
        if "master" not in self.metadata:
            return False
        if "world_size" not in self.metadata:
            return False
        if "global_rank" not in self.metadata:
            return False

        return True

    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes
        yourself).
        """
        return True

    @property
    def main_address(self) -> str:
        """Return master worker address."""
        return self.metadata["master"]

    @property
    def main_port(self) -> int:
        """Return master worker port."""
        return self._master_port

    def creates_children(self) -> bool:
        """Fork children when generate a cluster."""
        return False

    def world_size(self) -> int:
        """Return worker world size."""
        return int(self.metadata["world_size"])

    def global_rank(self) -> int:
        """Return worker global rank."""
        return int(self.metadata["global_rank"])

    def local_rank(self) -> int:
        """Return worker local rank."""
        if "local_rank" in self.metadata:
            return int(self.metadata["local_rank"])
        else:
            return 0

    def node_rank(self) -> int:
        """Return worker node rank (which is similar to global rank)."""
        return int(self.metadata["global_rank"])

    def set_world_size(self, size: int) -> None:
        self.metadata["world_size"] = size

    def set_global_rank(self, rank: int) -> None:
        self.metadata["global_rank"] = rank
