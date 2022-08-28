#!/usr/bin/env python3

from dasf.pipeline import Operator


class Histogram(Operator):
    def __init__(self,
                 bins=None,
                 range=None,
                 normed=False,
                 weights=None,
                 density=None):

        super().__init__(name="Histogram")
        self.bins = bins
        self.range = range
        self.normed = normed
        self.weights = weights
        self.density = density

    def run(self, X):
        if not self.range:
            range_min = self.xp.min(X)
            range_max = self.xp.max(X)
            self.range = [range_min, range_max]

        return self.xp.histogram(X, self.bins, self.range, self.normed,
                                 self.weights, self.density)
