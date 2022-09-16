#!/usr/bin/env python3

from dasf.pipeline import Operator


class Histogram(Operator):
    """Operator to extract the histogram of a data.

    Parameters
    ----------
    bins : Optional[int]
        Number of bins (the default is None).
    range : tuple
        2-element tuple with the lower and upper range of the bins. If not
        provided, range is simply (X.min(), X.max()) (the default is None).
    normed : bool
        If the historgram must be normalized (the default is False).
    weights : type
        An array of weights, of the same shape as X. Each value in a only
        contributes its associated weight towards the bin count
        (the default is None).
    density : type
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (the default is None).

    Attributes
    ----------
    bins
    range
    normed
    weights
    density

    """

    def __init__(self,
                 bins: int = None,
                 range: tuple = None,
                 normed: bool = False,
                 weights=None,
                 density=None):

        super().__init__(name="Histogram")
        self.bins = bins
        self.range = range
        self.normed = normed
        self.weights = weights
        self.density = density

    def run(self, X):
        """Calculates the histogram of data X.

        Parameters
        ----------
        X : Any
            The data.

        Returns
        -------
        type
            2 element tuple, with the following elements (in order):

            - The values of the histogram.
            - The bin edges.

        """
        if not self.range:
            range_min = self.xp.min(X)
            range_max = self.xp.max(X)
            self.range = [range_min, range_max]

        return self.xp.histogram(X, self.bins, self.range, self.normed,
                                 self.weights, self.density)
