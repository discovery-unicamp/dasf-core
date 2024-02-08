#!/usr/bin/env python3

import numpy as np
import dask.array as da
try:
    import cupy as cp
except ImportError: # pragma: no cover
    pass

from dasf.transforms.base import Transform, ReductionTransform
from dasf.ml.inference.loader.base import BaseLoader


class Reshape:
    """Reshape data with a new shape.

    Parameters
    ----------
    shape : tuple
        The new shape of the data.

    """
    def __init__(self, shape: tuple):
        self.shape = shape

    def run(self, X):
        print(X.shape)
        return X.reshape(self.shape)


class SliceArray(Transform):
    def __init__(self, output_size):
        self.x = list(output_size)

    def transform(self, X):
        if len(self.x) == 1:
            return X[0:self.x[0]]
        elif len(self.x) == 2:
            return X[0:self.x[0], 0:self.x[1]]
        elif len(self.x) == 3:
            return X[0:self.x[0], 0:self.x[1], 0:self.x[2]]
        else:
            raise Exception("The dimmension is not known")


class SliceArrayByPercent(Transform):
    def __init__(self, x=100.0, y=100.0, z=100.0):
        self.x = float(x / 100.0)
        self.y = float(y / 100.0)
        self.z = float(z / 100.0)

    def transform(self, X):
        if self.x > 1 or self.y > 1 or self.z > 1:
            raise Exception("Percentages cannot be higher than 100% (1.0)")

        if X.ndim == 1:
            return X[0:int(self.x * X.shape[0])]
        elif X.ndim == 2:
            return X[0:int(self.x * X.shape[0]), 0:int(self.y * X.shape[1])]
        elif X.ndim == 3:
            return X[
                0:int(self.x * X.shape[0]),
                0:int(self.y * X.shape[1]),
                0:int(self.z * X.shape[2]),
            ]
        else:
            raise Exception("The dimmension is not known")


class SliceArrayByPercentile(Transform):
    def __init__(self, percentile):
        self.p = percentile

    def __internal_chunk_array_positive(self, block, axis=None, keepdims=False, xp=np):
        block[block < 0] = 0
        block[block != 0]
        return xp.array([xp.percentile(block.flatten(), self.p)])

    def __internal_aggregate_array_positive(self, block, axis=None, keepdims=False, xp=np):
        return xp.array([xp.max(block)])

    def __internal_chunk_array_negative(self, block, axis=None, keepdims=False, xp=np):
        block *= -1
        block[block < 0] = 0
        block[block != 0]
        return xp.array([-xp.percentile(block.flatten(), self.p)])

    def __internal_aggregate_array_negative(self, block, axis=None, keepdims=False, xp=np):
        return xp.array([xp.min(block)])

    def _lazy_transform_cpu(self, X):
        positive = ReductionTransform(func_chunk=self.__internal_chunk_array_positive,
                                      func_aggregate=self.__internal_aggregate_array_positive,
                                      output_size=[0])

        negative = ReductionTransform(func_chunk=self.__internal_chunk_array_negative,
                                      func_aggregate=self.__internal_aggregate_array_negative,
                                      output_size=[0])

        p = positive._lazy_transform_cpu(X, axis=[0])
        n = negative._lazy_transform_cpu(X, axis=[0])

        # Unfortunately, we need to compute first.
        pos_cutoff = p.compute()[0]
        neg_cutoff = n.compute()[0]

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _lazy_transform_gpu(self, X):
        positive = ReductionTransform(func_chunk=self.__internal_aggregate_array_positive,
                                      func_aggregate=self.__internal_aggregate_array_positive,
                                      output_size=[0])

        negative = ReductionTransform(func_chunk=self.__internal_aggregate_array_negative,
                                      func_aggregate=self.__internal_aggregate_array_negative,
                                      output_size=[0])

        p = positive._lazy_transform_gpu(X)
        n = negative._lazy_transform_gpu(X)

        # Unfortunately, we need to compute first.
        pos_cutoff = p.compute()[0]
        neg_cutoff = n.compute()[0]

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_cpu(self, X):
        pos_cutoff = self.__internal_chunk_array_positive(X, xp=np)
        neg_cutoff = self.__internal_chunk_array_negative(X, xp=np)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_gpu(self, X):
        pos_cutoff = self.__internal_chunk_array_positive(X, xp=cp)
        neg_cutoff = self.__internal_chunk_array_negative(X, xp=cp)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

class ApplyPatches(Transform):
    def __init__(self, function, input_size, overlap, offsets, combine_function):
        self._function = function
        self._input_size = input_size
        self._overlap_config = overlap
        self._offsets = offsets
        self._combine_function = combine_function

    def _apply_patches(self, patch_set):
        if callable(self._function):
            return np.array(list(map(self._function, patch_set)))
        if isinstance(self._function, BaseLoader):
            return self._function.predict(patch_set)
        raise NotImplementedError("Requested Apply Method not supported")

    def _reconstruct_patches(self, patches, index):
        reconstruct_shape = np.array(self._input_size) * np.array(index)
        reconstruct = np.zeros(reconstruct_shape)
        weight = np.ones(reconstruct_shape)
        for patch_index, patch in zip(np.ndindex(index), patches):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, self._input_size)
            ]
            reconstruct[tuple(sl)] = patch
        return reconstruct, weight

    def _adjust_patches(self, patches, weight, ref_shape, offset):
        pad_width = []
        sl = []
        for idx, lenght, ref in zip(offset, patches.shape, ref_shape):

            if idx > 0:
                sl.append(slice(0, min(lenght, ref), None))
                pad_width.append((idx, max(ref - lenght - idx, 0)))
            else:
                sl.append(slice(np.abs(idx), min(lenght, ref - idx), None))
                pad_width.append((0, max(ref - lenght - idx, 0)))
        reconstruct = np.pad(patches[tuple(sl)], pad_width=pad_width, mode="constant")
        weight = np.pad(weight[tuple(sl)], pad_width=pad_width, mode="constant")
        return reconstruct, weight

    def _combine_patches(self, results, offsets, indexes):
        reconstructed = []
        weights = []
        for patches, offset, shape in zip(results, offsets, indexes):
            reconstruct, weight = self._reconstruct_patches(patches, shape)
            if len(reconstructed) > 0:
                reconstruct, weight = self._adjust_patches(
                    reconstruct, weight, reconstructed[0].shape, offset
                )
            reconstructed.append(reconstruct)
            weights.append(weight)
        reconstructed = np.stack(reconstructed, axis=0)
        weights = np.stack(weights, axis=0)
        return np.sum(reconstructed * weights, axis=0) / np.sum(weights, axis=0)

    def _extract_patches(self, data, patch_shape):
        indexes = tuple(np.array(data.shape) // np.array(patch_shape))
        patches = []
        for patch_index in np.ndindex(indexes):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, patch_shape)
            ]
            patches.append(data[tuple(sl)])
        return np.asarray(patches), indexes

    def _operation(self, chunk):
        offsets = list(self._offsets)
        base = self._overlap_config["padding"]
        offsets.insert(0, tuple([0] * len(base)))

        slices = [
            tuple([slice(i + base, None) for i, base in zip(offset, base)])
            for offset in offsets
        ]
        results = []
        indexes = []
        for sl in slices:
            patch_set, patch_idx = self._extract_patches(chunk[sl], self._input_size)
            results.append(self._apply_patches(patch_set))
            indexes.append(patch_idx)
        output_slice = tuple(
            [slice(0, lenght - 2 * pad) for lenght, pad in zip(chunk.shape, base)]
        )
        return self._combine_patches(results, offsets, indexes)[output_slice]

    def _transform(self, X):
        X_overlap = np.pad(
            X,
            pad_width=[(pad, pad) for pad in self._overlap_config["padding"]],
            mode=self._overlap_config["boundary"],
        )

        return self._operation(X_overlap)

    def _lazy_transform(self, X):
        X_overlap = da.overlap.overlap(
            X,
            depth=self._overlap_config["padding"],
            boundary=self._overlap_config["boundary"],
        )

        return X_overlap.map_blocks(
            self._operation, dtype=X_overlap.dtype, chunks=X.chunks
        )

    def _lazy_transform_cpu(self, X, **kwargs):
        return self._lazy_transform(X)

    def _lazy_transform_gpu(self, X, **kwargs):
        X = X.map_blocks(cp.asnumpy, dtype=X.dtype, meta=np.array((), dtype=X.dtype))
        return self._lazy_transform(X).map_blocks(
            cp.asarray, dtype=X.dtype, meta=cp.array((), dtype=X.dtype)
        )

    def _transform_cpu(self, X, **kwargs):
        return self._transform(X)

    def _transform_gpu(self, X, **kwargs):
        X = cp.asnumpy(X)
        return cp.asarray(self._transform(X))
