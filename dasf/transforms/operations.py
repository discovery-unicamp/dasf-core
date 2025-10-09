#!/usr/bin/env python3

""" Basic transform operations module. """

import dask.array as da
import numpy as np
from scipy import stats

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    pass

from dasf.ml.inference.loader.base import BaseLoader
from dasf.transforms.base import Fit, ReductionTransform, Transform
from dasf.utils.types import is_array, is_dataframe


class Reshape(Fit):
    """Get a slice of a cube. An inline slice is a section over the x-axis.

    Parameters
    ----------
    iline_index : int
        The index of the inline to get.

    """
    def __init__(self, shape: tuple = None):
        self.shape = shape

    def fit(self, X, y=None):
        if self.shape:
            cube_shape = self.shape
        elif y is not None and hasattr(y, "shape"):
            cube_shape = y.shape
        else:
            raise Exception("Missing shape input.")

        if is_array(X):
            slice_array = X
        elif is_dataframe(X):
            slice_array = X.values
        else:
            raise ValueError("X is not a known datatype.")

        return slice_array.reshape(cube_shape)


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

        if self.x <= 0 or self.y <= 0 or self.z <= 0:
            raise Exception("Percentages cannot be negative or 0")

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

    def _internal_chunk_array_positive(self, block, axis=None, keepdims=False, xp=np):
        block[block < 0] = 0
        block[block != 0]
        return xp.array([xp.percentile(block.flatten(), self.p)])

    def _internal_aggregate_array_positive(self, block, axis=None, keepdims=False, xp=np):
        block = xp.array(block)

        return xp.array([xp.max(block)])

    def _internal_chunk_array_negative(self, block, axis=None, keepdims=False, xp=np):
        block *= -1
        block[block < 0] = 0
        block[block != 0]
        return xp.array([-xp.percentile(block.flatten(), self.p)])

    def _internal_aggregate_array_negative(self, block, axis=None, keepdims=False, xp=np):
        block = xp.array(block)

        return xp.array([xp.min(block)])

    def _lazy_transform_cpu(self, X):
        positive = ReductionTransform(
                func_chunk=self._internal_chunk_array_positive,
                func_aggregate=self._internal_aggregate_array_positive,
                output_size=[0]
                )

        negative = ReductionTransform(
                func_chunk=self._internal_chunk_array_negative,
                func_aggregate=self._internal_aggregate_array_negative,
                output_size=[0]
                )

        p = positive._lazy_transform_cpu(X, concatenate=False)
        n = negative._lazy_transform_cpu(X, concatenate=False)

        # Unfortunately, we need to compute first.
        pos_cutoff = p.compute()[0]
        neg_cutoff = n.compute()[0]

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _lazy_transform_gpu(self, X):
        positive = ReductionTransform(
                func_chunk=self._internal_aggregate_array_positive,
                func_aggregate=self._internal_aggregate_array_positive,
                output_size=[0]
                )

        negative = ReductionTransform(
                func_chunk=self._internal_aggregate_array_negative,
                func_aggregate=self._internal_aggregate_array_negative,
                output_size=[0]
                )

        p = positive._lazy_transform_gpu(X, concatenate=False)
        n = negative._lazy_transform_gpu(X, concatenate=False)

        # Unfortunately, we need to compute first.
        pos_cutoff = p.compute()[0]
        neg_cutoff = n.compute()[0]

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_cpu(self, X):
        pos_cutoff = self._internal_chunk_array_positive(X, xp=np)
        neg_cutoff = self._internal_chunk_array_negative(X, xp=np)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_gpu(self, X):
        pos_cutoff = self._internal_chunk_array_positive(X, xp=cp)
        neg_cutoff = self._internal_chunk_array_negative(X, xp=cp)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X


class Overlap(Transform):
    """
    Operator to get chunks with their respective overlaps.
    Useful when it is desired to use the same chunks with overlaps
    for multiple operations.
    """

    def __init__(self, pad=(1, 1, 1)):
        self._pad = pad

    def _lazy_transform(self, X):
        return da.overlap.overlap(X, depth=self._pad, boundary="nearest")

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X)

    def _transform(self, X, xp):
        return xp.pad(
            X,
            [
                (self._pad[0], self._pad[0]),
                (self._pad[1], self._pad[1]),
                (self._pad[2], self._pad[2]),
            ],
            mode="edge",
        )

    def _transform_gpu(self, X):
        return self._transform(X, cp)

    def _transform_cpu(self, X):
        return self._transform(X, np)


class Trim(Transform):
    """
    Operator to trim dask array that was produced by an Overlap
    transform or subsequent results from that transform.
    """

    def __init__(self, trim=(1, 1, 1)):
        self._trim = trim

    def _lazy_transform(self, X):
        return da.overlap.trim_overlap(
            X,
            depth=self._trim,
            boundary="nearest",
        )

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X)

    def _transform(self, X):
        sl = [slice(t, -t) if t != 0 else slice(None, None) for t in self._trim]
        return X[tuple(sl)]

    def _transform_gpu(self, X):
        return self._transform(X)

    def _transform_cpu(self, X):
        return self._transform(X)


class Roll(Transform):
    """
    Operator to perform a roll along multiple axis
    """

    def __init__(self, shift=(1, 1, 1)):
        self._shift = shift

    def _transform_generic(self, X, xp):
        return xp.roll(X, shift=self._shift, axis=list(range(len(self._shift))))

    def _lazy_transform_gpu(self, X):
        return X.map_blocks(self._transform_generic, xp=cp)

    def _lazy_transform_cpu(self, X):
        return X.map_blocks(self._transform_generic, xp=np)

    def _transform_gpu(self, X):
        return self._transform_generic(X, cp)

    def _transform_cpu(self, X):
        return self._transform_generic(X, np)


class ApplyPatchesBase(Transform):
    """
    Base Class for ApplyPatches Functionalities
    """

    def __init__(self, function, weight_function, input_size, overlap, offsets):
        """
        function: Function to be applied to each patch, can be eiter a Python
                  Function or a ModelLoader.
        weight_function: Weight attribution function, must receive a shape and
                         produce a NDArray with the respective weights for each
                         array position.
        input_size: Size of input to the function to be applied.
        overlap: Dictionary containing overlapping/padding configurations to use
                 with np.pad or dask.overlap.overlap. Its important that for the
                 base patch set the whole "chunk core" is covered by the patches.
        offsets: List of offsets for overlapping patches extraction
        """
        self._function = function
        self._weight_function = weight_function
        self._input_size = input_size
        self._offsets = offsets if offsets is not None else []
        overlap = overlap if overlap is not None else {}
        self._overlap_config = {
            "padding": overlap.get("padding", tuple(len(input_size) * [0])),
            "boundary": overlap.get("boundary", 0),
        }

    def _apply_patches(self, patch_set):
        """
        Applies function to each patch in a patch set

        """
        if callable(self._function):
            return np.array(list(map(self._function, patch_set)))
        if isinstance(self._function, BaseLoader):
            return self._function.predict(patch_set)
        raise NotImplementedError("Requested Apply Method not supported")

    def _reconstruct_patches(self, patches, index, weights, inner_dim=None):
        """
        Rearranges patches to reconstruct area of interest from patches and weights
        """
        reconstruct_shape = np.array(self._input_size) * np.array(index)
        if weights:
            weight = np.zeros(reconstruct_shape)
            base_weight = (
                self._weight_function(self._input_size)
                if self._weight_function
                else np.ones(self._input_size)
            )
        else:
            weight = None
        if inner_dim is not None:
            reconstruct_shape = np.append(reconstruct_shape, inner_dim)
        reconstruct = np.zeros(reconstruct_shape)
        for patch_index, patch in zip(np.ndindex(index), patches):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, self._input_size)
            ]
            if weights:
                weight[tuple(sl)] = base_weight
            if inner_dim is not None:
                sl.append(slice(None, None, None))
            reconstruct[tuple(sl)] = patch
        return reconstruct, weight

    def _adjust_patches(self, arrays, ref_shape, offset, pad_value=0):
        """
        Pads reconstructed_patches with 0s to have same shape as the reference
        shape from the base patch set.
        """
        pad_width = []
        sl = []
        ref_shape = list(ref_shape)
        arr_shape = list(arrays[0].shape)
        if len(offset) < len(ref_shape):
            ref_shape = ref_shape[:-1]
            arr_shape = arr_shape[:-1]
        for idx, lenght, ref in zip(offset, arr_shape, ref_shape):
            if idx > 0:
                sl.append(slice(0, min(lenght, ref), None))
                pad_width.append((idx, max(ref - lenght - idx, 0)))
            else:
                sl.append(slice(np.abs(idx), min(lenght, ref - idx), None))
                pad_width.append((0, max(ref - lenght - idx, 0)))
        adjusted = [
            np.pad(
                arr[tuple([*sl, slice(None, None, None)])],
                pad_width=[*pad_width, (0, 0)],
                mode="constant",
                constant_values=pad_value,
            )
            if len(offset) < len(arr.shape)
            else np.pad(
                arr[tuple(sl)],
                pad_width=pad_width,
                mode="constant",
                constant_values=pad_value,
            )
            for arr in arrays
        ]
        return adjusted

    def _combine_patches(self, results, offsets, indexes):
        """
        How results are combined is dependent on what is being combined.
        ApplyPatchesWeightedAvg uses Weighted Average
        ApplyPatchesVoting uses Voting (hard or soft)
        """
        raise NotImplementedError("Combine patches method must be implemented")

    def _extract_patches(self, data, patch_shape):
        """
        Patch extraction method. It will be called once for the base patch set and
        also for the requested offsets (overlapping patch sets).
        """
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
        """
        Operation to be performed on each chunk
        """
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
        if isinstance(self._overlap_config["boundary"], int):
            X_overlap = np.pad(
                X,
                pad_width=[(pad, pad) for pad in self._overlap_config["padding"]],
                mode="constant",
                constant_values=self._overlap_config["boundary"],
            )
        else:
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
        new_chunks = []
        for chunk_set, padding in zip(X_overlap.chunks, self._overlap_config["padding"]):
            new_chunks.append(tuple(np.array(chunk_set) - 2*padding))
        new_chunks = tuple(new_chunks)

        X = X_overlap.map_blocks(
            self._operation, dtype=X_overlap.dtype, chunks=new_chunks
        )
        X = X.rechunk()
        return X

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


class ApplyPatchesWeightedAvg(ApplyPatchesBase):
    """
    ApplyPatches with Weighted Average combination function.
    """

    def _combine_patches(self, results, offsets, indexes):
        reconstructed = []
        weights = []
        for patches, offset, shape in zip(results, offsets, indexes):
            reconstruct, weight = self._reconstruct_patches(
                patches, shape, weights=True
            )
            if len(reconstructed) > 0:
                adjusted = self._adjust_patches(
                    [reconstruct, weight], reconstructed[0].shape, offset
                )
                reconstruct = adjusted[0]
                weight = adjusted[1]
            reconstructed.append(reconstruct)
            weights.append(weight)
        reconstructed = np.stack(reconstructed, axis=0)
        weights = np.stack(weights, axis=0)
        return np.sum(reconstructed * weights, axis=0) / np.sum(weights, axis=0)


class ApplyPatchesVoting(ApplyPatchesBase):
    """
    ApplyPatches with Voting combination function.
    """

    def __init__(
        self,
        function,
        weight_function,
        input_size,
        overlap,
        offsets,
        voting,
        num_classes,
    ):
        """
        function: Function to be applied to each patch, can be eiter a Python
                  Function or a ModelLoader.
        weight_function: Weight attribution function, must receive a shape and
                         produce a NDArray with the respective weights for each
                         array position.
        input_size: Size of input to the function to be applied.
        overlap: Dictionary containing overlapping/padding configurations to use
                 with np.pad or dask.overlap.overlap. Its important that for the
                 base patch set the whole "chunk core" is covered by the patches.
        offsets: List of offsets for overlapping patches extraction.
        voting: Voting method. "hard"  or "soft".
        num_classes: Number of classes possible.
        """
        super().__init__(function, weight_function, input_size, overlap, offsets)
        self._voting = voting  # Types: Hard Voting, Soft Voting
        self._num_classes = num_classes

    def _combine_patches(self, results, offsets, indexes):
        if self._voting == "hard":
            result = self._hard_voting(results, offsets, indexes)
        elif self._voting == "soft":
            result = self._soft_voting(results, offsets, indexes)
        else:
            raise ValueError("Invalid Voting Type. Should be either soft or hard.")
        return result

    def _hard_voting(self, results, offsets, indexes):
        """
        Hard voting combination function
        """
        reconstructed = []
        for patches, offset, shape in zip(results, offsets, indexes):
            reconstruct, _ = self._reconstruct_patches(
                patches, shape, weights=False, inner_dim=self._num_classes
            )
            reconstruct = np.argmax(reconstruct, axis=-1).astype(np.float32)
            if len(reconstructed) > 0:
                adjusted = self._adjust_patches(
                    [reconstruct], reconstructed[0].shape, offset, pad_value=np.nan
                )
                reconstruct = adjusted[0]
            reconstructed.append(reconstruct)
        reconstructed = np.stack(reconstructed, axis=0)
        ret = stats.mode(reconstructed, axis=0, nan_policy="omit", keepdims=False)[0]
        return ret

    def _soft_voting(self, results, offsets, indexes):
        """
        Soft voting combination function
        """
        reconstructed = []
        for patches, offset, shape in zip(results, offsets, indexes):
            reconstruct, _ = self._reconstruct_patches(
                patches, shape, weights=False, inner_dim=self._num_classes
            )
            if len(reconstructed) > 0:
                adjusted = self._adjust_patches(
                    [reconstruct], reconstructed[0].shape, offset
                )
                reconstruct = adjusted[0]
            reconstructed.append(reconstruct)
        reconstructed = np.stack(reconstructed, axis=0)
        return np.argmax(np.sum(reconstructed, axis=0), axis=-1)
