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
        """
        Initialize the Reshape transform.

        Parameters
        ----------
        shape : tuple, optional
            Target shape for reshaping the data.
        """
        self.shape = shape

    def fit(self, X, y=None):
        """
        Fit the transform and reshape the input data.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data to reshape.
        y : array-like, optional
            Target data with shape information.

        Returns
        -------
        array-like
            Reshaped data.

        Raises
        ------
        Exception
            If shape cannot be determined.
        ValueError
            If X is not a known datatype.
        """
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
    """
    Transform to slice arrays to a specified output size.

    This class extracts a slice from the beginning of an array
    up to the specified dimensions.

    Parameters
    ----------
    output_size : tuple
        Target size for each dimension of the output array.
    """
    def __init__(self, output_size):
        """
        Initialize the SliceArray transform.

        Parameters
        ----------
        output_size : tuple
            Target size for each dimension of the output array.
        """
        self.x = list(output_size)

    def transform(self, X):
        """
        Transform the input array by slicing to output size.

        Parameters
        ----------
        X : array-like
            Input array to slice.

        Returns
        -------
        array-like
            Sliced array.

        Raises
        ------
        Exception
            If the dimension is not supported (>3D).
        """
        if len(self.x) == 1:
            return X[0:self.x[0]]
        elif len(self.x) == 2:
            return X[0:self.x[0], 0:self.x[1]]
        elif len(self.x) == 3:
            return X[0:self.x[0], 0:self.x[1], 0:self.x[2]]
        else:
            raise Exception("The dimmension is not known")


class SliceArrayByPercent(Transform):
    """
    Transform to slice arrays by percentage of each dimension.

    This class extracts a percentage of the array from the beginning
    of each dimension.

    Parameters
    ----------
    x : float, default=100.0
        Percentage of the first dimension to keep.
    y : float, default=100.0
        Percentage of the second dimension to keep.
    z : float, default=100.0
        Percentage of the third dimension to keep.
    """
    def __init__(self, x=100.0, y=100.0, z=100.0):
        """
        Initialize the SliceArrayByPercent transform.

        Parameters
        ----------
        x : float, default=100.0
            Percentage of the first dimension to keep.
        y : float, default=100.0
            Percentage of the second dimension to keep.
        z : float, default=100.0
            Percentage of the third dimension to keep.
        """
        self.x = float(x / 100.0)
        self.y = float(y / 100.0)
        self.z = float(z / 100.0)

    def transform(self, X):
        """
        Transform the input array by slicing by percentages.

        Parameters
        ----------
        X : array-like
            Input array to slice.

        Returns
        -------
        array-like
            Sliced array.

        Raises
        ------
        Exception
            If percentages are > 100% or <= 0%, or if dimension is not supported.
        """
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
    """
    Transform to clip array values based on percentile thresholds.

    This class clips array values to positive and negative percentile
    thresholds to remove outliers.

    Parameters
    ----------
    percentile : float
        Percentile value for clipping thresholds.
    """
    def __init__(self, percentile):
        """
        Initialize the SliceArrayByPercentile transform.

        Parameters
        ----------
        percentile : float
            Percentile value for clipping thresholds.
        """
        self.p = percentile

    def _internal_chunk_array_positive(self, block, axis=None, keepdims=False, xp=np):
        """
        Internal method to compute positive percentile for a chunk.

        Parameters
        ----------
        block : array-like
            Input block to process.
        axis : int, optional
            Axis along which to compute the percentile.
        keepdims : bool, default=False
            Whether to keep dimensions.
        xp : module, default=np
            Array module (numpy or cupy).

        Returns
        -------
        array-like
            Percentile value for positive elements.
        """
        block[block < 0] = 0
        block[block != 0]
        return xp.array([xp.percentile(block.flatten(), self.p)])

    def _internal_aggregate_array_positive(self, block, axis=None, keepdims=False, xp=np):
        """
        Internal method to aggregate positive percentile values.

        Parameters
        ----------
        block : array-like
            Input block to aggregate.
        axis : int, optional
            Axis along which to aggregate.
        keepdims : bool, default=False
            Whether to keep dimensions.
        xp : module, default=np
            Array module (numpy or cupy).

        Returns
        -------
        array-like
            Maximum value from the block.
        """
        block = xp.array(block)

        return xp.array([xp.max(block)])

    def _internal_chunk_array_negative(self, block, axis=None, keepdims=False, xp=np):
        """
        Internal method to compute negative percentile for a chunk.

        Parameters
        ----------
        block : array-like
            Input block to process.
        axis : int, optional
            Axis along which to compute the percentile.
        keepdims : bool, default=False
            Whether to keep dimensions.
        xp : module, default=np
            Array module (numpy or cupy).

        Returns
        -------
        array-like
            Percentile value for negative elements.
        """
        block *= -1
        block[block < 0] = 0
        block[block != 0]
        return xp.array([-xp.percentile(block.flatten(), self.p)])

    def _internal_aggregate_array_negative(self, block, axis=None, keepdims=False, xp=np):
        """
        Internal method to aggregate negative percentile values.

        Parameters
        ----------
        block : array-like
            Input block to aggregate.
        axis : int, optional
            Axis along which to aggregate.
        keepdims : bool, default=False
            Whether to keep dimensions.
        xp : module, default=np
            Array module (numpy or cupy).

        Returns
        -------
        array-like
            Minimum value from the block.
        """
        block = xp.array(block)

        return xp.array([xp.min(block)])

    def _lazy_transform_cpu(self, X):
        """
        CPU lazy transform for clipping by percentiles.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array to clip.

        Returns
        -------
        dask.array.Array
            Clipped array.
        """
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
        """
        GPU lazy transform for clipping by percentiles.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array to clip.

        Returns
        -------
        dask.array.Array
            Clipped array.
        """
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
        """
        CPU transform for clipping by percentiles.

        Parameters
        ----------
        X : array-like
            Input array to clip.

        Returns
        -------
        array-like
            Clipped array.
        """
        pos_cutoff = self._internal_chunk_array_positive(X, xp=np)
        neg_cutoff = self._internal_chunk_array_negative(X, xp=np)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_gpu(self, X):
        """
        GPU transform for clipping by percentiles.

        Parameters
        ----------
        X : array-like
            Input array to clip.

        Returns
        -------
        array-like
            Clipped array.
        """
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
        """
        Initialize the Overlap transform.

        Parameters
        ----------
        pad : tuple, default=(1, 1, 1)
            Padding size for each dimension.
        """
        self._pad = pad

    def _lazy_transform(self, X):
        """
        Apply lazy transform with overlap.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with overlap added.
        """
        return da.overlap.overlap(X, depth=self._pad, boundary="nearest")

    def _lazy_transform_gpu(self, X):
        """
        GPU lazy transform with overlap.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with overlap added.
        """
        return self._lazy_transform(X)

    def _lazy_transform_cpu(self, X):
        """
        CPU lazy transform with overlap.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with overlap added.
        """
        return self._lazy_transform(X)

    def _transform(self, X, xp):
        """
        Apply transform with overlap using padding.

        Parameters
        ----------
        X : array-like
            Input array.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        array-like
            Array with padding added.
        """
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
        """
        GPU transform with overlap.

        Parameters
        ----------
        X : array-like
            Input array.

        Returns
        -------
        array-like
            Array with padding added.
        """
        return self._transform(X, cp)

    def _transform_cpu(self, X):
        """
        CPU transform with overlap.

        Parameters
        ----------
        X : array-like
            Input array.

        Returns
        -------
        array-like
            Array with padding added.
        """
        return self._transform(X, np)


class Trim(Transform):
    """
    Operator to trim dask array that was produced by an Overlap
    transform or subsequent results from that transform.
    """

    def __init__(self, trim=(1, 1, 1)):
        """
        Initialize the Trim transform.

        Parameters
        ----------
        trim : tuple, default=(1, 1, 1)
            Trimming size for each dimension.
        """
        self._trim = trim

    def _lazy_transform(self, X):
        """
        Apply lazy transform to trim overlap.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array with overlap.

        Returns
        -------
        dask.array.Array
            Array with overlap trimmed.
        """
        return da.overlap.trim_overlap(
            X,
            depth=self._trim,
            boundary="nearest",
        )

    def _lazy_transform_gpu(self, X):
        """
        GPU lazy transform to trim overlap.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with overlap trimmed.
        """
        return self._lazy_transform(X)

    def _lazy_transform_cpu(self, X):
        """
        CPU lazy transform to trim overlap.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with overlap trimmed.
        """
        return self._lazy_transform(X)

    def _transform(self, X):
        """
        Apply transform to trim array.

        Parameters
        ----------
        X : array-like
            Input array to trim.

        Returns
        -------
        array-like
            Trimmed array.
        """
        sl = [slice(t, -t) if t != 0 else slice(None, None) for t in self._trim]
        return X[tuple(sl)]

    def _transform_gpu(self, X):
        """
        GPU transform to trim array.

        Parameters
        ----------
        X : array-like
            Input array to trim.

        Returns
        -------
        array-like
            Trimmed array.
        """
        return self._transform(X)

    def _transform_cpu(self, X):
        """
        CPU transform to trim array.

        Parameters
        ----------
        X : array-like
            Input array to trim.

        Returns
        -------
        array-like
            Trimmed array.
        """
        return self._transform(X)


class Roll(Transform):
    """
    Operator to perform a roll along multiple axis
    """

    def __init__(self, shift=(1, 1, 1)):
        """
        Initialize the Roll transform.

        Parameters
        ----------
        shift : tuple, default=(1, 1, 1)
            Number of positions to shift elements along each axis.
        """
        self._shift = shift

    def _transform_generic(self, X, xp):
        """
        Generic transform to roll array elements.

        Parameters
        ----------
        X : array-like
            Input array to roll.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        array-like
            Array with elements rolled.
        """
        return xp.roll(X, shift=self._shift, axis=list(range(len(self._shift))))

    def _lazy_transform_gpu(self, X):
        """
        GPU lazy transform to roll array elements.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with elements rolled.
        """
        return X.map_blocks(self._transform_generic, xp=cp)

    def _lazy_transform_cpu(self, X):
        """
        CPU lazy transform to roll array elements.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.

        Returns
        -------
        dask.array.Array
            Array with elements rolled.
        """
        return X.map_blocks(self._transform_generic, xp=np)

    def _transform_gpu(self, X):
        """
        GPU transform to roll array elements.

        Parameters
        ----------
        X : array-like
            Input array.

        Returns
        -------
        array-like
            Array with elements rolled.
        """
        return self._transform_generic(X, cp)

    def _transform_cpu(self, X):
        """
        CPU transform to roll array elements.

        Parameters
        ----------
        X : array-like
            Input array.

        Returns
        -------
        array-like
            Array with elements rolled.
        """
        return self._transform_generic(X, np)


class ApplyPatchesBase(Transform):
    """
    Base Class for ApplyPatches Functionalities
    """

    def __init__(self, function, weight_function, input_size, overlap, offsets):
        """
        Initialize the ApplyPatchesBase transform.

        Parameters
        ----------
        function : callable or BaseLoader
            Function to be applied to each patch, can be either a Python
            Function or a ModelLoader.
        weight_function : callable
            Weight attribution function, must receive a shape and
            produce a NDArray with the respective weights for each
            array position.
        input_size : tuple
            Size of input to the function to be applied.
        overlap : dict
            Dictionary containing overlapping/padding configurations to use
            with np.pad or dask.overlap.overlap. Its important that for the
            base patch set the whole "chunk core" is covered by the patches.
        offsets : list
            List of offsets for overlapping patches extraction.
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
        Apply function to each patch in a patch set.

        Parameters
        ----------
        patch_set : array-like
            Set of patches to apply function to.

        Returns
        -------
        array-like
            Results of applying function to each patch.

        Raises
        ------
        NotImplementedError
            If the function type is not supported.
        """
        if callable(self._function):
            return np.array(list(map(self._function, patch_set)))
        if isinstance(self._function, BaseLoader):
            return self._function.predict(patch_set)
        raise NotImplementedError("Requested Apply Method not supported")

    def _reconstruct_patches(self, patches, index, weights, inner_dim=None):
        """
        Rearrange patches to reconstruct area of interest from patches and weights.

        Parameters
        ----------
        patches : array-like
            Input patches to reconstruct.
        index : tuple
            Index shape for reconstruction.
        weights : bool
            Whether to use weights in reconstruction.
        inner_dim : int, optional
            Inner dimension size for reconstruction.

        Returns
        -------
        tuple
            Reconstructed array and weight array.
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
        """
        Apply the operation to input data with overlap padding.

        Parameters
        ----------
        X : array-like
            Input data to transform.

        Returns
        -------
        array-like
            Transformed data with operations applied to patches.
        """
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
        """
        Apply lazy transformation with overlap handling for distributed arrays.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array to transform.

        Returns
        -------
        dask.array.Array
            Transformed dask array with operations applied to patches.
        """
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
        """
        CPU lazy transform for applying patches.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dask.array.Array
            Transformed array.
        """
        return self._lazy_transform(X)

    def _lazy_transform_gpu(self, X, **kwargs):
        """
        GPU lazy transform for applying patches.

        Parameters
        ----------
        X : dask.array.Array
            Input dask array.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dask.array.Array
            Transformed array.
        """
        X = X.map_blocks(cp.asnumpy, dtype=X.dtype, meta=np.array((), dtype=X.dtype))
        return self._lazy_transform(X).map_blocks(
            cp.asarray, dtype=X.dtype, meta=cp.array((), dtype=X.dtype)
        )

    def _transform_cpu(self, X, **kwargs):
        """
        CPU transform for applying patches.

        Parameters
        ----------
        X : array-like
            Input array.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array-like
            Transformed array.
        """
        return self._transform(X)

    def _transform_gpu(self, X, **kwargs):
        """
        GPU transform for applying patches.

        Parameters
        ----------
        X : array-like
            Input array.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array-like
            Transformed array.
        """
        X = cp.asnumpy(X)
        return cp.asarray(self._transform(X))


class ApplyPatchesWeightedAvg(ApplyPatchesBase):
    """
    ApplyPatches with Weighted Average combination function.
    """

    def _combine_patches(self, results, offsets, indexes):
        """
        Combine patches using weighted average.

        Parameters
        ----------
        results : list
            List of patch results from different offsets.
        offsets : list
            List of offset values for each patch set.
        indexes : list
            List of index shapes for each patch set.

        Returns
        -------
        array-like
            Combined result using weighted average.
        """
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
        """
        Combine patches using voting method (hard or soft).

        Parameters
        ----------
        results : list
            List of patch results from different offsets.
        offsets : list
            List of offset values for each patch set.
        indexes : list
            List of index shapes for each patch set.

        Returns
        -------
        array-like
            Combined result using the specified voting method.
        """
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
