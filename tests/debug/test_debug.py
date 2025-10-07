#!/usr/bin/env python3

import os
import unittest

import dask.array as da
import dask.dataframe as ddf
import numpy as np
import pandas as pd

try:
    import GPUtil
    if len(GPUtil.getGPUs()) == 0:  # check if GPU are available in current env
        raise ImportError("There is no GPU available here")
    import cudf
    import cupy as cp
    import dask_cudf as cuddf
except:
    pass

from io import StringIO

from IPython.core.display import HTML
from mock import ANY, Mock, patch

from dasf.debug import Debug, VisualizeDaskData
from dasf.utils.funcs import is_gpu_supported


class TestDebug(unittest.TestCase):
    def test_debug_array_cpu(self):
        data = np.random.random((40, 40, 40))

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (40, 40, 40)", fake_out.getvalue())
            self.assertIn("Datatype is: <class 'numpy.ndarray'>", fake_out.getvalue()) 
            self.assertIn("Data content is: [[[", fake_out.getvalue())

    def test_debug_dask_array_cpu(self):
        data = da.random.random((40, 40, 40), chunks=(5, 5, 5))

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (40, 40, 40)", fake_out.getvalue())
            self.assertIn("Datatype is: <class 'dask.array.core.Array'>", fake_out.getvalue())
            self.assertIn("Data content is: dask.array<random_sample, shape=(40, 40, 40), dtype=float64, chunksize=(5, 5, 5), chunktype=numpy.ndarray>", fake_out.getvalue())

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_debug_array_gpu(self):
        data = cp.random.random((40, 40, 40))

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (40, 40, 40)", fake_out.getvalue())
            self.assertIn("Datatype is: <class 'cupy.ndarray'>", fake_out.getvalue())
            self.assertIn("Data content is: [[[", fake_out.getvalue())

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_debug_dask_array_gpu(self):
        data = da.from_array(cp.random.random((40, 40, 40)), chunks=(5, 5, 5))

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (40, 40, 40)", fake_out.getvalue())
            self.assertIn("Datatype is: <class 'dask.array.core.Array'>", fake_out.getvalue())
            self.assertIn("Data content is: dask.array<array, shape=(40, 40, 40), dtype=float64, chunksize=(5, 5, 5), chunktype=cupy.ndarray>", fake_out.getvalue())

    def test_debug_dataframe_cpu(self):
        data = pd.DataFrame(np.random.random((3, 4)), columns=['A', 'B', 'C', 'D'])

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (3, 4)", fake_out.getvalue())
            self.assertIn("Datatype is: <class 'pandas.core.frame.DataFrame'>", fake_out.getvalue())
            self.assertIn("Data content is:           A         B         C         D", fake_out.getvalue())

    def test_debug_dask_dataframe_cpu(self):
        data = ddf.from_pandas(pd.DataFrame(np.random.random((3, 4)), columns=['A', 'B', 'C', 'D']), npartitions=3)

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (<dask_expr.expr.Scalar: expr=df.size() // 4, dtype=int64>, 4)",
                          fake_out.getvalue())
            self.assertIn("Datatype is: <class 'dask_expr._collection.DataFrame'>", fake_out.getvalue())
            self.assertIn("Data content is: Dask DataFrame Structure:\n"
                          "                     A        B        C        D\n"
                          "npartitions=3                                    \n", fake_out.getvalue())

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_debug_dataframe_gpu(self):
        data = cudf.DataFrame(cp.random.random((3, 4)), columns=['A', 'B', 'C', 'D'])

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (3, 4)", fake_out.getvalue())
            self.assertIn("Datatype is: <class 'cudf.core.dataframe.DataFrame'>", fake_out.getvalue())
            self.assertIn("Data content is:           A         B         C         D", fake_out.getvalue())

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_debug_dask_dataframe_gpu(self):
        data = cuddf.from_cudf(cudf.DataFrame(cp.random.random((3, 4)), columns=['A', 'B', 'C', 'D']), npartitions=3)

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datashape is: (<dask_expr.expr.Scalar: expr=df.size() // 4, dtype=int64>, 4)",
                          fake_out.getvalue())
            self.assertIn("Datatype is: <class 'dask_cudf.expr._collection.DataFrame'>", fake_out.getvalue())
            self.assertIn("Data content is: Dask DataFrame Structure:\n"
                          "                     A        B        C        D\n"
                          "npartitions=3                                    \n", fake_out.getvalue())

    def test_debug_list(self):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        debug = Debug()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            debug.display(X=data)
            self.assertIn("Datatype is: <class 'list'>", fake_out.getvalue())
            self.assertIn("Data content is: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", fake_out.getvalue())

    @patch('dasf.debug.debug.idisplay', return_value=Mock())
    @patch('dasf.debug.debug.is_notebook', Mock(return_value=True))
    def test_debug_dask_array_ipython(self, idisplay):
        data = da.random.random((40, 40, 40), chunks=(5, 5, 5))

        debug = Debug()

        debug.display(X=data)

        idisplay.assert_called_once_with(ANY)
        self.assertTrue(isinstance(idisplay.call_args.args[0], HTML))

    @patch('dasf.debug.debug.idisplay', return_value=Mock())
    @patch('dasf.debug.debug.is_notebook', Mock(return_value=True))
    def test_debug_dask_dataframe_ipython(self, idisplay):
        data = ddf.from_pandas(pd.DataFrame(np.random.random((3, 4)), columns=['A', 'B', 'C', 'D']), npartitions=3)

        print(type(data))

        debug = Debug()

        debug.display(X=data)

        idisplay.assert_called_once_with(ANY)
        self.assertTrue(isinstance(idisplay.call_args.args[0], HTML))


class TestVisualizeDaskData(unittest.TestCase):
    def test_visualize_dask_array_cpu(self):
        data = da.random.random((40, 40, 40), chunks=(5, 5, 5))

        visualize = VisualizeDaskData()

        with patch.object(data, 'visualize', wraps=data.visualize) as wrapped_visualize:
            visualize.display(X=data)
            wrapped_visualize.assert_called_once()

    def test_visualize_dask_dataframe_cpu(self):
        data = ddf.from_pandas(pd.DataFrame(np.random.random((3, 4)), columns=['A', 'B', 'C', 'D']), npartitions=3)

        visualize = VisualizeDaskData()

        with patch.object(data, 'visualize', wraps=data.visualize) as wrapped_visualize:
            visualize.display(X=data)
            wrapped_visualize.assert_called_once()

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_visualize_dask_array_gpu(self):
        data = da.from_array(cp.random.random((40, 40, 40)), chunks=(5, 5, 5))

        visualize = VisualizeDaskData()

        with patch.object(data, 'visualize', wraps=data.visualize) as wrapped_visualize:
            visualize.display(X=data)
            wrapped_visualize.assert_called_once()

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_visualize_dask_dataframe_gpu(self):
        data = cuddf.from_cudf(cudf.DataFrame(cp.random.random((3, 4)), columns=['A', 'B', 'C', 'D']), npartitions=3)

        visualize = VisualizeDaskData()

        with patch.object(data, 'visualize', wraps=data.visualize) as wrapped_visualize:
            visualize.display(X=data)
            wrapped_visualize.assert_called_once()

    def test_visualize_dask_array_cpu_with_filename(self):
        data = da.random.random((40, 40, 40), chunks=(5, 5, 5))

        visualize = VisualizeDaskData(filename="/tmp/bar")

        with patch.object(data, 'visualize', wraps=data.visualize) as wrapped_visualize:
            visualize.display(X=data)
            wrapped_visualize.assert_called_once_with("/tmp/bar")

            os.remove("/tmp/bar.png")

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_visualize_dask_array_gpu_with_filename(self):
        data = da.from_array(cp.random.random((40, 40, 40)), chunks=(5, 5, 5))

        visualize = VisualizeDaskData(filename="/tmp/bar")

        with patch.object(data, 'visualize', wraps=data.visualize) as wrapped_visualize:
            visualize.display(X=data)
            wrapped_visualize.assert_called_once_with("/tmp/bar")

            os.remove("/tmp/bar.png")

    def test_visualize_array_cpu(self):
        data = np.random.random((40, 40, 40))

        visualize = VisualizeDaskData()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            visualize.display(X=data)
            self.assertIn("WARNING: This is not a Dask element.", fake_out.getvalue())

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_visualize_array_gpu(self):
        data = cp.random.random((40, 40, 40))

        visualize = VisualizeDaskData()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            visualize.display(X=data)
            self.assertIn("WARNING: This is not a Dask element.", fake_out.getvalue())
