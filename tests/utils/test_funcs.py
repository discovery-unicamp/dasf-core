#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch
import os

from dasf.utils.funcs import (
    download_file,
    download_file_from_gdrive,
    is_kvikio_supported,
    is_gds_supported,
    is_kvikio_compat_mode,
    is_nvcomp_codec_supported,
    get_gpu_from_workers,
    get_dask_running_client,
    weight_gaussian,
    weight_radial,
)


class TestDownloadFunctions(unittest.TestCase):
    @patch('dasf.utils.funcs.is_notebook', return_value=False)
    @patch('gdown.download')
    @patch('os.path.exists', return_value=False)
    def test_download_file_basic(self, mock_exists, mock_gdown, mock_notebook):
        mock_gdown.return_value = "test_file.txt"

        result = download_file("http://example.com/file.txt", "test_file.txt")

        mock_gdown.assert_called_once_with(
                "http://example.com/file.txt",
                output=os.path.abspath(os.path.join(os.getcwd(),
                                                    "test_file.txt")))
        self.assertTrue(result.endswith("test_file.txt"))

    @patch('dasf.utils.funcs.is_notebook', return_value=False)
    @patch('gdown.download')
    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    def test_download_file_with_directory(self,
                                          mock_makedirs,
                                          mock_exists,
                                          mock_gdown,
                                          mock_notebook):
        mock_gdown.return_value = "test_file.txt"

        result = download_file("http://example.com/file.txt",
                               "test_file.txt", "/tmp/test")

        mock_makedirs.assert_called_once()
        mock_gdown.assert_called_once()
        self.assertTrue(result.endswith("test_file.txt"))

    @patch('dasf.utils.funcs.is_notebook', return_value=False)
    @patch('gdown.download')
    @patch('os.path.exists', return_value=True)
    def test_download_file_already_exists(self, mock_exists, mock_gdown, mock_notebook):
        _ = download_file("http://example.com/file.txt", "existing_file.txt")

        mock_gdown.assert_not_called()

    @patch('dasf.utils.funcs.is_notebook', return_value=True)
    @patch('dasf.utils.funcs.NotebookProgressBar')
    @patch('gdown.download')
    @patch('os.path.exists', return_value=False)
    def test_download_file_with_notebook(self,
                                         mock_exists,
                                         mock_gdown,
                                         mock_progressbar,
                                         mock_notebook):
        mock_pbar = Mock()
        mock_progressbar.return_value = mock_pbar
        mock_gdown.return_value = "test_file.txt"

        _ = download_file("http://example.com/file.txt", "test_file.txt")

        mock_pbar.show.assert_called_once()
        mock_pbar.start.assert_called_once()
        mock_pbar.set_current.assert_called_with(100, 100)

    def test_download_file_from_gdrive(self):
        expected_url = ("https://drive.google.com/"
                        "uc?export=download&confirm=9iBg&id=file123")

        with patch('dasf.utils.funcs.download_file') as mock_download:
            download_file_from_gdrive("file123", "test.txt", "/tmp")

            mock_download.assert_called_once_with(expected_url,
                                                  filename="test.txt",
                                                  directory="/tmp")


class TestKvikioFunctions(unittest.TestCase):
    @patch('dasf.utils.funcs.KVIKIO_SUPPORTED', True)
    def test_is_kvikio_supported_true(self):
        self.assertTrue(is_kvikio_supported())

    @patch('dasf.utils.funcs.KVIKIO_SUPPORTED', False)
    def test_is_kvikio_supported_false(self):
        self.assertFalse(is_kvikio_supported())

    @patch('dasf.utils.funcs.is_kvikio_supported', return_value=True)
    def test_is_gds_supported_true(self, mock_kvikio_supported):
        with patch('dasf.utils.funcs.kvikio', create=True) as mock_kvikio:
            mock_props = Mock()
            mock_props.is_gds_available = True
            mock_kvikio.DriverProperties.return_value = mock_props

            self.assertTrue(is_gds_supported())

    @patch('dasf.utils.funcs.is_kvikio_supported', return_value=False)
    def test_is_gds_supported_false_no_kvikio(self, mock_kvikio_supported):
        self.assertFalse(is_gds_supported())

    def test_is_kvikio_compat_mode(self):
        with patch('dasf.utils.funcs.kvikio', create=True) as mock_kvikio:
            mock_kvikio.defaults.compat_mode.return_value = True
            self.assertTrue(is_kvikio_compat_mode())

            mock_kvikio.defaults.compat_mode.return_value = False
            self.assertFalse(is_kvikio_compat_mode())

    @patch('dasf.utils.funcs.NV_COMP_BATCH_CODEC_SUPPORTED', True)
    def test_is_nvcomp_codec_supported_true(self):
        self.assertTrue(is_nvcomp_codec_supported())

    @patch('dasf.utils.funcs.NV_COMP_BATCH_CODEC_SUPPORTED', False)
    def test_is_nvcomp_codec_supported_false(self):
        self.assertFalse(is_nvcomp_codec_supported())


class TestDaskClientFunctions(unittest.TestCase):
    @patch('dask.distributed.Client.current')
    def test_get_dask_running_client_success(self, mock_current):
        mock_client = Mock()
        mock_current.return_value = mock_client

        result = get_dask_running_client()
        self.assertEqual(result, mock_client)

    @patch('dask.distributed.Client.current', side_effect=Exception("No client"))
    def test_get_dask_running_client_exception(self, mock_current):
        result = get_dask_running_client()
        self.assertIsNone(result)

    @patch('dasf.utils.funcs.get_dask_running_client')
    def test_get_gpu_from_workers_success(self, mock_get_client):
        mock_client = Mock()
        mock_client.cluster.scheduler_info = {
            "workers": {
                "worker1": {"gpu": [0]},
                "worker2": {"host": "localhost"}
            }
        }
        mock_get_client.return_value = mock_client

        result = get_gpu_from_workers()
        self.assertTrue(result)

    @patch('dasf.utils.funcs.get_dask_running_client')
    def test_get_gpu_from_workers_no_gpu(self, mock_get_client):
        mock_client = Mock()
        mock_client.cluster.scheduler_info = {
            "workers": {
                "worker1": {"host": "localhost"},
                "worker2": {"host": "localhost"}
            }
        }
        mock_get_client.return_value = mock_client

        result = get_gpu_from_workers()
        self.assertFalse(result)

    @patch('dasf.utils.funcs.get_dask_running_client', return_value=None)
    def test_get_gpu_from_workers_no_client(self, mock_get_client):
        result = get_gpu_from_workers()
        self.assertFalse(result)

    @patch('dasf.utils.funcs.get_dask_running_client')
    def test_get_gpu_from_workers_exception(self, mock_get_client):
        mock_client = Mock()
        mock_client.cluster.scheduler_info = {"invalid": "data"}
        mock_get_client.return_value = mock_client

        result = get_gpu_from_workers()
        self.assertFalse(result)


class TestWeightFunctions(unittest.TestCase):
    def test_weight_gaussian_2d(self):
        shape = (4, 4)
        result = weight_gaussian(shape)

        self.assertEqual(result.shape, shape)
        self.assertTrue(isinstance(result, type(result)))  # numpy array

        # Check that center has highest value
        center_value = result[2, 2]
        corner_value = result[0, 0]
        self.assertGreater(center_value, corner_value)

    def test_weight_gaussian_1d(self):
        shape = (10,)
        result = weight_gaussian(shape)

        self.assertEqual(result.shape, shape)
        self.assertTrue(len(result.shape) == 1)

        # Check that center has highest value
        center_value = result[5]
        edge_value = result[0]
        self.assertGreater(center_value, edge_value)

    def test_weight_gaussian_3d(self):
        shape = (3, 3, 3)
        result = weight_gaussian(shape)

        self.assertEqual(result.shape, shape)
        # Check that center has highest value
        center_value = result[1, 1, 1]
        corner_value = result[0, 0, 0]
        self.assertGreater(center_value, corner_value)

    def test_weight_radial_2d(self):
        shape = (4, 4)
        result = weight_radial(shape)

        self.assertEqual(result.shape, shape)

        # Check that center has highest value (should be 1.0)
        center_value = result[2, 2]
        corner_value = result[0, 0]
        self.assertGreater(center_value, corner_value)
        self.assertEqual(center_value, 1.0)

    def test_weight_radial_1d(self):
        shape = (6,)
        result = weight_radial(shape)

        self.assertEqual(result.shape, shape)
        # For 1D with even length, center should be at 2.5, so check both middle indices
        # Check that middle values are greater than edge values
        middle_value1 = result[2]
        middle_value2 = result[3]
        edge_value = result[0]
        self.assertGreater(max(middle_value1, middle_value2), edge_value)

    def test_weight_radial_symmetric(self):
        shape = (5, 5)
        result = weight_radial(shape)

        # Check symmetry - corners equidistant from center should have same values
        # (0,4) and (4,0) are equidistant from center (2.5, 2.5)
        self.assertAlmostEqual(result[0, 4], result[4, 0], places=6)
        # Check that center has higher values than edges
        center_value = result[2, 2]
        corner_value = result[0, 0]
        self.assertGreater(center_value, corner_value)


class TestProgressBarIntegration(unittest.TestCase):
    def test_notebook_progress_bar_initialization(self):
        from dasf.utils.funcs import NotebookProgressBar

        pbar = NotebookProgressBar()
        self.assertIsNone(pbar.bar)
        self.assertIsNone(pbar.percentage)
        self.assertIsNone(pbar.data)
        self.assertEqual(pbar._NotebookProgressBar__current, NotebookProgressBar.MIN_CUR)
        self.assertEqual(pbar._NotebookProgressBar__total, NotebookProgressBar.MIN_TOTAL)

    @patch('dasf.utils.funcs.FloatProgress')
    @patch('dasf.utils.funcs.Label')
    @patch('dasf.utils.funcs.HBox')
    @patch('dasf.utils.funcs.disp.display')
    def test_notebook_progress_bar_show(self,
                                        mock_display,
                                        mock_hbox,
                                        mock_label,
                                        mock_float_progress):
        from dasf.utils.funcs import NotebookProgressBar

        mock_bar = Mock()
        mock_percentage = Mock()
        mock_data = Mock()
        mock_float_progress.return_value = mock_bar
        mock_label.side_effect = [mock_percentage, mock_data]
        mock_box = Mock()
        mock_hbox.return_value = mock_box

        pbar = NotebookProgressBar()
        pbar.show()

        mock_float_progress.assert_called_once_with(value=0, min=0, max=100)
        self.assertEqual(mock_label.call_count, 2)
        mock_hbox.assert_called_once()
        mock_display.assert_called_once_with(mock_box)

    def test_notebook_progress_bar_set_current(self):
        from dasf.utils.funcs import NotebookProgressBar

        pbar = NotebookProgressBar()
        pbar.set_current(50, 100)

        self.assertEqual(pbar._NotebookProgressBar__current, 50)
        self.assertEqual(pbar._NotebookProgressBar__total, 100)

    def test_notebook_progress_bar_set_error(self):
        from dasf.utils.funcs import NotebookProgressBar

        pbar = NotebookProgressBar()
        pbar.set_error(True)

        self.assertTrue(pbar._NotebookProgressBar__error)
