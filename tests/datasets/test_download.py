#!/usr/bin/env python3

import unittest

from unittest.mock import Mock, patch

from dasf.datasets.download import DownloadGDrive, DownloadWget


class TestDownloadWget(unittest.TestCase):
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download(self, gdown):
        gdown.download = Mock(return_value=None)

        download = DownloadWget(url='https://www.foo.com/',
                                filename='bar',
                                root='/tmp/baz')

        download.download()

        gdown.download.assert_called()

    @patch('dasf.utils.funcs.is_notebook', Mock(return_value=True))
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download_in_notebook(self, gdown):
        gdown.download = Mock(return_value=None)

        download = DownloadWget(url='https://www.foo.com/',
                                filename='bar',
                                root='/tmp/baz')

        download.download()

        gdown.download.assert_called()

    @patch('dasf.utils.funcs.is_notebook', Mock(return_value=True))
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download_false(self, gdown):
        gdown.download = Mock(return_value=None)

        download = DownloadWget(url='https://www.foo.com/',
                                filename='bar',
                                root='/tmp/baz',
                                download=False)

        download.download()

        gdown.download.assert_not_called()

    @patch('dasf.utils.funcs.is_notebook', Mock(return_value=True))
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download_check(self, gdown):
        def download_check(cls):
            return True

        gdown.download = Mock(return_value=None)

        download = DownloadWget(url='https://www.foo.com/',
                                filename='bar',
                                root='/tmp/baz',
                                download=True)

        download._download_check = download_check.__get__(download)

        download.download()

        gdown.download.assert_called()


class TestDownloadGDrive(unittest.TestCase):
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download(self, gdown):
        gdown.download = Mock(return_value=None)

        download = DownloadGDrive(google_file_id='1a2c3b4d5e',
                                  filename='bar',
                                  root='/tmp/baz')

        download.download()

        gdown.download.assert_called()

    @patch('dasf.utils.funcs.is_notebook', Mock(return_value=True))
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download_in_notebook(self, gdown):
        gdown.download = Mock(return_value=None)

        download = DownloadGDrive(google_file_id='1a2c3b4d5e',
                                  filename='bar',
                                  root='/tmp/baz')

        download.download()

        gdown.download.assert_called()

    @patch('dasf.utils.funcs.is_notebook', Mock(return_value=True))
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download_false(self, gdown):
        gdown.download = Mock(return_value=None)

        download = DownloadGDrive(google_file_id='1a2c3b4d5e',
                                  filename='bar',
                                  root='/tmp/baz',
                                  download=False)

        download.download()

        gdown.download.assert_not_called()

    @patch('dasf.utils.funcs.is_notebook', Mock(return_value=True))
    @patch('dasf.utils.funcs.gdown', return_value=Mock())
    def test_download_check(self, gdown):
        def download_check(cls):
            return True

        gdown.download = Mock(return_value=None)

        download = DownloadGDrive(google_file_id='1a2c3b4d5e',
                                  filename='bar',
                                  root='/tmp/baz',
                                  download=True)

        download._download_check = download_check.__get__(download)

        download.download()

        gdown.download.assert_called()
