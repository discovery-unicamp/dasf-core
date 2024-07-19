#!/usr/bin/env python3

""" Download module for datasets. """

from dasf.datasets.base import Dataset
from dasf.utils.funcs import download_file, download_file_from_gdrive


class DownloadWget(Dataset):
    """Dataset downloadable via wget.

    Parameters
    ----------
    url : str
        The url to fetch the resource.
    filename : str
        Name of the file.
    root : str
        Directory to store the downloaded file.
    download : bool
        If it the dataset must be downloaded (the default is True).

    """
    def __init__(self,
                 url: str,
                 filename: str,
                 root: str,
                 download: bool = True):
        """ Constructor of the object DownloadWget. """
        self.__url = url
        self.__filename = filename

        # Set download as false because this class overrides download()
        Dataset.__init__(self, name="Download Wget", download=download, root=root)

    def download(self):
        """ Download the dataset. """
        if not self._download or self.__url is None:
            return

        if hasattr(self, "download") and self._download is True:
            self._root_file = download_file(
                self.__url, self.__filename, self._root
            )

            if hasattr(self, "_download_check") and callable(self._download_check):
                self._download_check()


class DownloadGDrive(Dataset):
    """Dataset downloadable via Google Drive.

    Parameters
    ----------
    google_file_id : str
        Id of the google drive resource.
    filename : str
        Name of the file.
    root : str
        Directory to store the downloaded file.
    download : bool
        If it the dataset must be downloaded (the default is True).

    """
    def __init__(self,
                 google_file_id: str,
                 filename: str,
                 root: str,
                 download: bool = True):
        """ Constructor of the object DownloadGDrive. """
        self.__google_file_id = google_file_id
        self.__filename = filename

        # Set download as false because this class overrides download()
        Dataset.__init__(
            self, name="Download Google Drive", download=download, root=root
        )

    def download(self):
        """ Download the dataset. """
        if not self._download or self.__google_file_id is None:
            return

        if hasattr(self, "download") and self._download is True:
            self._root_file = download_file_from_gdrive(
                self.__google_file_id, self.__filename, self._root
            )

            if hasattr(self, "_download_check") and callable(self._download_check):
                self._download_check()
