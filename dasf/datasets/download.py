#!/usr/bin/env python3

from dasf.utils import utils
from dasf.datasets.base import Dataset


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
    def __init__(self, url: str, filename: str, root: str, download: bool = True):
        self.__url = url
        self.__filename = filename

        # Set download as false because this class overrides download()
        Dataset.__init__(self, name="Download Wget",
                         download=download, root=root)

    def download(self):
        """Download the dataset.

        """
        if not self._download or self.__url is None:
            return

        if hasattr(self, "download") and self._download is True:
            self._root_file = utils.download_file(self.__url,
                                                  self.__filename,
                                                  self._root)


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
    def __init__(self, google_file_id, filename, root, download=True):
        self.__google_file_id = google_file_id
        self.__filename = filename

        # Set download as false because this class overrides download()
        Dataset.__init__(self, name="Download Google Drive",
                         download=download, root=root)

    def download(self):
        """Download the dataset.

        """
        if not self._download or self.__google_file_id is None:
            return

        if hasattr(self, "download") and self._download is True:
            self._root_file = \
                utils.download_file_from_gdrive(self.__google_file_id,
                                                self.__filename,
                                                self._root)
