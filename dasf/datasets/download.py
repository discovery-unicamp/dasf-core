#!/usr/bin/env python3

from dasf.utils import utils
from dasf.datasets.base import Dataset


class DownloadWget(Dataset):
    def __init__(self, url, filename, root, download=True):
        self.__url = url
        self.__filename = filename

        # Set download as false because this class overrides download()
        Dataset.__init__(self, name="Download Wget", download=download, root=root)

    def download(self):
        if not self._download or self.__url is None:
            return

        if hasattr(self, "download") and self._download is True:
            self._root_file = utils.download_file(
                self.__url, self.__filename, self._root
            )


class DownloadGDrive(Dataset):
    def __init__(self, google_file_id, filename, root, download=True):
        self.__google_file_id = google_file_id
        self.__filename = filename

        # Set download as false because this class overrides download()
        Dataset.__init__(
            self, name="Download Google Drive", download=download, root=root
        )

    def download(self):
        if not self._download or self.__google_file_id is None:
            return

        if hasattr(self, "download") and self._download is True:
            self._root_file = utils.download_file_from_gdrive(
                self.__google_file_id, self.__filename, self._root
            )
