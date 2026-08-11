"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os.path
import re
import numpy as np
import xarray as xr
import fsspec
import s3fs
import pandas as pd
import warnings
from typing import Union, Any
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import logging
from pprint import pprint
import time
from posixpath import join as posix_join
from os.path import join as os_join

from MAST_tools.utils.data_utils import (
    ShotInfo,
    BaseDataSourceType,
    ShotInfoType,
    DataFormatType,
    DataSourceHandleType,
)
from MAST_tools.utils.general_utils import get_random_string, warning_print
from MAST_tools.utils.path_utils import DEFAULT_SIGNAL_AVAILABILITY_FILE


# ----------------------------------------------------------------------------------------------------------------------

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ----------------------------------------------------------------------------------------------------------------------
# Default values

DEFAULT_BASE_FSSPEC_PROTOCOL = "simplecache"
DEFAULT_TARGET_FSSPEC_PROTOCOL = "s3"
DEFAULT_S3_ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
DEFAULT_S3_MAST_DATASET_PATH = "/mast/tokamark/v1"
DEFAULT_BASE_LOCAL_DATA_PATH = "/mast/tokamark/v1"  # <- Replace default value if different installation dir is used.

DEFAULT_BASE_LOCAL_ZARR_PATH = DEFAULT_BASE_LOCAL_DATA_PATH
"""Deprecated alias for `DEFAULT_BASE_LOCAL_DATA_PATH`, kept for backwards compatibility."""

DEFAULT_LOCAL_FLAG_VALUE = False

DEFAULT_DATA_FORMAT: DataFormatType = "zarr"
SUPPORTED_DATA_FORMATS: tuple[DataFormatType, ...] = ("zarr", "netcdf")
FORMAT_FILE_EXTENSIONS: dict[DataFormatType, str] = {"zarr": ".zarr", "netcdf": ".nc"}
NETCDF_ENGINE = "h5netcdf"
FORMAT_ENGINES: dict[DataFormatType, str] = {"zarr": "zarr", "netcdf": NETCDF_ENGINE}
"""`xarray` engine used to read each supported data format."""

_REMOTE_URI_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


# ----------------------------------------------------------------------------------------------------------------------
def is_remote_uri(handle: Any) -> bool:
    """
    Check whether a data source handle is a remote URI (e.g. "s3://mast/tokamark/v1/30471.zarr").

    Parameters
    ----------
    handle : Any
        Data source handle to be checked. Non-str handles (local paths as `os.PathLike`, file-like objects) are
        never considered remote URIs.

    Returns
    -------
    bool
        True if `handle` is a str with a URI scheme prefix, False otherwise.

    """

    return isinstance(handle, str) and bool(_REMOTE_URI_PATTERN.match(handle))


# ======================================================================================================================
class MASTStorageManager:
    """
    Class with storage management tools for MAST data.

    Attributes
    ----------
    data_format : DataFormatType
        On-disk/S3 data format to load, either "zarr" or "netcdf". Both are read via `xarray`.
    base_fsspec_protocol : str
        Base protocol used by 'fsspec'.
    target_fsspec_protocol : str
        Target filesystem protocol for the selected base 'fsspec' protocol.
    s3_endpoint_url : str
        Endpoint of the cloud S3 bucket used for remote data pulling.
    s3_mast_dataset_path : str
        Path for the target MAST dataset within the configured S3 bucket.
    base_local_data_path : str | None
        Local root path used for local data pulling.
    fs_local_fsspec : fsspec.implementations.local.LocalFileSystem
        A LocalFileSystem instance.
    fs_remote_fsspec : Any
        A filesystem instance via fsspec.filesystem.
    fs_remote_s3fs : Any
        A filesystem instance via s3fs.S3FileSystem.
    store_manager_id : str
        User-defined storage manager ID.

    Methods
    -------
    _validate_base_local_data_path()
        Check if local path for the data database is set, raising SystemError if not.
    _check_local_data_database()
        Run access checks for a potential local data database.
    _is_digit(item)
        Check if provided item is of type digit.
    _shot_uri(shot_id, local)
        Build the local path or remote URI for a given shot.
    _open_kwargs(handle)
        Build the `xarray` open keyword arguments for a given data source handle.
    _parse_shot_info_dict(shot_info)
        Parse dictionary with shot information.
    _check_shot_id(shot_id)
        Check shot ID.
    _check_list_of_shot_ids(shot_ids)
        Check list of shot IDs.
    _create_fs_remote(library, warn)
        Create filesystem instance either using 'fsspec' or 's3fs' for remote data request.
    _read_fsspec_listdir(path, local)
        Evaluate the list dir method of 'fsspec' filesystem instance on the provided path.
    check_data_origin(data_origin)
        Check MAST data origin.
    list_all_shots(local)
        Get a list of available MAST shot IDs.
    list_shots_by_signal_availability(availability_data_file_path, required_signals)
        List shot IDs following composite condition for signal availability and given availability file.
    get_all_sources(shot_ids, local)
        Return a dictionary with all available sources per shot ID.
    get_all_signals(shot_ids, local, verbose)
        Return a dictionary with all available signals per shot ID.
    make_shot_store(shot_info, verbose)
        Make a data source handle (local path or remote URI) for a given target shot.
    make_shot_group(data_origin, verbose)
        Make an `xarray.DataTree` for a shot from data origin (either a data source handle or shot info).
    open_shot_group(data_origin, verbose)
        Context manager around `make_shot_group()`, closing the tree on exit if it was opened by this call.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        base_fsspec_protocol: str = DEFAULT_BASE_FSSPEC_PROTOCOL,
        target_fsspec_protocol: str = DEFAULT_TARGET_FSSPEC_PROTOCOL,
        s3_endpoint_url: str = DEFAULT_S3_ENDPOINT_URL,
        s3_mast_dataset_path: str = DEFAULT_S3_MAST_DATASET_PATH,
        base_local_data_path: str | None = DEFAULT_BASE_LOCAL_DATA_PATH,
        data_format: DataFormatType = DEFAULT_DATA_FORMAT,
        base_local_zarr_path: str | None = None,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        data_format : DataFormatType
            On-disk/S3 data format to load, either "zarr" or "netcdf". Both are loaded via xarray; netCDF reads use
            the `h5netcdf` engine, which supports both local paths and remote file-like objects (unlike `netCDF4`,
            which cannot reliably read from S3 file-like objects).
            Default: MAST_tools.utils.store_utils.DEFAULT_DATA_FORMAT.
        base_fsspec_protocol: str
            Base protocol used by 'fsspec'. Some supported protocols include:
            `blockcache`:
                - With this option, data is downloaded block-wise.
                - Restrictions:
                    - It has a storage/OS combination which supports sparse files.
                    - The backend implementation uses files which derive from AbstractBufferedFile.
                    - The library you pass the resultant object to accepts generic python file-like objects.
            `filecache`:
                - Works for all file system implementations, and provides a real local file for other libraries to use.
            `simplecache`:
                - Same as `filecache`, except without options for cache expiry and to check original source.
                - Only option guaranteed to be thread/process-safe.
            Full list of supported protocols obtained via `fsspec.available_protocols()`.
            More info available at https://filesystem-spec.readthedocs.io/en/latest/features.html.
            Default: MAST_tools.utils.store_utils.DEFAULT_BASE_FSSPEC_PROTOCOL.
        target_fsspec_protocol : str
            Target filesystem protocol for the selected base 'fsspec' protocol.
            Default: MAST_tools.utils.store_utils.DEFAULT_TARGET_FSSPEC_PROTOCOL.
        s3_endpoint_url : str
            Endpoint of the cloud S3 bucket used for remote data pulling (i.e., for local=False in some methods).
            Default: MAST_tools.utils.store_utils.DEFAULT_S3_ENDPOINT_URL.
        s3_mast_dataset_path : str
            Path for the target MAST dataset within the configured S3 bucket.
            Default: MAST_tools.utils.store_utils.DEFAULT_S3_MAST_DATASET_PATH.
        base_local_data_path : str | None
            Local root path used for local data pulling.
            Default: MAST_tools.utils.store_utils.DEFAULT_BASE_LOCAL_DATA_PATH.
        base_local_zarr_path : str | None
            Deprecated alias for `base_local_data_path`. If set, it takes precedence and a DeprecationWarning is
            emitted.
            Optional. Default: None.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Raises
        ------
        FileNotFoundError
            If provided `base_local_data_path` directory is not found.
        ValueError
            If provided `data_format` is not one of `SUPPORTED_DATA_FORMATS`.

        """

        if data_format not in SUPPORTED_DATA_FORMATS:
            raise ValueError(f"Invalid `data_format` '{data_format}': it must be one of {SUPPORTED_DATA_FORMATS}.")
        self.data_format = data_format

        if base_local_zarr_path is not None:
            warnings.warn(
                "Parameter `base_local_zarr_path` is deprecated: use `base_local_data_path` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            base_local_data_path = base_local_zarr_path

        self.base_fsspec_protocol = base_fsspec_protocol
        self.target_fsspec_protocol = target_fsspec_protocol
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_mast_dataset_path = s3_mast_dataset_path
        self.base_local_data_path = base_local_data_path

        self._check_local_data_database()
        self.fs_local_fsspec = fsspec.filesystem("file")
        self.fs_remote_fsspec = self._create_fs_remote(library="fsspec")
        self.fs_remote_s3fs = self._create_fs_remote(library="s3fs")

        self.store_manager_id = f"store_manager_{get_random_string(4)}"

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def engine(self) -> str:
        """`xarray` engine matching the configured data format."""

        return FORMAT_ENGINES[self.data_format]

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def base_local_zarr_path(self) -> str | None:
        """Deprecated alias for `self.base_local_data_path`, kept for backwards compatibility."""

        return self.base_local_data_path

    # ------------------------------------------------------------------------------------------------------------------
    def _validate_base_local_data_path(self):
        """Check if local path for the data database is set, raising SystemError if not."""
        if self.base_local_data_path is None:
            raise SystemError(
                "No path for a local data database was set during the creation of the MASTStorageManager instance."
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _check_local_data_database(self):
        """Run access checks for a potential local data database (Zarr or netCDF, per `self.data_format`)."""

        file_extension = FORMAT_FILE_EXTENSIONS[self.data_format]
        format_label = self.data_format.capitalize()

        warning_message = ""
        if self.base_local_data_path is None:
            warning_message = (
                f"No path for local {format_label} database was set during the creation of the MASTStorageManager "
                f"instance."
            )
        else:
            # Warn about inexistent path
            if not os.path.isdir(self.base_local_data_path):
                warning_message = (
                    f"The path `{self.base_local_data_path}` for a local {format_label} database, which was set "
                    f"during the creation of the MASTStorageManager instance, does not correspond to a valid local "
                    f"directory."
                )
            else:
                data_files = [
                    file_ for file_ in os.listdir(self.base_local_data_path) if file_.endswith(file_extension)
                ]
                if len(data_files):
                    print(
                        f"[INFO] Local {format_label} database identified under `{self.base_local_data_path}` with "
                        f"{len(data_files)} {format_label} files in it."
                    )
                else:
                    # Warn about no data files found
                    warning_message = (
                        f"No {format_label} files found under the path `{self.base_local_data_path}`, which was set "
                        f"during the creation of the MASTStorageManager instance."
                    )
        if warning_message:
            additional_warning = (
                f"This will cause local pipelines to fail. To avoid this, either provide a valid installation path for "
                f"a local {format_label} database, or use default MASTStorageManager settings and install the "
                f"database under the default directory `{DEFAULT_BASE_LOCAL_DATA_PATH}`."
            )
            warning_print(f"\n{warning_message} {additional_warning}\n")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _is_digit(item: Any) -> bool:
        """
        Check if provided item is of type digit.

        Parameters
        ----------
        item : Any
            An arbitrary input item.

        Returns
        -------
        bool
            True if `item` is digit (either of int type, or of type str representing a digit), False otherwise.

        """

        is_digit = False
        if isinstance(item, str):
            if item.isdigit():
                is_digit = True
        elif isinstance(item, (int, np.int_)):
            is_digit = True

        return is_digit

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _is_file_like(item: Any) -> bool:
        """Duck-type check for a file-like object (e.g. an `s3fs`/`fsspec` remote file handle)."""

        return hasattr(item, "read") and hasattr(item, "seek")

    # ------------------------------------------------------------------------------------------------------------------
    def _shot_uri(self, shot_id: Any, local: bool) -> str:
        """
        Build the data source handle (local path or remote URI) for a given shot, per `self.data_format`.

        Parameters
        ----------
        shot_id : Any
            Target shot ID, already validated via `self._check_shot_id()`.
        local : bool
            If True, a local path under `self.base_local_data_path` is built, otherwise a remote URI under
            `self.s3_mast_dataset_path` is built.

        Returns
        -------
        str
            Local file path (e.g. "/mast/tokamark/v1/30471.zarr"), or remote URI (e.g.
            "s3://mast/tokamark/v1/30471.zarr").

        """

        file_extension = FORMAT_FILE_EXTENSIONS[self.data_format]
        file_name = f"{shot_id}{file_extension}"

        if local:
            self._validate_base_local_data_path()
            return os_join(str(self.base_local_data_path), file_name)

        remote_path = posix_join(self.s3_mast_dataset_path, file_name)

        return f"{self.target_fsspec_protocol}://{remote_path.lstrip('/')}"

    # ------------------------------------------------------------------------------------------------------------------
    def _open_kwargs(self, handle: DataSourceHandleType) -> dict[str, Any]:
        """
        Build the keyword arguments used to open a data source handle via `xarray`.

        Parameters
        ----------
        handle : DataSourceHandleType
            Data source handle (local path, remote URI, or file-like object).

        Returns
        -------
        dict[str, Any]
            Keyword arguments for `xarray.open_datatree()`/`xarray.open_dataset()`, i.e., the engine matching
            `self.data_format` plus, for remote URIs only, the S3 storage options.

        """

        open_kwargs: dict[str, Any] = {"engine": self.engine, "create_default_indexes": False}
        if is_remote_uri(handle):
            open_kwargs["storage_options"] = {"anon": True, "endpoint_url": self.s3_endpoint_url}

        return open_kwargs

    # ------------------------------------------------------------------------------------------------------------------
    def _parse_shot_info_dict(
        self,
        shot_info: ShotInfoType,
    ) -> dict[str, Any]:
        """
        Parse dictionary with shot information.

        Parameters
        ----------
        shot_info : ShotInfoType
            Dictionary with shot information required for store creation, with valid keys and types as defined in
            `MAST_tools.data_models.ShotInfo`. Default value for the non-required key "local" is
            `MAST_tools.utils.store_utils.DEFAULT_LOCAL_FLAG_VALUE`.

        Returns
        -------
        dict[str, Any]
            Dictionary with parsed items.

        Raises
        ------
        KeyError
            If field "shot_id" is missing in parameter `shot_info`.
        TypeError
            If field "local" in `shot_info` is not boolean.

        """

        parsed_shot_info = {}

        # Validate "shot_id" field
        if "shot_id" not in shot_info:
            raise KeyError("Missing field `shot_id`.")
        else:
            parsed_shot_info["shot_id"] = shot_info.get("shot_id")
            self._check_shot_id(shot_id=parsed_shot_info["shot_id"])

        # Validate "local" field
        parsed_shot_info["local"] = shot_info.get("local", DEFAULT_LOCAL_FLAG_VALUE)
        if not isinstance(parsed_shot_info["local"], bool):
            raise TypeError("Invalid field `local`: it must be of type bool.")

        return parsed_shot_info

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def check_data_origin(
        data_origin: BaseDataSourceType,
    ) -> None:
        """
        Check MAST data origin.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Object expected to define data origin for tree/handle creation: a Mapping (shot info), a data source
            handle (local path str, `os.PathLike`, remote URI, or file-like object), or an already opened
            `xarray.DataTree`.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If parameter `data_origin` is not a dict, a str/`os.PathLike` path or URI, an `xarray.DataTree`, or a
            file-like object.

        """

        is_valid = isinstance(data_origin, (dict, str, os.PathLike, xr.DataTree)) or MASTStorageManager._is_file_like(
            data_origin
        )
        if not is_valid:
            raise TypeError(
                "Invalid parameter `data_origin`: it must be of type dict, str, os.PathLike, xarray.DataTree, or "
                "file-like."
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _check_shot_id(self, shot_id: Any) -> None:
        """
        Check if provided shot ID has a valid type/value. Allowed types are int, or str with value being a digit.

        Parameters
        ----------
        shot_id : Any
            Target shot ID.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If parameter `shot_id` not of type int or str.
        ValueError
            If parameter `shot_id` is of type str and its value is not a digit.

        """

        if not self._is_digit(item=shot_id):
            raise ValueError(
                "Invalid field/parameter 'shot_id': it must be of type int, or of type str representing a digit."
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _check_list_of_shot_ids(
        self,
        shot_ids: Union[list[int], tuple[int]],
    ) -> None:
        """
        Check list of shot IDs.

        Parameters
        ----------
        shot_ids : Union[list[int], tuple[int]]
            List of shot IDs.

        Returns
        -------
        None

        """

        for id_ in shot_ids:
            self._check_shot_id(shot_id=id_)

    # ------------------------------------------------------------------------------------------------------------------
    def _create_fs_remote(self, library: str, warn: bool = False) -> Any:
        """
        Create a filesystem instance either using 'fsspec' or 's3fs' libraries for remote data request.

        Parameters
        ----------
        library : str
            The target library for filesystem instance creation.
        warn : bool
            If True, warn about issues during filesystem instance creation.
            Default: False.

        Returns
        -------
        Any
            A filesystem instance, created via either `fsspec.filesystem` or `s3fs.S3FileSystem`.

        Raises
        ------
        NotImplementedError
            If a library other than 'fsspec' or 's3fs' is provided.

        """

        if library == "fsspec":
            if warn:
                warnings.warn("WARNING: Library `fsspec` cannot create asynchronous instances of file systems.")

            return fsspec.filesystem(
                protocol=self.base_fsspec_protocol,
                target_protocol=self.target_fsspec_protocol,
                target_options={"anon": True, "endpoint_url": self.s3_endpoint_url},
                # cache_storage='.cache',  # Uncomment and define cache folder, if required.
                # asynchronous=True  # REMARK: This is allowed, but it does not make the instance asynchronous.
            )

        elif library == "s3fs":
            return s3fs.S3FileSystem(anon=True, endpoint_url=self.s3_endpoint_url, asynchronous=True)
        else:
            raise NotImplementedError(f"Error: Library {library} not supported.")

    # ------------------------------------------------------------------------------------------------------------------
    def _read_fsspec_listdir(self, path: str, local: bool = False) -> list:
        """
        Evaluate the list dir method of 'fsspec' filesystem instance on the provided path.

        Parameters
        ----------
        path : str
            Target path to be evaluated via the invoked list dir method.
        local : bool
            Boolean flag to define the 'fsspec' instance to be used. If True, it corresponds to `self.fs_local_fsspec`,
            i.e., a `fsspec.implementations.local.LocalFileSystem` instance; otherwise, `self.fs_remote_fsspec` is used,
            which corresponds to a 'fsspec' instance created either via `fsspec.filesystem` or via `s3fs.S3FileSystem`.
            Default: False.

        Returns
        -------
        list
            List of items in filesystem instance.

        Raises
        ------
        FileNotFoundError
            If invalid path is provided.

        """
        try:
            if local:
                return self.fs_local_fsspec.ls(path)
            else:
                return self.fs_remote_fsspec.ls(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No data available for path {path}.")

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_shots(
        self,
        local: bool = False,
    ) -> list[int]:
        """
        Get a list of available MAST shot IDs.

        Parameters
        ----------
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.

        Returns
        -------
        list[int]
            List of all available shots IDs in the dataset.

        """

        # FSSpec pipeline
        if local:
            self._validate_base_local_data_path()
            all_filenames = self._read_fsspec_listdir(path=str(self.base_local_data_path), local=True)
        else:
            all_filenames = [
                item["Key"] for item in self._read_fsspec_listdir(path=self.s3_mast_dataset_path, local=False)
            ]

        file_extension = FORMAT_FILE_EXTENSIONS[self.data_format]
        raw_shot_ids = [
            filename.split("/")[-1].split(file_extension)[0]
            for filename in all_filenames
            if filename.endswith(file_extension)
        ]

        shot_ids = [int(raw_shot_id) for raw_shot_id in raw_shot_ids if self._is_digit(raw_shot_id)]
        shot_ids.sort()

        return shot_ids

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def list_shots_by_signal_availability(
        required_signals: Mapping[str, list[str]], availability_data_file_path: str = DEFAULT_SIGNAL_AVAILABILITY_FILE
    ) -> list:
        """
        List shot IDs following composite condition for signal availability and given availability file.

        Parameters
        ----------
        required_signals: Mapping[str, list[str]]
            Dictionary with required signal availability.
            Example: {"thomson_scattering": ["n_e"], "summary": ["power_nbi", "ip"]}
        availability_data_file_path : str
            Path to suitable CSV file with signal availability.
            Optional. Default: MAST_tools.utils.path_utils.DEFAULT_SIGNAL_AVAILABILITY_FILE.

        Returns
        -------
        list
            List of shot IDs.

        Raises
        ------
        ValueError
            If empty `required_signals` is provided.
        FileNotFoundError
            If invalid `availability_data_file_path` is provided.

        """

        if len(required_signals) == 0:
            raise ValueError("Empty `required_signals` was provided.")

        try:
            availability_data = pd.read_csv(availability_data_file_path)  # noqa - Ignore missing parameter(s) warning
        except FileNotFoundError:
            raise FileNotFoundError(f"Invalid `availability_data_file_path` '{availability_data_file_path}'.")

        source_signals_to_have = []
        for kk, vv in required_signals.items():
            source_signals_to_have += [f"{kk}-{val}" for val in vv]

        composite_condition = availability_data[source_signals_to_have[0]] == True  # noqa - "is" comparison misbehaves
        for signal in source_signals_to_have[1:]:
            composite_condition &= availability_data[signal] == True  # noqa - "is" comparison misbehaves

        return list(availability_data.loc[composite_condition]["shot_id"])  # noqa - Ignore missing attribute warning

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_sources(self, shot_ids: list[int] | None = None, local: bool = False) -> dict[int, list]:
        """
        Return a dictionary with all available sources per shot ID.

        Parameters
        ----------
        shot_ids : list[int] | None
            Target shot IDs to be checked in the MAST database. If None is provided, all available shots are checked.
            Optional. Default: None.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.

        Returns
        -------
        dict[int, list]
            Dictionary with available sources per shot ID.

        """

        if shot_ids is not None:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        # FSSpec pipeline

        if shot_ids is None:
            shot_ids = self.list_all_shots(local=local)

        source_info = {}
        for id_ in shot_ids:
            with self.open_shot_group(data_origin=ShotInfo(shot_id=id_, local=local)) as shot_tree:
                source_info[id_] = list(shot_tree.children.keys())

        return source_info

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_signals(
        self, shot_ids: list[int] | None = None, local: bool = False, verbose: bool = False
    ) -> dict[int, list[str]]:
        """
        Return a dictionary with all available signals per shot ID.

        Parameters
        ----------
        shot_ids : list[int] | None
            Target shot IDs to be checked in the MAST database. If None is provided, all shots are checked.
            Optional. Default: None.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.
        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        dict[int, list]
            Dictionary with available signals per shot ID.

        """

        if shot_ids is None:
            shot_ids = self.list_all_shots(local=local)
        else:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        # FSSpec pipeline
        signal_info = {}
        for shot_id in shot_ids:
            with self.open_shot_group(data_origin=ShotInfo(shot_id=shot_id, local=local), verbose=verbose) as shot_tree:
                signal_info[shot_id] = self.get_all_signals_in_group(group=shot_tree)

        return signal_info

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_store(self, shot_info: ShotInfoType, verbose: bool = False) -> DataSourceHandleType:
        """
        Make a data source handle for a given target shot, per `self.data_format`.

        The handle is a local file path (e.g. "/mast/tokamark/v1/30471.zarr") or a remote URI (e.g.
        "s3://mast/tokamark/v1/30471.zarr"), for both supported data formats. It is resolved into an
        `xarray.DataTree` by `self.make_shot_group()`, using the engine matching `self.data_format`.

        Parameters
        ----------
        shot_info : ShotInfoType
            Dictionary with shot information required for handle creation, with valid keys and types as defined in
            `MAST_tools.data_models.ShotInfo`. Keys and values are validated via `self._parse_shot_info_dict()`,
            where default values for non-required keys are also set.

        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        DataSourceHandleType
            Local file path, or remote URI, for the target shot.

        """

        parsed_shot_info = self._parse_shot_info_dict(shot_info=shot_info)
        store = self._shot_uri(shot_id=parsed_shot_info["shot_id"], local=parsed_shot_info["local"])

        if verbose:
            location = "local" if parsed_shot_info["local"] else "remote (S3)"
            print(f"{self.data_format} handle ({location}) for shot {parsed_shot_info['shot_id']}: {store}")

        return store

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_group(self, data_origin: BaseDataSourceType, verbose: bool = False) -> xr.DataTree:
        """
        Make an `xarray.DataTree` for a shot from data origin (either a data source handle or shot info).

        The same code path serves both supported data formats: only the `xarray` engine (and, for remote URIs, the
        storage options) differ, as resolved by `self._open_kwargs()`.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for tree creation. It can be a Mapping (dictionary) with shot information (as in the class
            method `self.make_shot_store()`), a data source handle (DataSourceHandleType instance), or an already
            opened `xarray.DataTree` (in which case it is returned unchanged, making this method idempotent).
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        xarray.DataTree
            DataTree for the target shot, with one child node per source group.

        """

        self.check_data_origin(data_origin=data_origin)

        if isinstance(data_origin, xr.DataTree):
            # Already an opened tree: nothing to do.
            return data_origin

        handle: DataSourceHandleType
        if isinstance(data_origin, Mapping):
            # Create handle from shot info mapping. REMARK: `xarray.DataTree` is itself a Mapping, hence this check
            # must stay after the early return above.
            handle = self.make_shot_store(shot_info=data_origin, verbose=verbose)
        else:
            # Use the given data source handle as-is
            handle = data_origin

        shot_tree = xr.open_datatree(handle, **self._open_kwargs(handle=handle))

        if verbose:
            print(f"DataTree created for `{handle}` with groups: {shot_tree.groups}")

        return shot_tree

    # ------------------------------------------------------------------------------------------------------------------
    @contextmanager
    def open_shot_group(self, data_origin: BaseDataSourceType, verbose: bool = False) -> Iterator[xr.DataTree]:
        """
        Context manager wrapping `self.make_shot_group()`, closing the tree on exit only if it was opened here.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for tree creation, as in `self.make_shot_group()`. If an already opened `xarray.DataTree`
            is provided, it is yielded unchanged and left open, since its lifetime belongs to the caller.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Yields
        ------
        xarray.DataTree
            DataTree for the target shot.

        """

        shot_tree = self.make_shot_group(data_origin=data_origin, verbose=verbose)
        try:
            yield shot_tree
        finally:
            if shot_tree is not data_origin:
                shot_tree.close()

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_all_signals_in_group(group: xr.DataTree) -> list[str]:
        """
        Get list of all signals in a given group.

        Parameters
        ----------
        group : xarray.DataTree
            DataTree to be inspected for signals.

        Returns
        -------
        list[str]
            List of signals in group using the format [<source>-<signal>].

        """

        found_signals = []
        for group_path in group.groups:
            if group_path == "/":
                continue
            group_name = group_path.strip("/").split("/")[-1]
            node_dataset = group[group_path].dataset
            found_signals.extend(f"{group_name}-{member}" for member in node_dataset.data_vars)

        return found_signals

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_signals_in_store(self, store: DataSourceHandleType) -> list[str]:
        """
        Get list of all signals in a given group.

        Parameters
        ----------
        store : DataSourceHandleType
            Data source handle to be inspected for signals.

        Returns
        -------
        list[str]
            List of signals in store using the format [<source>-<signal>].

        """

        with self.open_shot_group(data_origin=store) as shot_tree:
            found_signals = self.get_all_signals_in_group(group=shot_tree)

        return found_signals

    # ------------------------------------------------------------------------------------------------------------------
    def are_signals_in_group(self, group: xr.DataTree, signals: list[str]) -> dict[str, bool]:
        """
        Evaluate if a given signal is in a target group.

        group : xarray.DataTree
            Target group.
        signals: list[str]
            List of signals to be searched within the target group. It expects the format [<source>-<signal>].

        Returns
        -------
        dict[str, bool]
            Dictionary with check results per signal.

        """

        all_signals_in_group = self.get_all_signals_in_group(group=group)
        check_results = {signal_: (signal_ in all_signals_in_group) for signal_ in signals}

        return check_results

    # ------------------------------------------------------------------------------------------------------------------
    def are_signals_in_store(self, store: DataSourceHandleType, signals: list[str]) -> dict[str, bool]:
        """
        Evaluate if a given signal is in a target store.

        store : DataSourceHandleType
            Target store.
        signals: list[str]
            List of signals to be searched within the target store. It expects the format [<source>-<signal>].

        Returns
        -------
        dict[str, bool]
            Dictionary with check results per signal.

        """

        with self.open_shot_group(data_origin=store) as shot_tree:
            return self.are_signals_in_group(group=shot_tree, signals=signals)

    # ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def tests() -> None:
    """
    Quick tests for module functionality.

    Returns
    -------
    None

    """

    t0_all_tests = time.time()

    # ..................................................................................................................

    # Base settings
    shot_ids = [30471]  # None for entire dataset (which takes very long time if local=False).
    shot_id = 30471
    local = True
    verbose = False

    shot_info = ShotInfo(shot_id=shot_id, local=local)

    # Creation of MASTStorageManager instance for test
    store_manager = MASTStorageManager(
        base_fsspec_protocol="simplecache",
        target_fsspec_protocol="s3",
        s3_endpoint_url="https://s3.echo.stfc.ac.uk",
        s3_mast_dataset_path="/mast/tokamark/v1",
        base_local_data_path="/mast/tokamark/v1",
    )

    TESTS_TO_RUN = {  # noqa - Ignore lowercase warning
        "get_all_shot_ids": False,
        "get_all_sources": False,
        "get_all_signals": False,
        "make_group_from_store": False,
        "make_group_from_shot_info": False,
        "check_signal_in_store": True,
        "get_all_signals_in_store": True,
        "check_signal_availability": False,
        "storage_fixture": False,  # -> See `tests/test_storage.py` for fixture generation.
    }

    # ..................................................................................................................
    # List all shot IDs for the entire dataset, using different pipelines

    if TESTS_TO_RUN["get_all_shot_ids"]:
        pipeline_tag = "ffspec"
        local_tag = "local" if local else "remote"  # NOSONAR - Ignore weak warning
        print(f"Getting available shot IDs ({pipeline_tag} pipeline, {local_tag} bucket)...\n")
        t0 = time.time()
        all_shots_ids = store_manager.list_all_shots(local=local)

        print(f"Number of shots: {len(list(all_shots_ids))}")
        print(f"Shot IDs: {list(all_shots_ids)}")
        print(f"Elapsed time: {round(time.time() - t0, 2)} s\n")

    # ..................................................................................................................
    # List all sources

    if TESTS_TO_RUN["get_all_sources"]:
        all_sources = store_manager.get_all_sources(
            shot_ids=shot_ids,
            local=local,
        )
        print(f"all_sources[shot_ids[0]]: {all_sources[shot_ids[0]]}\n")

    # ..................................................................................................................
    # List all signals

    if TESTS_TO_RUN["get_all_signals"]:
        all_signals = store_manager.get_all_signals(shot_ids=shot_ids, local=local, verbose=verbose)
        print(f"all_signals[shot_ids[0]]: {all_signals[shot_ids[0]]}\n")

    # ..................................................................................................................
    # Make group for a given shot via existing store object

    if TESTS_TO_RUN["make_group_from_store"]:
        store_ = store_manager.make_shot_store(shot_info=shot_info)

        with store_manager.open_shot_group(data_origin=store_) as group_from_store:
            print(f"group_from_store.groups (group from store): {group_from_store.groups}\n")

    # ..................................................................................................................
    # Make group for a given shot directly from shot_info

    if TESTS_TO_RUN["make_group_from_shot_info"]:
        with store_manager.open_shot_group(data_origin=shot_info) as group_from_shot_id:
            print(f"group_from_shot_id.groups (group from shot ID): {group_from_shot_id.groups}\n")

    # ..................................................................................................................
    # Get all the signals available for a given store

    if TESTS_TO_RUN["get_all_signals_in_store"]:
        store_ = store_manager.make_shot_store(shot_info=shot_info)

        all_signals = store_manager.get_all_signals_in_store(store=store_)
        print("All signals in store:")
        pprint(all_signals)

    # ..................................................................................................................
    # Check if signal is in store

    if TESTS_TO_RUN["check_signal_in_store"]:
        store_ = store_manager.make_shot_store(shot_info=shot_info)

        signals_ = ["abc", "summary-power_radiated"]

        check_results = store_manager.are_signals_in_store(store=store_, signals=signals_)
        print("\nCheck results for signals in store:")
        pprint(check_results)

    # ..................................................................................................................
    # List all shots IDs for given signal availability

    if TESTS_TO_RUN["check_signal_availability"]:
        dict_target_signals = {
            "thomson_scattering": ["n_e"],
            "spectrometer_visible": ["filter_spectrometer_bes_voltage"],
            "summary": ["power_nbi", "ip"],
        }
        signal_availability_file = DEFAULT_SIGNAL_AVAILABILITY_FILE

        print("\nSignals to be checked for simultaneous availability across all shots:")
        pprint(dict_target_signals)

        filtered_ids = store_manager.list_shots_by_signal_availability(
            required_signals=dict_target_signals, availability_data_file_path=signal_availability_file
        )

        print(f"\nfiltered_ids ({len(list(filtered_ids))} shots):")
        pprint(filtered_ids)

    # ..................................................................................................................
    # Run against the synthetic fixtures generated by `tests/test_storage.py`, for both data formats.

    if TESTS_TO_RUN["storage_fixture"]:
        import tempfile
        from pathlib import Path

        from tests.test_storage import FIXTURE_SHOT_ID, _write_fixture

        with tempfile.TemporaryDirectory() as fixture_dir:
            for data_format_ in SUPPORTED_DATA_FORMATS:
                _write_fixture(Path(fixture_dir), data_format=data_format_)
                fixture_store_manager = MASTStorageManager(data_format=data_format_, base_local_data_path=fixture_dir)
                fixture_shot_info = ShotInfo(shot_id=FIXTURE_SHOT_ID, local=True)

                fixture_store_ = fixture_store_manager.make_shot_store(shot_info=fixture_shot_info, verbose=True)
                with fixture_store_manager.open_shot_group(data_origin=fixture_store_) as fixture_group:
                    print(f"\n[{data_format_}] fixture_group.groups: {fixture_group.groups}")

                fixture_all_signals = fixture_store_manager.get_all_signals_in_store(store=fixture_store_)
                print(f"[{data_format_}] fixture_all_signals: {fixture_all_signals}\n")

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
