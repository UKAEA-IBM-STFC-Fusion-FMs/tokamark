"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
from collections.abc import Mapping
from typing import Union, Any, TypedDict, Optional, Literal
from typing_extensions import NotRequired
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree


# ======================================================================================================================
# Data format types

DataFormatType = Literal["zarr", "netcdf"]
"""Supported on-disk/S3 data formats, both loaded via xarray."""


# ======================================================================================================================
class StoreManagerParameters(TypedDict):
    """User-defined parameters for the creation of a store manager instance. If not set, default values are used."""

    base_fsspec_protocol: NotRequired[str]
    """Base protocol used by 'fsspec'."""

    target_fsspec_protocol: NotRequired[str]
    """Target filesystem protocol for the selected base 'fsspec' protocol."""

    s3_endpoint_url: NotRequired[str]
    """Endpoint of the cloud S3 bucket used for remote data pulling."""

    s3_mast_dataset_path: NotRequired[str]
    """Path for the target MAST dataset within the configured S3 bucket."""

    base_local_data_path: NotRequired[Optional[str]]
    """Local root path used for local data pulling."""

    base_local_zarr_path: NotRequired[Optional[str]]
    """Deprecated alias for `base_local_data_path`, kept for backwards compatibility."""

    data_format: NotRequired[DataFormatType]
    """On-disk/S3 data format to load. Either "zarr" (default) or "netcdf"."""


# ======================================================================================================================
class ShotInfo(TypedDict):
    """
    Information to pull shot data/metadata from the MAST dataset. The default value for the optional parameter `local`
    is assigned via `MAST_tools.utils.store_utils.MASTStorageManager._parse_shot_info_dict()`.
    """

    shot_id: int
    """ID of a target shot to be pulled from the MAST dataset."""

    local: NotRequired[bool]
    """Boolean flag to define data/metadata location. If True, the target shot is pulled from locally stored data 
    (e.g., in the CSD3 cluster), otherwise it is pulled from the registered remote data repository (e.g., a cloud S3
    bucket)."""


# ======================================================================================================================
# Data types

ShotInfoType = Mapping[str, Any]
StoreManagerParametersType = Mapping[str, Any]

XarrayDatasetType = Dataset
XarrayDataTreeType = DataTree

DataSourceHandleType = Union[str, os.PathLike, Any]
"""Backend-agnostic handle for a shot, as returned by `MASTStorageManager.make_shot_store()`: a local file path, an
`fsspec` URI (e.g. "s3://..."), or a file-like object. It is opened via `xarray` with the engine matching the
configured data format, so no format-specific store object is ever built. There is no single stable public type
covering "file-like" across `fsspec`/`s3fs`, hence `Any`."""

BaseDataSourceType = Union[ShotInfoType, DataSourceHandleType]
ExtendedDataSourceType = Union[BaseDataSourceType, XarrayDatasetType, XarrayDataTreeType]
