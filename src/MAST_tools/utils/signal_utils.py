"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import numpy as np
from typing import Union, Optional, Any

from MAST_tools.utils.data_utils import (
    ShotInfo,
    StoreManagerParameters,
    BaseDataSourceType,
    ExtendedDataSourceType,
    XarrayDatasetType,
    StoreManagerParametersType,
)
from MAST_tools.utils.store_utils import MASTStorageManager
from MAST_tools.utils.general_utils import get_random_string


# ======================================================================================================================
class MASTSignalManager:
    """
    Class with signal management tools for MAST data.

    Attributes
    ----------
    signal_manager_id : str
        User-defined signal manager ID.
    store_manager_settings : StoreManagerParametersType | None
        Settings for the store manager instance.
    store_manager : MASTStorageManager
        Instance of the MASTStorageManager class.

    Methods
    -------
    _set_store_manager(store_manager)
        Set the store_manager instance attribute.
    _get_source_dataset(data_origin, source_name, verbose)
        Get a source group as an xarray Dataset from a given data origin.
    get_source_profiles(data_origin, source_name)
        Get source profiles from a given data origin.
    get_signal_values(signal_name, data_origin, source_name, verbose)
        Get signal values from a given data origin.
    get_signal_times_and_time_type(signal_name, data_origin, source_name, verbose)
        Get signal times and time type from a given data origin.
    get_signal_profile(signal_name, data_origin, source_name, verbose)
        Get signal profile from a given data origin.
    get_channel_names(signal_name, data_origin, source_name, verbose)
        Get channel names from a given data origin.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, store_manager_settings: StoreManagerParametersType | None = None) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        store_manager_settings : StoreManagerParametersType | None
            Settings for the store manager instance provided as a kwargs dictionary, with keywords and required value
            types as defined in `MAST_tools.utils.data_utils.StoreManagerParameters`. Only valid (keyword, value) pairs
            are used to update default values, e.g. {"target_fsspec_protocol": "s3"}.
            Optional. Default: None, which results in the default values for all the keywords as defined in
            `MAST_tools.store_utils.MASTStorageManager.__init__`.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Notes
        -----
        - Upon creation of a signal manager instance from the `MASTSignalManager` class, a store manager from the
          `MASTStorageManager` class is created as instance attribute. This facilitates the process to get signal values
          as no separate store manager instance must be created. However, if an existing store manager instance is
          available, it could be passed.

        """

        self.signal_manager_id = f"store_manager_{get_random_string(4)}"
        self.store_manager_settings = store_manager_settings or StoreManagerParameters()
        self.store_manager = MASTStorageManager(**self.store_manager_settings)

    # ------------------------------------------------------------------------------------------------------------------
    def _set_store_manager(self, store_manager: MASTStorageManager) -> None:
        """
        Set the `store_manager` attribute.

        Parameters
        ----------
        store_manager : MASTStorageManager
            Instance of the MASTStorageManager class.

        Returns
        -------
        None

        """

        self.store_manager = store_manager

    # ------------------------------------------------------------------------------------------------------------------
    def _get_source_dataset(
        self,
        data_origin: Any,
        source_name: str | None,
        verbose: bool = False,
    ) -> XarrayDatasetType | None:
        """
        Get a source group as an xarray Dataset from a given data origin.

        The data origin is resolved to an `xarray.DataTree` via `MASTStorageManager.make_shot_group()` (which is
        backend-agnostic and idempotent for an already opened tree), and the source group is then read from it.

        Parameters
        ----------
        data_origin : Any
            Origin of data: shot info Mapping, data source handle (local path/remote URI/file-like object), or an
            already opened `xarray.DataTree`.
        source_name : str | None
            Name of target source (group).
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        XarrayDatasetType | None
            Source dataset, or None if the group could not be found.

        """

        if source_name is None:
            if verbose:
                print("No `source_name` was provided, so no source group can be resolved.")
            return None

        profile = None
        try:
            shot_tree = self.store_manager.make_shot_group(data_origin=data_origin, verbose=verbose)
            profile = shot_tree[source_name].dataset  # noqa - Ignore expected type warning
        except (KeyError, OSError, AttributeError) as e:
            if verbose:
                print(f"Exception: {e}")

        return profile

    # ------------------------------------------------------------------------------------------------------------------
    def get_source_profiles(
        self, data_origin: BaseDataSourceType, source_name: str, verbose: bool = False
    ) -> XarrayDatasetType | None:
        """
        Get source profiles from a given data origin.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for source profile creation: shot info Mapping, data source handle, or an already opened
            `xarray.DataTree`.
        source_name : str
            Name of target source.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        XarrayDatasetType | None
            Source profiles from given data origin.

        """

        self.store_manager.check_data_origin(data_origin)

        return self._get_source_dataset(data_origin=data_origin, source_name=source_name, verbose=verbose)

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_values(
        self,
        signal_name: str,
        data_origin: ExtendedDataSourceType,
        source_name: str | None = None,
        verbose: bool = False,
    ) -> Union[np.ndarray, None]:
        """
        Get signal values from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : ExtendedDataSourceType
            Origin of data for signal value retrieval: Mapping, data source handle, XarrayDataTreeType, or
            XarrayDatasetType.
        source_name : str | None
            Name of target source. It must be provided unless `data_origin` is already a source dataset.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[numpy.ndarray, None]
            Signal values, or None if error.

        """

        signal_profile = self.get_signal_profile(
            signal_name=signal_name, data_origin=data_origin, source_name=source_name, verbose=verbose
        )

        if signal_profile is not None:
            return signal_profile.values  # noqa - Ignore missing attribute warning
        else:
            # If here, an error occurred while creating the signal profile.
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_times_and_time_type(
        self,
        signal_name: str,
        data_origin: ExtendedDataSourceType,
        source_name: str | None = None,
        verbose: bool = False,
    ) -> Union[tuple[np.ndarray, str], tuple[None, None]]:
        """
        Get signal times and time type from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : ExtendedDataSourceType
            Origin of data for signal value retrieval: Mapping, data source handle, XarrayDataTreeType, or
            XarrayDatasetType.
        source_name : str | None
            Name of target source. It must be provided unless `data_origin` is already a source dataset.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[list[np.ndarray, str], list[None, None]]
            List with signal times with time type (i.e., list[np.ndarray, str]), or [None, None] if error.

        """

        signal_profile = self.get_signal_profile(
            signal_name=signal_name, data_origin=data_origin, source_name=source_name, verbose=verbose
        )

        if signal_profile is not None:
            try:
                coords_keys = signal_profile.coords.keys()  # noqa - Ignore missing attribute warning
                time_type_ = [str(kk) for kk in coords_keys if str(kk).startswith("time")][0]
            except IndexError:
                # If here, no time info was found, and so signal is not time-dependent.
                return None, None

            signal_times_ = signal_profile[time_type_].values  # noqa - Ignore missing attribute warning

            return signal_times_, time_type_
        else:
            # If here, an error occurred while creating the signal profile.
            return None, None

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_profile(  # NOSONAR - Ignore cognitive complexity
        self,
        signal_name: str,
        data_origin: ExtendedDataSourceType,
        source_name: str | None = None,
        verbose: bool = False,
    ) -> Any | None:
        """
        Get signal profile from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : ExtendedDataSourceType
            Origin of data for signal value retrieval: Mapping, data source handle, XarrayDataTreeType, or
            XarrayDatasetType.
        source_name : str | None
            Name of target source. It must be provided unless `data_origin` is already a source dataset.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Any | None
            Signal profile, or None if error.

        """

        if isinstance(data_origin, XarrayDatasetType):
            # From source profile (i.e., xarray.core.dataset)
            profile = data_origin

        else:
            # From shot info, a data source handle, or an already opened xarray.DataTree

            try:
                self.store_manager.check_data_origin(data_origin=data_origin)  # noqa - Ignore expected type warning
            except Exception as e:
                if verbose:
                    print(f"Exception: {e}")

            profile = self._get_source_dataset(data_origin=data_origin, source_name=source_name, verbose=verbose)

        if profile is not None:
            try:
                return profile[signal_name]
            except KeyError:
                if verbose:
                    print(f"Invalid `signal_name` {signal_name}.")
                return None
        else:
            # If here, an error occurred while creating the signal profile.
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def get_channel_names(
        self,
        signal_name: str,
        data_origin: ExtendedDataSourceType,
        source_name: str | None = None,
        verbose: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Get signal channel names.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : ExtendedDataSourceType
            Origin of data for signal profile retrieval: Mapping, data source handle, XarrayDataTreeType, or
            XarrayDatasetType.
        source_name : str | None
            Name of target source. It must be provided unless `data_origin` is already a source dataset.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Optional[np.ndarray]
            Available signal channels as np.ndarray, or None.

        """

        try:
            signal_profile = self.get_signal_profile(
                signal_name=signal_name, data_origin=data_origin, source_name=source_name, verbose=verbose
            )

            non_time_coords = [coord for coord in signal_profile.coords if coord != "time"]  # noqa - Ignore missing att
            if non_time_coords:
                channel_coord = non_time_coords[0]
                return signal_profile.coords[channel_coord].values  # noqa - Ignore missing atttribute warning
            else:
                return None
        except Exception as e:
            if verbose:
                print(f"Error getting signal profile: {e}")
            return None


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

    shot_info = ShotInfo(shot_id=30421, local=False)
    source_name = "magnetics"
    signal_name = "flux_loop_flux"

    TESTS_TO_RUN = {  # noqa - Ignore lowercase warning
        "source_from_store": False,
        "signal_values_from_store": False,
        "signal_values_from_shot_info": False,
        "signal_times_from_shot_info": True,
        "storage_fixture": False,  # -> See `tests/test_storage.py` for fixture generation.
    }

    base_local_data_path = "/mast/tokamark/v1"  # -> REMARK: Use correct local folder for local tests.
    signal_manager = MASTSignalManager(
        store_manager_settings=StoreManagerParameters(base_local_data_path=base_local_data_path)
    )

    # print(type(signal_manager.store_manager))
    # print(signal_manager.store_manager.base_fsspec_protocol)
    # print(signal_manager.store_manager.target_fsspec_protocol)

    # ..................................................................................................................
    # Get signal values from store
    if TESTS_TO_RUN["source_from_store"]:
        source_from_shot_info = signal_manager.get_source_profiles(data_origin=shot_info, source_name=source_name)

        print(source_from_shot_info)
        print(type(source_from_shot_info))

        if source_from_shot_info is not None:
            print(source_from_shot_info[signal_name])
            print(type(source_from_shot_info[signal_name]))

    # ..................................................................................................................
    # Get signal values from store
    if TESTS_TO_RUN["signal_values_from_store"]:
        store_from_shot_info = signal_manager.store_manager.make_shot_store(shot_info=shot_info)
        signal_values = signal_manager.get_signal_values(
            signal_name=signal_name, data_origin=store_from_shot_info, source_name=source_name, verbose=True
        )

        print(f"Signal values: {signal_values}\n")
        print(f"Signal shape: {signal_values.shape}\n")  # noqa - Ignore missing attribute warning

    # ..................................................................................................................
    # Get signal values from shot info

    if TESTS_TO_RUN["signal_values_from_shot_info"]:
        signal_values = signal_manager.get_signal_values(
            signal_name=signal_name, data_origin=shot_info, source_name=source_name
        )

        print(f"Signal values: {signal_values}\n")

    # ..................................................................................................................
    # Get signal times from shot info

    if TESTS_TO_RUN["signal_times_from_shot_info"]:
        signal_times, signal_type = signal_manager.get_signal_times_and_time_type(
            signal_name=signal_name, data_origin=shot_info, source_name=source_name
        )

        print(f"Signal type: '{signal_type}'\n")
        print(f"Signal times: {signal_times}\n")

    # ..................................................................................................................
    # Run against the synthetic fixtures generated by `tests/test_storage.py`, for both data formats.

    if TESTS_TO_RUN["storage_fixture"]:
        import tempfile
        from pathlib import Path

        from tests.test_storage import FIXTURE_SHOT_ID, _write_fixture

        with tempfile.TemporaryDirectory() as fixture_dir:
            for data_format_ in ("netcdf", "zarr"):
                _write_fixture(Path(fixture_dir), data_format=data_format_)
                fixture_signal_manager = MASTSignalManager(
                    store_manager_settings=StoreManagerParameters(
                        data_format=data_format_, base_local_data_path=fixture_dir
                    )
                )
                fixture_shot_info = ShotInfo(shot_id=FIXTURE_SHOT_ID, local=True)

                fixture_signal_values = fixture_signal_manager.get_signal_values(
                    signal_name=signal_name, data_origin=fixture_shot_info, source_name=source_name, verbose=True
                )
                print(f"[{data_format_}] signal values: {fixture_signal_values}\n")

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
