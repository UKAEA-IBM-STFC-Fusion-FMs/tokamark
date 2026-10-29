"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import yaml
import numpy as np
from pprint import pprint
from typing import Optional, Sequence, Any
from collections.abc import Callable, Mapping
from torch.utils.data import Dataset

from MAST_tools.utils import signal_utils
from MAST_tools.utils.path_utils import RANDOM_SPLIT_OUTLIER_METADATA_FILE

from MAST_tools.utils.data_utils import ShotInfo, StoreManagerParametersType


# ======================================================================================================================
class CachedDataset(Dataset):
    """
    Class to cache a base dataset into local memory.

    Attributes
    ----------
    base_dataset : Collection
        Base dataset to be cached.
    cache : list[Any]
        List of cached dataset items.
    _is_cached = list[bool]
        List of boolean flags to define if a given dataset item is cached or not.

    Methods
    -------
    __len__()
        Return the size of the dataset.
    __getitem__(idx)
        Return samples by shot index.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, base_dataset: Sequence) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        base_dataset : Sequence
            Base dataset to be cached.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        self.base_dataset = base_dataset
        self.cache = [None] * len(base_dataset)
        self._is_cached = [False] * len(base_dataset)

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """
        Return the size of the dataset.

        Returns
        -------
        int
            Length of self.base_dataset.

        """

        return len(self.base_dataset)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx) -> Any:
        """
        Return samples by shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        Any
            A cached dataset sample.

        """

        # If dataset item is not cached, load it once
        if not self._is_cached[idx]:
            self.cache[idx] = self.base_dataset[idx]
            self._is_cached[idx] = True

        return self.cache[idx]


# ======================================================================================================================
class MastDataset(Dataset):
    """
    Dataset class for MAST data.

    Attributes
    ----------
    local : bool
        Boolean flag to activate local mode. If True, use local MAST database, otherwise use remote S3 bucket.
    shots_list : list[int]
        List of shot IDs.
    source_signal_list : list[list[str] | tuple[str]]
        List of data names to load using the format [[<source>, <signal>], ..., [<source>, <signal>]].
    signal_level_transform_map : Optional[Mapping[str, Callable]]
        Map of transforms to apply at signal level.
    sig : signal_utils.MASTSignalManager
        Instance of `signal_utils.MASTSignalManager`.
    verbose : bool
        Boolean flag to activate/deactivate verbose mode.

    Methods
    -------
    __len__()
        Return the size of the dataset.
    __getitem__(idx)
        Return samples by shot index.
    get_shot_id(idx)
        Return shot ID from shot index.
    get_windows_for_shot(idx)
        Return the list of windows for the given shot index ([] if empty/missing).

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        local: bool,
        shots_list: list[int],
        source_signal_list: list[list[str] | tuple[str]],
        signal_level_transform_map: Optional[Mapping[str, Callable]] = None,
        remove_outliers: bool = False,
        outlier_metadata_file: str = RANDOM_SPLIT_OUTLIER_METADATA_FILE,
        remove_bad_efit_rating: bool = False,
        store_manager_settings: StoreManagerParametersType | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        local : bool
            If True, use local MAST database, otherwise use remote S3 bucket.
        shots_list : list[int]
            List of shot IDs to load data for.
        source_signal_list : list[list[str] | tuple[str]]
            List of data names to load using the format [[<source>, <signal>], ..., [<source>, <signal>]].
        signal_level_transform_map : Optional[Mapping[str, Callable]]
            Map of transforms to apply at signal level.
            Optional. Default: None.
        remove_outliers : bool
            If True, outliers are removed using information from the `outlier_metadata_file`.
            Optional. Default: False.
        outlier_metadata_file : str
            Source file for outlier metadata, only used when `remove_outliers` is True.
            Optional: Default: `MAST_tools.utils.path_utils.RANDOM_SPLIT_OUTLIER_METADATA_FILE`.
        store_manager_settings : StoreManagerParametersType | None
            Settings for the store manager instance provided as a kwargs dictionary, with keywords and required value
            types as defined in `MAST_tools.data_models.StoreManagerParametersType`. Only valid (keyword, value)
            pairs are used to update default values, e.g. {"target_fsspec_protocol": "s3"}. This includes
            "data_format" ("zarr" or "netcdf") to select the on-disk/S3 data format to load via xarray.
            Optional. Default: None, which results in the default values for all the keywords as defined in
            `MAST_tools.store_utils.MASTStorageManager.__init__`.
        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        # General MAST settings
        self.local = local

        # Shot-specific settings
        self.shots_list = shots_list
        self.source_signal_list = source_signal_list
        self.signal_level_transform_map = signal_level_transform_map
        self.remove_outliers = remove_outliers
        if self.remove_outliers:
            with open(outlier_metadata_file, "r") as f:
                self.dict_outlier_metadata = yaml.safe_load(f)

        self.remove_bad_efit_rating = remove_bad_efit_rating
        # Signal manager instance
        self.sig = signal_utils.MASTSignalManager(store_manager_settings=store_manager_settings)

        # Other settings
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns
        -------
        int
            Length of self.shot_list.

        """

        return len(self.shots_list)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(  # NOSONAR - Ignore cognitive complexity
        self, idx: int
    ) -> dict:
        """
        Return samples by shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        dict
            Dictionary containing a raw shot.

        """

        shot = {}

        # Removing outliers from signal to load
        if self.remove_outliers:
            all_outliers = set(self.dict_outlier_metadata.get(self.shots_list[idx], []))
            source_signal_to_load = []
            for source, signal in self.source_signal_list:
                if f"{source}-{signal}" in all_outliers:
                    if self.verbose:
                        print(f"Replacing outlier {source}-{signal} from shot {self.shots_list[idx]}")
                    shot[f"{source}-{signal}"] = {"time": np.array([]), "values": np.array([])}
                else:
                    source_signal_to_load.append([source, signal])
        else:
            source_signal_to_load = self.source_signal_list

        # Group signals per source
        source_signals = {}
        for source, signal in source_signal_to_load:
            if source in source_signals:
                source_signals[source].append(signal)
            else:
                source_signals[source] = [signal]

        # Open the shot once as an xarray.DataTree, then read each source group from it.
        with self.sig.store_manager.open_shot_group(
            data_origin=ShotInfo(shot_id=self.shots_list[idx], local=self.local), verbose=self.verbose
        ) as shot_tree:
            # First, iterate over each source to get source store.
            # Then, iterate over each signal in source group.
            for source in source_signals:
                source_store = self.sig.get_source_profiles(data_origin=shot_tree, source_name=source)

                load_signals = True
                mask_efit_rating = None
                if (source_store is not None) and (source == "equilibrium") and self.remove_bad_efit_rating:
                    if "ip_rating" in list(source_store.data_vars):
                        mask_efit_rating = (  # noqa - Ignore attribute warning
                            self.sig.get_signal_profile(
                                data_origin=source_store, signal_name="ip_rating", verbose=self.verbose
                            )
                            == 0
                        ).values

                    else:
                        if self.verbose:
                            print(f"ip_rating not available for shot {self.get_shot_id(idx)}.")
                        load_signals = False

                for signal in source_signals[source]:
                    shot_vals = None
                    shot_time = None

                    if (source_store is not None) and load_signals:
                        signal_profile = self.sig.get_signal_profile(
                            data_origin=source_store, signal_name=signal, verbose=self.verbose
                        )

                        if signal_profile is not None:
                            try:
                                shot_time, _ = self.sig.get_signal_times_and_time_type(
                                    signal_name=signal,
                                    data_origin=source_store,
                                    source_name=source,
                                    verbose=self.verbose,
                                )

                            except Exception as e:
                                print(f"Error while getting time for shot {self.shots_list[idx]}: {e}.")
                                shot_time = None

                            try:
                                signal_profile_vals = signal_profile.values  # noqa - Ignore missing attribute warning
                                shot_vals = (
                                    np.expand_dims(signal_profile_vals, axis=0)
                                    if signal_profile_vals.ndim == 1
                                    else signal_profile_vals
                                )

                                if mask_efit_rating is not None:
                                    # Expand mask to match shot_vals dimensions
                                    expand_shape = (1,) * (shot_vals.ndim - 1) + (mask_efit_rating.shape[0],)  # noqa
                                    mask_expanded = mask_efit_rating.reshape(expand_shape)  # noqa - Ignore missing att
                                    # Apply mask
                                    shot_vals = np.where(mask_expanded, np.nan, shot_vals)

                            except AttributeError:
                                shot_vals = None

                    # Apply variable-level transforms only if we have both time and values
                    if shot_vals is not None and shot_time is not None:
                        if self.signal_level_transform_map is not None:
                            shot[f"{source}-{signal}"] = self.signal_level_transform_map[f"{source}-{signal}"](
                                {"time": shot_time, "values": shot_vals}
                            )
                        else:
                            shot[f"{source}-{signal}"] = {"time": shot_time, "values": shot_vals}
                    else:
                        # Keep missing signals as {"time": np.array([]), "values": np.array([])}
                        shot[f"{source}-{signal}"] = {"time": np.array([]), "values": np.array([])}

        return shot

    # ------------------------------------------------------------------------------------------------------------------
    def get_shot_id(self, idx: int) -> int:
        """
        Return shot ID from shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        int
            Shot ID.

        """

        return self.shots_list[idx]

    # ------------------------------------------------------------------------------------------------------------------
    def get_windows_for_shot(self, idx: int) -> list:
        """
        Return the list of windows for the given shot index ([] if empty/missing).

        Parameters
        ----------
        idx : int
            Shot ID.

        Returns
        -------
        list
            The list of windows for the given shot index.

        """

        try:
            obj = self.__getitem__(idx)
        except IndexError:
            return []

        if isinstance(obj, list):
            return obj
        else:
            # If here, 'obj' is the entire shot → treat as a single-window shot.
            return [obj]


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

    dummy_dataset = MastDataset(
        local=True,
        shots_list=[30421],
        source_signal_list=[
            ["summary", "power_nbi"],
            ["magnetics", "b_field_pol_probe_obr_field"],
            ["magnetics", "b_field_pol_probe_omv_voltage"],
            ["magnetics", "b_field_tor_probe_omaha_channel"],
        ],
    )

    print(f"\ndummy_dataset.__len__: {dummy_dataset.__len__()}")
    print(f"\ndummy_dataset.get_shot_id(0): {dummy_dataset.get_shot_id(idx=0)}")

    print("\ndummy_dataset.__getitem__(0):")
    pprint(dummy_dataset.__getitem__(0))

    print("\ndummy_dataset.get_windows_for_shot(0):")
    pprint(dummy_dataset.get_windows_for_shot(idx=0))

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
