"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray as xarrayDataArray
from xarray.core.dataset import Dataset as xarrayDataset
import matplotlib.pyplot as plt
from typing import Union, Optional
from types import NoneType
import warnings

from MAST_tools.utils.data_utils import ShotInfo, BaseDataSourceType, StoreManagerParametersType
from MAST_tools.utils import signal_utils


# ======================================================================================================================
class MASTPlottingManager:
    """
    Class with plotting management tools for MAST data.

    Attributes
    ----------
    plot_manager_id : str
        User-defined plotting manager ID.

    Methods
    -------
    _plot_single_1d_profile(profile, ax)
        Helper function for plotting single 1D profile.
    _check_fig_size(fig_size)
        Helper function for checking `fig_size`.
    plot_1d_profiles(profiles, fig_size)
        Helper function for plotting 1D profiles, either from `xarrayDataArray` (single profile) or from `xarrayDataset`
        (group of profiles).
    plot_signal(data_origin, source_name, signal_name, fig_size)
        Helper function for plotting individual signals.
    plot_group(data_origin, source_name, fig_size)
        Helper function for plotting entire group of signals.
    plot_plasma_current(data_origin, fig_size)
        Helper function for plotting `summary__plasma_current` signal.
    plot_power_nbi(data_origin, fig_size)
        Helper function for plotting `summary__power_nbi` signal.
    plot_magnetics(data_origin, fig_size)
        Helper function for plotting `magnetics` group.
    plot_spectrometer(data_origin, fig_size)
        Helper function for plotting `spectrometer_visible` group.
    plot_charge_exchange(data_origin, fig_size)
        Helper function for plotting `charge_exchange` group.
    plot_thomson_scattering(data_origin, fig_size)
        Helper function for plotting `thomson_scattering` group.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, manager_id: str = "", store_manager_settings: StoreManagerParametersType | None = None) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        manager_id : str
            User defined manager ID. Default: "".
        store_manager_settings : StoreManagerParametersType | None
            Settings for the underlying store manager instance (e.g. `data_format`, `base_local_data_path`), provided
            as a kwargs dictionary with keywords/value types as defined in
            `MAST_tools.utils.data_utils.StoreManagerParameters`.
            Optional. Default: None, which results in default `MASTStorageManager` settings.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        self.plot_manager_id = manager_id
        self.sig = signal_utils.MASTSignalManager(store_manager_settings=store_manager_settings)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _plot_single_1d_profile(profile: xr.DataArray, ax: Optional[Union[np.ndarray, plt.Axes]] = None) -> None:
        """
        Helper function for plotting single 1D profiles.

        Parameters
        ----------
        profile : xr.DataArray
            Profile data.
        ax : Optional[Union[np.ndarray, plt.Axes]]
            Matplotlib axis.

        Returns
        -------
        None

        """

        try:
            profile.plot(ax=ax)
            if ax is None:
                ax = plt.gca()
            ax.grid(visible=True, alpha=0.5)  # noqa - Ignore missing attribute warning
        except Exception as e:
            print(f"Error: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_fig_size(fig_size: Optional[Union[list[int], set[int]]] = None) -> None:
        """
        Helper function for checking `fig_size`.

        Parameters
        ----------
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `fig_size` not a list/tuple.
        ValueError
            If `fig_size` not of length 2.

        """

        if not isinstance(fig_size, Union[list, tuple, NoneType]):
            raise TypeError("Invalid fig_size type: it must be list or tuple.")

        if (fig_size is not None) and (len(fig_size) != 2):
            raise ValueError("Invalid fig_size value: it must be a list/tuple of length 2.")

    # ------------------------------------------------------------------------------------------------------------------
    def plot_1d_profiles(
        self, profiles: Union[xarrayDataArray, xarrayDataset], fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting 1D profiles, either from `xarrayDataArray` (single profile) or from `xarrayDataset`
        (group of profiles).

        Parameters
        ----------
        profiles : Union[xarrayDataArray, xarrayDataset]
            Target 1D signal profiles.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        Raises
        ------
        None

        """

        warnings.warn(
            "WARNING: This method may misbehave if not run in a notebook with `%matplotlib inline` or "
            "`%matplotlib notebook`header."
        )

        self._check_fig_size(fig_size)

        if isinstance(profiles, xarrayDataset):
            # Group of profiles from Dataset
            n_rows = int(np.ceil(len(profiles.data_vars) / 2))
            _, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(8, 2 * n_rows) if fig_size is None else fig_size)
            axes = axes.flatten()

            for ii, name in enumerate(profiles.data_vars.keys()):
                self._plot_single_1d_profile(profiles[name], ax=axes[ii])
        else:
            # Single profile from DataArray
            _, axes = plt.subplots(nrows=1, figsize=(8, 4) if fig_size is None else fig_size)
            self._plot_single_1d_profile(profile=profiles, ax=axes)

        plt.show()
        plt.tight_layout()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_signal(
        self,
        data_origin: BaseDataSourceType,
        source_name: str,
        signal_name: str,
        fig_size: Optional[Union[list[int], set[int]]] = None,
    ) -> None:
        """
        Helper function for plotting individual signals.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting, either Mapping or a data source handle.
        source_name : str
            Name of target source.
        signal_name : str
            Name of target signal.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `source_name` not found.

        """

        self._check_fig_size(fig_size=fig_size)

        profiles = self.sig.get_source_profiles(data_origin=data_origin, source_name=source_name)
        if profiles is None:
            raise KeyError(f"Group `{source_name}` not found.")

        self.plot_1d_profiles(profiles=profiles[signal_name], fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_group(
        self, data_origin: BaseDataSourceType, source_name: str, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting entire group of signals.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for group plotting, either Mapping or a data source handle.
        source_name : str
            Name of target source.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `source_name` not found.

        """

        warnings.warn(
            "WARNING: This method may misbehave if not run in a notebook with `%matplotlib inline` or "
            "`%matplotlib notebook`header."
        )

        self._check_fig_size(fig_size=fig_size)

        profiles = self.sig.get_source_profiles(data_origin=data_origin, source_name=source_name)
        if profiles is None:
            raise KeyError(f"Group `{source_name}` not found.")

        n_rows = int(np.ceil(len(profiles.data_vars) / 2))
        _, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(8, 2 * n_rows) if fig_size is None else fig_size)
        axes = axes.flatten()

        for ii, name in enumerate(profiles.data_vars.keys()):
            signal = profiles[name]
            if len(signal.shape) == 1:
                self._plot_single_1d_profile(signal, ax=axes[ii])
            elif len(signal.shape) == 2:
                for channel in range(signal.shape[0]):
                    s = signal.isel({signal.dims[0]: channel})
                    s.plot(ax=axes[ii])
            axes[ii].set_title(name)

        for ax in axes:
            ax.grid("on", alpha=0.5)

        plt.show()
        plt.tight_layout()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_plasma_current(
        self, data_origin: BaseDataSourceType, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting `summary__plasma_current` signal.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting: Mapping, data source handle, or an already opened `xarray.DataTree`.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        """

        self.plot_signal(data_origin=data_origin, source_name="summary", signal_name="ip", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_power_nbi(
        self, data_origin: BaseDataSourceType, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting `summary__power_nbi` signal.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting: Mapping, data source handle, or an already opened `xarray.DataTree`.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        """

        self.plot_signal(data_origin=data_origin, source_name="summary", signal_name="power_nbi", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_magnetics(
        self, data_origin: BaseDataSourceType, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting `magnetics` group.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting: Mapping, data source handle, or an already opened `xarray.DataTree`.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        """

        self.plot_group(data_origin=data_origin, source_name="magnetics", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_spectrometer(
        self, data_origin: BaseDataSourceType, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting `spectrometer_visible` group.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting: Mapping, data source handle, or an already opened `xarray.DataTree`.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        """

        self.plot_group(data_origin=data_origin, source_name="spectrometer_visible", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_charge_exchange(
        self, data_origin: BaseDataSourceType, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting `charge_exchange` group.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting: Mapping, data source handle, or an already opened `xarray.DataTree`.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        """

        if fig_size is None:
            fig_size = [8, 4]

        self.plot_group(data_origin=data_origin, source_name="charge_exchange", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_thomson_scattering(
        self, data_origin: BaseDataSourceType, fig_size: Optional[Union[list[int], set[int]]] = None
    ) -> None:
        """
        Helper function for plotting `thomson_scattering` group.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for signal plotting: Mapping, data source handle, or an already opened `xarray.DataTree`.
        fig_size : Optional[Union[list[int], set[int]]]
            Size of target Matplotlib figure.
            Default: None.

        Returns
        -------
        None

        """

        if fig_size is None:
            fig_size = [8, 6]

        self.plot_group(data_origin=data_origin, source_name="thomson_scattering", fig_size=fig_size)

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
    # Settings

    # Creation of managers
    signal_manager = signal_utils.MASTSignalManager()
    plotting_manager = MASTPlottingManager()

    # Create source from shot_info
    store_from_shot_info = signal_manager.store_manager.make_shot_store(shot_info=ShotInfo(shot_id=30421, local=False))
    # print(f"store_from_shot_id.tree() (store store_from_shot_id shot if): {store_from_shot_id.tree()}\n")

    source_profiles = signal_manager.get_source_profiles(data_origin=store_from_shot_info, source_name="summary")

    # ..................................................................................................................
    # Plot 1d profiles

    if source_profiles is not None:
        plotting_manager.plot_1d_profiles(profiles=source_profiles, fig_size=[8, 8])
    else:
        print("Warning: No source profiles found.")

    # ..................................................................................................................
    # Plot specific group

    plotting_manager.plot_plasma_current(data_origin=store_from_shot_info)

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
