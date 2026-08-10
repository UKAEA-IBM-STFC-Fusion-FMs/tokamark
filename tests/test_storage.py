"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from MAST_tools.utils.data_utils import ShotInfo, StoreManagerParameters
from MAST_tools.utils.signal_utils import MASTSignalManager
from MAST_tools.utils.store_utils import MASTStorageManager, is_remote_uri

# ----------------------------------------------------------------------------------------------------------------------
# Fixture settings
#
# No real MAST data exists in this repo, so this suite generates a small synthetic `<shot_id>.nc` / `<shot_id>.zarr`
# file per test, mirroring the per-shot-file / per-source-group layout of the real dataset, then exercises the
# store/signal management layer against it. Local stores are written straight to `tmp_path`. Remote (S3) stores reuse
# the same on-disk fixture as the object a real bucket would serve, and redirect `xarray.open_datatree` to it via
# `unittest.mock.patch`, so no network access or S3 credentials are required to test the remote code path.

FIXTURE_SHOT_ID = 99999
DATA_FORMATS = ("netcdf", "zarr")
FILE_EXTENSIONS = {"netcdf": ".nc", "zarr": ".zarr"}
NETCDF_ENGINE = "h5netcdf"

_TIME = np.linspace(0.0, 1.0, 10)
_CHANNELS = np.array(["ch0", "ch1", "ch2"])


# ----------------------------------------------------------------------------------------------------------------------
def _fixture_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    """
    Build the two synthetic source datasets used by the fixtures.

    Returns
    -------
    tuple[xarray.Dataset, xarray.Dataset]
        A "magnetics" dataset (2D signal "flux_loop_flux" over channel/time), and a "summary" dataset (1D signal
        "power_nbi" over time).

    """

    rng = np.random.default_rng(seed=42)

    magnetics_ds = xr.Dataset(
        data_vars={"flux_loop_flux": (("channel", "time"), rng.normal(size=(len(_CHANNELS), len(_TIME))))},
        coords={"time": _TIME, "channel": _CHANNELS},
    )
    summary_ds = xr.Dataset(
        data_vars={"power_nbi": (("time",), rng.normal(size=len(_TIME)))},
        coords={"time": _TIME},
    )

    return magnetics_ds, summary_ds


# ----------------------------------------------------------------------------------------------------------------------
def _write_fixture(base_dir: Path, data_format: str, shot_id: int = FIXTURE_SHOT_ID) -> Path:
    """
    Write the synthetic shot fixture for a given data format under `base_dir`.

    Parameters
    ----------
    base_dir : Path
        Directory to write the fixture file/directory into.
    data_format : str
        Target data format, either "netcdf" or "zarr".
    shot_id : int
        Shot ID used for the fixture file name.
        Optional. Default: FIXTURE_SHOT_ID.

    Returns
    -------
    Path
        Path to the fixture file/directory, holding a "magnetics" group and a "summary" group.

    """

    fixture_path = base_dir / f"{shot_id}{FILE_EXTENSIONS[data_format]}"
    magnetics_ds, summary_ds = _fixture_datasets()

    if data_format == "netcdf":
        magnetics_ds.to_netcdf(fixture_path, group="magnetics", engine=NETCDF_ENGINE, mode="w")
        summary_ds.to_netcdf(fixture_path, group="summary", engine=NETCDF_ENGINE, mode="a")
    else:
        magnetics_ds.to_zarr(fixture_path, group="magnetics", mode="w")
        summary_ds.to_zarr(fixture_path, group="summary", mode="a")

    return fixture_path


# ----------------------------------------------------------------------------------------------------------------------
def _redirect_remote_open(fixture_path: Path, engine: str = "zarr"):
    """
    Patch `xarray.open_datatree` so that a remote (S3) URI transparently resolves to a local fixture file.

    This is what stands in for a real S3 bucket in the remote-store tests: it asserts that the store manager builds
    a remote URI and passes S3 `storage_options` through as usual, then serves the on-disk fixture instead of making
    a real network call.

    Parameters
    ----------
    fixture_path : Path
        Local fixture file/directory to serve in place of the remote object.
    engine : str
        `xarray` engine to use when opening `fixture_path`.
        Optional. Default: "zarr".

    Returns
    -------
    unittest.mock._patch
        Patch context manager/decorator for `MAST_tools.utils.store_utils.xr.open_datatree`.

    """

    real_open_datatree = xr.open_datatree

    def fake_open_datatree(handle, **kwargs):
        assert is_remote_uri(handle), f"Expected a remote URI, got {handle!r}"
        assert kwargs.get("storage_options"), "Remote opens must pass S3 `storage_options`."
        return real_open_datatree(fixture_path, engine=engine)

    return patch("MAST_tools.utils.store_utils.xr.open_datatree", side_effect=fake_open_datatree)


# ----------------------------------------------------------------------------------------------------------------------
# Pure-logic tests: URI/handle construction, no filesystem or network access.


@pytest.mark.parametrize(
    ("handle", "expected"),
    [
        ("s3://mast/tokamark/v1/30471.zarr", True),
        ("https://example.com/30471.zarr", True),
        ("/mast/tokamark/v1/30471.zarr", False),
        (Path("/mast/tokamark/v1/30471.zarr"), False),
    ],
)
def test_is_remote_uri(handle, expected) -> None:
    assert is_remote_uri(handle) is expected


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_shot_uri_local_vs_remote(tmp_path, data_format) -> None:
    store_manager = MASTStorageManager(
        data_format=data_format,
        base_local_data_path=str(tmp_path),
        s3_mast_dataset_path="/mast/tokamark/v1",
        target_fsspec_protocol="s3",
    )
    extension = FILE_EXTENSIONS[data_format]

    local_uri = store_manager._shot_uri(shot_id=FIXTURE_SHOT_ID, local=True)
    assert local_uri == str(tmp_path / f"{FIXTURE_SHOT_ID}{extension}")
    assert not is_remote_uri(local_uri)

    remote_uri = store_manager._shot_uri(shot_id=FIXTURE_SHOT_ID, local=False)
    assert remote_uri == f"s3://mast/tokamark/v1/{FIXTURE_SHOT_ID}{extension}"
    assert is_remote_uri(remote_uri)


def test_open_kwargs_storage_options_only_for_remote(tmp_path) -> None:
    store_manager = MASTStorageManager(data_format="zarr", base_local_data_path=str(tmp_path))

    local_kwargs = store_manager._open_kwargs(handle=str(tmp_path / "30471.zarr"))
    assert local_kwargs == {"engine": "zarr"}

    remote_kwargs = store_manager._open_kwargs(handle="s3://mast/tokamark/v1/30471.zarr")
    assert remote_kwargs == {
        "engine": "zarr",
        "storage_options": {"anon": True, "endpoint_url": store_manager.s3_endpoint_url},
    }


def test_list_all_shots_remote_parses_key_field(tmp_path) -> None:
    """`list_all_shots(local=False)` reads S3-style listings (dicts with a "Key" field), unlike the local path."""

    store_manager = MASTStorageManager(
        data_format="zarr", base_local_data_path=str(tmp_path), s3_mast_dataset_path="/mast/tokamark/v1"
    )
    fake_listing = [
        {"Key": "/mast/tokamark/v1/30471.zarr"},
        {"Key": "/mast/tokamark/v1/30472.zarr"},
        {"Key": "/mast/tokamark/v1/README.md"},
    ]

    with patch.object(store_manager, "_read_fsspec_listdir", return_value=fake_listing) as mocked_listdir:
        shot_ids = store_manager.list_all_shots(local=False)

    mocked_listdir.assert_called_once_with(path="/mast/tokamark/v1", local=False)
    assert shot_ids == [30471, 30472]


# ----------------------------------------------------------------------------------------------------------------------
# Local store tests: real temp-file fixtures, no mocking required.


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_local_store_roundtrip(tmp_path, data_format) -> None:
    _write_fixture(tmp_path, data_format)

    store_manager = MASTStorageManager(data_format=data_format, base_local_data_path=str(tmp_path))
    shot_info = ShotInfo(shot_id=FIXTURE_SHOT_ID, local=True)

    # make_shot_store / make_shot_group: the handle is a plain path str for both formats.
    store = store_manager.make_shot_store(shot_info=shot_info, verbose=True)
    assert isinstance(store, str), f"Expected a local path str, got {type(store)}"
    assert not is_remote_uri(store)

    with store_manager.open_shot_group(data_origin=store) as group:
        assert isinstance(group, xr.DataTree), f"Expected an xarray.DataTree, got {type(group)}"

        # An already opened tree must be passed through unchanged (idempotent).
        assert store_manager.make_shot_group(data_origin=group) is group

        all_signals_in_group = store_manager.get_all_signals_in_group(group=group)
        assert set(all_signals_in_group) == {"magnetics-flux_loop_flux", "summary-power_nbi"}, all_signals_in_group

    # get_all_sources / get_all_signals / are_signals_in_store
    all_sources = store_manager.get_all_sources(shot_ids=[FIXTURE_SHOT_ID], local=True)
    assert set(all_sources[FIXTURE_SHOT_ID]) == {"magnetics", "summary"}, all_sources

    all_signals = store_manager.get_all_signals(shot_ids=[FIXTURE_SHOT_ID], local=True)
    assert set(all_signals[FIXTURE_SHOT_ID]) == {
        "magnetics-flux_loop_flux",
        "summary-power_nbi",
    }, all_signals

    check_results = store_manager.are_signals_in_store(store=store, signals=["magnetics-flux_loop_flux", "abc"])
    assert check_results == {"magnetics-flux_loop_flux": True, "abc": False}, check_results


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_local_signal_manager_reads(tmp_path, data_format) -> None:
    """MASTSignalManager read path (used by MastDataset.__getitem__), against a local store."""

    _write_fixture(tmp_path, data_format)
    shot_info = ShotInfo(shot_id=FIXTURE_SHOT_ID, local=True)

    signal_manager = MASTSignalManager(
        store_manager_settings=StoreManagerParameters(data_format=data_format, base_local_data_path=str(tmp_path))
    )

    source_profile = signal_manager.get_source_profiles(data_origin=shot_info, source_name="magnetics")
    assert source_profile is not None
    assert "flux_loop_flux" in source_profile.data_vars

    # A missing source must be reported as None rather than raising.
    assert signal_manager.get_source_profiles(data_origin=shot_info, source_name="not_a_source") is None

    signal_values = signal_manager.get_signal_values(
        signal_name="flux_loop_flux", data_origin=shot_info, source_name="magnetics"
    )
    assert signal_values is not None
    assert signal_values.shape == (len(_CHANNELS), len(_TIME)), signal_values.shape

    signal_times, time_type = signal_manager.get_signal_times_and_time_type(
        signal_name="flux_loop_flux", data_origin=shot_info, source_name="magnetics"
    )
    assert time_type == "time"
    np.testing.assert_allclose(signal_times, _TIME)

    # Reading from an already opened DataTree must give the same values (the MastDataset.__getitem__ path).
    with signal_manager.store_manager.open_shot_group(data_origin=shot_info) as shot_tree:
        values_from_tree = signal_manager.get_signal_values(
            signal_name="flux_loop_flux", data_origin=shot_tree, source_name="magnetics"
        )
        np.testing.assert_allclose(values_from_tree, signal_values)


# ----------------------------------------------------------------------------------------------------------------------
# Remote (S3) store tests: same fixtures, but driven through the remote URI code path with `xr.open_datatree` mocked
# out (see `_redirect_remote_open()`), so no network access or S3 credentials are required.


def test_remote_zarr_store_roundtrip(tmp_path) -> None:
    fixture_path = _write_fixture(tmp_path, "zarr")
    store_manager = MASTStorageManager(
        data_format="zarr", base_local_data_path=None, s3_mast_dataset_path="/mast/tokamark/v1"
    )
    shot_info = ShotInfo(shot_id=FIXTURE_SHOT_ID, local=False)

    store = store_manager.make_shot_store(shot_info=shot_info, verbose=True)
    assert store == f"s3://mast/tokamark/v1/{FIXTURE_SHOT_ID}.zarr"
    assert is_remote_uri(store)

    with _redirect_remote_open(fixture_path), store_manager.open_shot_group(data_origin=store) as group:
        assert isinstance(group, xr.DataTree)
        all_signals_in_group = store_manager.get_all_signals_in_group(group=group)

    assert set(all_signals_in_group) == {"magnetics-flux_loop_flux", "summary-power_nbi"}, all_signals_in_group

    with _redirect_remote_open(fixture_path):
        check_results = store_manager.are_signals_in_store(store=store, signals=["magnetics-flux_loop_flux", "abc"])
    assert check_results == {"magnetics-flux_loop_flux": True, "abc": False}, check_results


def test_remote_zarr_signal_manager_read(tmp_path) -> None:
    fixture_path = _write_fixture(tmp_path, "zarr")
    signal_manager = MASTSignalManager(
        store_manager_settings=StoreManagerParameters(
            data_format="zarr", base_local_data_path=None, s3_mast_dataset_path="/mast/tokamark/v1"
        )
    )
    shot_info = ShotInfo(shot_id=FIXTURE_SHOT_ID, local=False)

    with _redirect_remote_open(fixture_path):
        signal_values = signal_manager.get_signal_values(
            signal_name="flux_loop_flux", data_origin=shot_info, source_name="magnetics"
        )
        signal_times, time_type = signal_manager.get_signal_times_and_time_type(
            signal_name="flux_loop_flux", data_origin=shot_info, source_name="magnetics"
        )

    assert signal_values is not None
    assert signal_values.shape == (len(_CHANNELS), len(_TIME)), signal_values.shape
    assert time_type == "time"
    np.testing.assert_allclose(signal_times, _TIME)


# ======================================================================================================================
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
