"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html

Ad-hoc smoke test for loading a local folder of NetCDF shot files.

Can be run either as a plain script, pointed at a folder on the command line:

    uv run python scripts/test_netcdf_local_load.py /path/to/your/netcdf/folder
    uv run python scripts/test_netcdf_local_load.py /path/to/your/netcdf/folder --shot-id 30471

or as a pytest test, pointed at a folder via an environment variable (skipped if unset):

    NETCDF_LOCAL_TEST_DATA_PATH=/path/to/your/netcdf/folder uv run pytest scripts/test_netcdf_local_load.py
"""

import argparse
import os

import pytest
import xarray as xr

from MAST_tools.utils.store_utils import MASTStorageManager

DATA_PATH_ENV_VAR = "NETCDF_LOCAL_TEST_DATA_PATH"


# ----------------------------------------------------------------------------------------------------------------------
def check_local_netcdf_folder_loads(data_path: str, shot_id: int | None = None) -> None:
    """
    Run load/structure/signal assertions against a local folder of `<shot_id>.nc` files.

    Parameters
    ----------
    data_path : str
        Folder containing `<shot_id>.nc` files.
    shot_id : int | None
        Shot ID to open. If None, the first shot found in `data_path` is used.
        Optional. Default: None.

    """

    store = MASTStorageManager(data_format="netcdf", base_local_data_path=data_path)

    shot_ids = store.list_all_shots(local=True)
    assert shot_ids, f"No shots found under {data_path!r} - check filenames match '<shot_id>.nc'."

    if shot_id is None:
        shot_id = shot_ids[0]
    else:
        assert shot_id in shot_ids, f"Shot {shot_id} not found under {data_path!r}."

    shot_store = store.make_shot_store({"shot_id": shot_id, "local": True})

    with store.open_shot_group(shot_store) as shot_tree:
        assert isinstance(shot_tree, xr.DataTree)
        assert len(shot_tree.children) > 0, f"Shot {shot_id} has no source groups."

        for source_name, source_node in shot_tree.children.items():
            assert len(source_node.dataset.data_vars) > 0, f"Source group {source_name!r} has no data variables."

    signals = store.get_all_signals_in_store(shot_store)
    assert signals, f"No signals found in shot {shot_id}."
    assert all("-" in signal for signal in signals), "Expected signals in '<source>-<signal>' format."

    print(f"OK: {len(shot_ids)} shot(s) found, shot {shot_id} loaded with {len(signals)} signal(s).")


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(DATA_PATH_ENV_VAR not in os.environ, reason=f"{DATA_PATH_ENV_VAR} not set")
def test_local_netcdf_folder_loads() -> None:
    check_local_netcdf_folder_loads(data_path=os.environ[DATA_PATH_ENV_VAR])


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local NetCDF shot loading.")
    parser.add_argument("data_path", help="Folder containing <shot_id>.nc files.")
    parser.add_argument(
        "--shot-id",
        type=int,
        default=None,
        help="Shot ID to open. Defaults to the first shot found in data_path.",
    )
    args = parser.parse_args()

    check_local_netcdf_folder_loads(data_path=args.data_path, shot_id=args.shot_id)


if __name__ == "__main__":
    main()
