"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import yaml
import argparse
from pathlib import Path
import numpy as np
from typing import Any
from multiprocessing import cpu_count

import torch.multiprocessing as mp

from MAST_tools.utils.general_utils import warning_print
from tokamark.data import initialize_MAST_dataset
from tokamark.data_split import get_train_test_val_shots
from tokamark.tools.path import RANDOM_SPLIT_SIGNALS_STATS_FILE, TEMPORAL_SPLIT_SIGNALS_STATS_FILE

from preproc_paths import (
    OUTPUT_DIR,
    RANDOM_SPLIT_SIGNALS_MEAN_STD_TRAIN_FILE,
    TEMPORAL_SPLIT_SIGNALS_MEAN_STD_TRAIN_FILE,
)


# ----------------------------------------------------------------------------------------------------------------------
def to_python(obj: Any) -> Any:
    """
    Turn a numpy-based object into a numpy-free object.

    Parameters
    ----------
    obj : Any
        An arbitrary target object.

    Returns
    -------
    Any
        A numpy-free version of the input `obj` object.

    """

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [to_python(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


# ======================================================================================================================
if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="config_get_metadata.yaml",
        help="Path to the YAML file with configuration to get metadata.",
    )
    parser.add_argument(
        "--splitting_mode", type=str, choices=["random", "temporal"], default="random", help="Splitting mode."
    )
    parser.add_argument(
        "--signals_mean_std_train_file_path",
        type=str,
        required=False,
        help="User-defined path to the `dict_signals_mean_std_train.yaml` file.",
    )
    parser.add_argument(
        "--signals_stats_saving_file_path",
        type=str,
        required=False,
        help="User-defined path to the YAML file where signals statistics will be saved.",
    )
    parser.add_argument("--demo_mode", action="store_true", help="Activate demo mode.")
    parser.add_argument(
        "--demo_suffix", type=str, default="_DEMO", help="Suffix used in demo mode when saving results."
    )

    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------------------------------------------------------

    # Split-dependent configurations
    if args.splitting_mode == "random":
        if args.signals_mean_std_train_file_path is None:
            args.signals_mean_std_train_file_path = RANDOM_SPLIT_SIGNALS_MEAN_STD_TRAIN_FILE
        if args.signals_stats_saving_file_path is None:
            args.signals_stats_saving_file_path = RANDOM_SPLIT_SIGNALS_STATS_FILE

    elif args.splitting_mode == "temporal":
        if args.signals_mean_std_train_file_path is None:
            args.signals_mean_std_train_file_path = TEMPORAL_SPLIT_SIGNALS_MEAN_STD_TRAIN_FILE
        if args.signals_stats_saving_file_path is None:
            args.signals_stats_saving_file_path = TEMPORAL_SPLIT_SIGNALS_STATS_FILE

    # Load Task YAML config
    with open(args.config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # REMARK: Demo mode could be enforced (e.g., for testing) by uncommenting the following line.
    # args.demo_mode = True

    # If required, override values with settings for demo mode
    if args.demo_mode:
        warning_print("Running in demo mode.")

        config["get_shots_settings"]["max_index"] = 2
        config["get_shots_settings"]["shuffle"] = True

        args.signals_stats_saving_file_path = Path(OUTPUT_DIR) / f"dict_signals_stats{args.demo_suffix}.yaml"

    # REMARK: Default values for `store_manager_settings` can be overridden as follows:
    # config["store_manager_settings"]["base_local_data_path"] = "/path/to/local/dataset"

    # ------------------------------------------------------------------------------------------------------------------
    # Specific Data Preprocessing for LCFS profiles
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(**config["get_shots_settings"])

    # ------------------------------------------------------------------------------------------------------------------
    # Load dict_signals_mean_std_train.yaml
    # ------------------------------------------------------------------------------------------------------------------

    with open(args.signals_mean_std_train_file_path) as f:
        dict_mean_std = yaml.safe_load(f)

    # ------------------------------------------------------------------------------------------------------------------
    # Create unstandardized train dataset
    # ------------------------------------------------------------------------------------------------------------------

    preprocessing_train_dataset = initialize_MAST_dataset(
        config_task=config["task_configuration"],
        shots_list=train_shots_[::-1],
        local_flag=config["local"],
        use_std_scaling=False,  # <- To get an unstandardized dataset
        remove_outliers=True,  # <- To mitigate effect of outliers in the calculation of signal statistics
        store_manager_settings=config["store_manager_settings"],
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------------------------------------------------------

    dict_metadata = {}

    # Get expected variable count
    first_sample = next(iter(preprocessing_train_dataset))
    target_vars = set(first_sample.keys())

    for i, sample in enumerate(preprocessing_train_dataset):  # noqa - Ignore type check warning
        # print(i)

        for var, signal in sample.items():
            # Only compute once per variable
            if var in dict_metadata:
                continue

            time = np.array(signal.get("time", []))
            values = np.array(signal.get("values", []))

            # Skip invalid signal
            if len(time) == 0 or len(values) == 0:
                print("Skipping var", var)
                continue

            # print("\nSaving var", var)

            # Compute median dt
            dt = round(np.median(np.diff(time)), 6)

            dict_metadata[var] = {
                "dt": dt,  # <- Time granularity
                "values_shape": values.shape[:-1],  # <- Exclude time dimension
                "mean": dict_mean_std[var]["mean"]["no_outliers_z6"],
                "std": dict_mean_std[var]["std"]["no_outliers_z6"],
            }

        # Stop once all variables are filled
        if set(dict_metadata.keys()) == target_vars:
            print("Metadata dictionary `dict_metadata` fully filled. Stopping.")
            break

    # ------------------------------------------------------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------------------------------------------------------

    if len(dict_metadata) == 0:
        raise ValueError("No valid signals found within limit.")

    # ------------------------------------------------------------------------------------------------------------------
    # Save and clean metadata
    # ------------------------------------------------------------------------------------------------------------------

    clean_dict = to_python(obj=dict_metadata)
    with open(args.signals_stats_saving_file_path, "w+") as f_:
        print(f"Saving results to file `{args.signals_stats_saving_file_path}`.")
        yaml.dump(data=clean_dict, stream=f_, sort_keys=False)

    # ------------------------------------------------------------------------------------------------------------------
