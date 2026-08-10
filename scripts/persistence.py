"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import argparse
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence, cast, Literal
import numpy as np
from multiprocessing import cpu_count
import psutil
from datetime import datetime

import torch
from torch import Tensor as TorchTensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate  # noqa - Ignore "access to protected method" warning
import torch.multiprocessing as mp

from MAST_tools.utils.general_utils import warning_print
from MAST_tools.utils.path_utils import (
    RANDOM_SPLIT_OUTLIER_METADATA_FILE,
    TEMPORAL_SPLIT_OUTLIER_METADATA_FILE,
)
from tokamark.tools.utils import get_config_from_yaml
from tokamark.tasks import get_task_config, get_task_metadata, get_signals_metadata
from tokamark.data_split import get_train_test_val_shots
from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset
from tokamark.evaluator import WindowMetricsAccumulator, compute_metrics, compute_summary_metrics
from tokamark.tools.path import (
    RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
    RANDOM_SPLIT_SIGNALS_STATS_FILE,
    TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
    TEMPORAL_SPLIT_SIGNALS_STATS_FILE,
)

# ----------------------------------------------------------------------------------------------------------------------

CONFIG_FILES_DIR = Path("/.config_files")


# ======================================================================================================================
class PersistenceTransform:
    """
    Class for the Persistence Transform.

    Attributes
    ----------
    signals : list[str]
        List of source-signal items.

    Methods
    -------
    __call__(segment)
        Call method for the class instances to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, signals: list[str]) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        signals : list[str]
            List of source-signal items.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.signals = signals

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, segment: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        segment : Mapping[str, Any]
            Dictionary with metadata of target segment.

        Returns
        -------
        dict[str, Any]
            Signal data sample.

        """

        # ..............................................................................................................
        def get_signals_data(label: str) -> dict[str, Any]:
            """Get signal data given target label."""

            subset = {}
            for signal_name in self.signals:
                if signal_name in segment[label]:
                    subset[signal_name] = segment[label][signal_name]["values"]

            return subset

        # ..............................................................................................................

        return {
            "shot_id": segment["shot_id"],
            "window_index": segment["window_index"],
            "x": get_signals_data("input"),
            "y": get_signals_data("output"),
        }

        # ..............................................................................................................


# ----------------------------------------------------------------------------------------------------------------------
def MAST_collate_fn(  # noqa - Ignore lowercase warning
    batch: Sequence, verbose: bool = True
) -> list | None:
    """
    MAST collate function.

    Parameters
    ----------
    batch: Sequence
        Batch of data samples.
    verbose : bool
        If True, activate verbose mode.
        Optional. Default: True.

    Returns
    -------
    list | None
        Flattened batch, or None if batch is empty.

    """

    # ..............................................................................................................
    def tensor_size(t: TorchTensor) -> float:
        """Get size of given tensor."""
        return t.nelement() * t.element_size() / 1024**2

    # ..............................................................................................................
    # Flatten the batch of lists into a single list, and return it

    # New (working) approach
    flattened_batch = [(item["shot_id"], item["window_index"], item["x"], item["y"]) for item in batch]
    flattened_batch = default_collate(batch=flattened_batch) if (len(flattened_batch) > 0) else None

    if verbose:
        proc = psutil.Process(pid=os.getpid())
        mem = proc.memory_info().rss / (1024**2)
        print(f"[Worker PID={proc.pid}] Memory={mem:.2f} MB")

        if flattened_batch is None:
            print("batch is None")
        else:
            flattened_batch = cast(list, flattened_batch)
            print(f"The batch contains: {len(batch)} shots and {len(flattened_batch[0])} samples")

            x = flattened_batch[2]
            y = flattened_batch[3]
            size = sum([tensor_size(t=t) for _, t in x.items()])
            size += sum([tensor_size(t=t) for _, t in y.items()])
            print(f"X and Y tensors memory={size:.2f} MB")

    return flattened_batch

    # ..............................................................................................................


# ----------------------------------------------------------------------------------------------------------------------
def forward_fill_last_dim(x: Any):
    """Forward-fill last dimension of the specified object."""

    # x: (..., T)
    mask = torch.isnan(x)

    # Indices of valid values
    idx = torch.where(~mask, torch.arange(x.size(-1), device=x.device), 0)

    # Forward-fill indices using cumulative max
    idx = idx.cummax(dim=-1).values

    # Gather values using filled indices
    return torch.gather(x, dim=-1, index=idx)


# ----------------------------------------------------------------------------------------------------------------------
def persistence_evaluation_loop(  # NOSONAR - Ignore cognitive complexity
    test_dataloader: DataLoader,
    feature_names: list[str],
    signal_metadata_path: str,
    accumulator: WindowMetricsAccumulator,
    model: str = "persistence",
):
    """
    Evaluate persistence model per shot/window and populate an accumulator.

    Parameters
    ----------
    test_dataloader : DataLoader
        Test DataLoader instance.
    feature_names : list[str]
        List of feature (signal) names.
    signal_metadata_path : str
        Target file path for signals metadata.
    accumulator : WindowMetricsAccumulator
        Input WindowMetricsAccumulator instance.
    model : str
        Model type. Valid options are "persistence" and "mean".
        Optional. Default: "persistence".

    Returns
    -------
    None

    """

    # ..................................................................................................................
    def contains_nans(data: Any) -> bool:  # noqa - Ignore unused function warning
        """Check if given data contains NaN values."""
        return any(np.isnan(x_).any() for x_ in data)

    # ..................................................................................................................

    # ..................................................................................................................
    # Evaluation loop
    # ..................................................................................................................

    for batch_idx, batch_ in enumerate(test_dataloader):
        if batch_ is None:
            warning_print(f"Empty batch {batch_idx} skipped.")
            continue

        shot_ids, window_indices, x_test, y_test = batch_  # noqa - Right number of values to unpack

        print(f"\n-> Processing batch {batch_idx} with {len(shot_ids.unique())} shots and {len(shot_ids)} elements.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model prediction
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        y_pred = {}
        if model.startswith("persistence"):
            signal_metadata = get_signals_metadata(file_path=signal_metadata_path)
            for key, x in x_test.items():
                x_test_filled = forward_fill_last_dim(x=x)
                last = x_test_filled[..., -1].unsqueeze(-1)

                # 🔁 Fallback: replace remaining NaNs with mean
                if torch.isnan(last).any():
                    warning_print("Fallback in persistence model due to NaN presence: NaNs replaced with mean.")
                    mean = signal_metadata[key]["mean"]
                    last = torch.nan_to_num(input=last, nan=mean)

                y_pred[key] = last.expand_as(other=y_test[key])

        elif model.startswith("mean"):
            signal_metadata = get_signals_metadata(file_path=signal_metadata_path)
            for key in y_test:
                mean = signal_metadata[key]["mean"]
                y_pred[key] = torch.ones_like(input=y_test[key]) * mean

        else:
            warning_print(f"Persistence pipeline: model {model} is not known.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute RMSEs per feature
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for feature_name in feature_names:
            try:
                y_t = y_test[feature_name].squeeze(1).reshape(len(shot_ids), -1).numpy()
                y_p = y_pred[feature_name].squeeze(1).reshape(len(shot_ids), -1).numpy()

                accumulator.add_batch(
                    y_target=y_t,
                    y_pred=y_p,
                    shot_ids=shot_ids,
                    window_indices=window_indices,
                    feature_name=feature_name,
                )

            except KeyError:
                warning_print(
                    f"Missing feature {feature_name} for batch {batch_idx}, or last elements are NaN values. Skipping."
                )

    print("\n✅ Evaluation done. Window metrics accumulator is ready.")
    print("---------------------------------------------------------\n")


# ----------------------------------------------------------------------------------------------------------------------
def run_persistence_pipeline(
    task: str, split: Literal["random", "temporal"], pipeline_config: Mapping[str, Any]
) -> None:
    """

    Run persistence pipeline.

    Parameters
    ----------
    task : str
        Target benchmark task.
    split : Literal["random", "temporal"]
        Splitting method to be used.
    pipeline_config : Mapping[str, Any]
        Dictionary with pipeline configuration.

    Returns
    -------
    None

    """

    # ..................................................................................................................
    # Configure proper split
    # ..................................................................................................................

    if split == "random":
        print("Random splitting")

        DATA_SPLIT = RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE  # noqa - Ignore lowercase warning
        OUTLIER_FILE = RANDOM_SPLIT_OUTLIER_METADATA_FILE  # noqa - Ignore lowercase warning
        SIGNAL_STATS = RANDOM_SPLIT_SIGNALS_STATS_FILE  # noqa - Ignore lowercase warning

    elif split == "temporal":
        print("Temporal splitting")

        DATA_SPLIT = TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE  # noqa - Ignore lowercase warning
        OUTLIER_FILE = TEMPORAL_SPLIT_OUTLIER_METADATA_FILE  # noqa - Ignore lowercase warning
        SIGNAL_STATS = TEMPORAL_SPLIT_SIGNALS_STATS_FILE  # noqa - Ignore lowercase warning

    else:
        raise ValueError(f"Unknown split: {split}")

    # ..................................................................................................................
    # Load task configuration and modify it to keep auto-regressive data
    # ..................................................................................................................

    config_task = get_task_config(task_name=task)
    target_model = pipeline_config["persistence_settings"]["model"]

    if target_model == "persistence":
        # Pipeline for "persistence" model

        # 0. Set task type as markovian, so that collate function does not fail due to varying shapes.
        config_task["task_type"] = "markovian"

        # 1. Identify chosen signals.
        ins = [f"{source}-{signal}" for source, signal in config_task["sources_and_signals"]["input_name"]]
        outs = [f"{source}-{signal}" for source, signal in config_task["sources_and_signals"]["output_name"]]
        chosen_signals = [signal for signal in outs if signal in ins]

        # 2. Update values of `input_name` key.
        config_task["sources_and_signals"]["input_name"] = [
            [source, signal]
            for source, signal in config_task["sources_and_signals"]["input_name"]
            if f"{source}-{signal}" in chosen_signals
        ]

        # 3. Re-assign `input_keys` values with updated `input_name` values.
        config_task["task_window_segmenter"]["input_keys"] = config_task["sources_and_signals"]["input_name"]

        # 4. Update values of `output_name` key.
        config_task["sources_and_signals"]["output_name"] = [
            [source, signal]
            for source, signal in config_task["sources_and_signals"]["output_name"]
            if f"{source}-{signal}" in chosen_signals
        ]

        # 5. Re-assign `output_keys` values with updated `output_name` values.
        config_task["task_window_segmenter"]["output_keys"] = config_task["sources_and_signals"]["output_name"]

        # 6. Set `actuator_name` and `actuator_keys` values to None.
        config_task["sources_and_signals"]["actuator_name"] = None
        config_task["task_window_segmenter"]["actuator_keys"] = None

    elif target_model == "mean":
        # Pipeline for "mean" model

        # 1. Identify chosen signals
        chosen_signals = [f"{source}-{signal}" for source, signal in config_task["sources_and_signals"]["output_name"]]

        # 2. Set values of `input_name` key with a dummy signal.
        # REMARK: A dummy signal is used as no inputs create errors (although inputs are not used in mean pipeline).
        config_task["sources_and_signals"]["input_name"] = [["magnetics", "flux_loop_flux"]]

        # 3. Re-assign `input_keys` values with updated `input_name` values.
        config_task["task_window_segmenter"]["input_keys"] = config_task["sources_and_signals"]["input_name"]

        # 4. Set `actuator_name` and `actuator_keys` values to None.
        config_task["sources_and_signals"]["actuator_name"] = None
        config_task["task_window_segmenter"]["actuator_keys"] = None

    else:
        raise ValueError(f"Unknown persistence model {target_model}.")

    # ..................................................................................................................
    # Proceed if chosen signals are available
    # ..................................................................................................................

    if len(chosen_signals) > 0:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize task-specific metadata
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dict_task_metadata = get_task_metadata(config_task=config_task, verbose=False)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize MAST datasets
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _, test_shots_, _ = get_train_test_val_shots(
            **pipeline_config["get_shots_settings"], data_splits_file_path=DATA_SPLIT
        )
        print(f"Number of test shots: {len(test_shots_)}")

        test_mast_dataset = initialize_MAST_dataset(
            config_task=config_task,
            shots_list=test_shots_,
            local_flag=pipeline_config["local"],
            use_std_scaling=False,
            remove_outliers=True,
            outlier_metadata_file=OUTLIER_FILE,
            store_manager_settings=pipeline_config["store_manager_settings"],
        )

        test_dataset = initialize_TokaMark_dataset(
            dataset=test_mast_dataset,
            task_metadata=dict_task_metadata,
            config_metadata=config_task,
            custom_transform=PersistenceTransform(signals=chosen_signals),
            test_mode=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,  # noqa - Ignore expected type warning
            collate_fn=MAST_collate_fn,
            **pipeline_config["dataloader_settings"],
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        accumulator = WindowMetricsAccumulator(task=task)

        persistence_evaluation_loop(
            test_dataloader=test_dataloader,
            feature_names=chosen_signals,
            signal_metadata_path=SIGNAL_STATS,
            accumulator=accumulator,
            model=pipeline_config["persistence_settings"]["model"],
        )

        if accumulator.is_empty():
            print(f"No windows metrics were computed for task {task}.")
            return

        compute_metrics(
            task=task,
            output_dir=Path(pipeline_config["persistence_settings"]["output_dir"]) / split,
            window_metrics_accumulator=accumulator,
            save_windows_metrics=pipeline_config["persistence_settings"]["save_windows_metrics"],
            save_task_metrics=pipeline_config["persistence_settings"]["save_task_metrics"],
        )
    else:
        print("No common signals between input and output - not possible to run persistence.")  # REMARK: Only case.


# ======================================================================================================================
if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistence_model",
        type=str,
        choices=["persistence", "mean"],
        default="persistence",
        help="Target model for the persistence pipeline.",
    )
    parser.add_argument("--split", type=str, choices=["random", "temporal"], default="random", help="Splitting used.")
    parser.add_argument("--demo_mode", action="store_true", help="Activate demo mode.")
    parser.add_argument(
        "--demo_suffix", type=str, default="_DEMO", help="Suffix used in demo mode when saving results."
    )

    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------------------------------------------------------

    # Load Persistence YAML config
    config_filepath = CONFIG_FILES_DIR / f"config_{args.persistence_model}_model.yaml"
    config = get_config_from_yaml(file_path=config_filepath)

    # REMARK: Demo mode could be enforced for testing purposes by uncommenting the following line.
    # args.demo_mode = True

    # If required, override values with settings for demo mode
    if args.demo_mode:
        warning_print("Running in demo mode.")

        config["get_shots_settings"]["max_index"] = 2
        config["get_shots_settings"]["shuffle"] = False

        config["persistence_settings"]["output_dir"] += args.demo_suffix
        config["persistence_settings"]["ar_tasks"] = [config["persistence_settings"]["ar_tasks"][0]]

    # REMARK: Default values for `store_manager_settings` can be overridden here as follows:
    # config["store_manager_settings"]["base_local_data_path"] = "/path/to/local/dataset"

    # ------------------------------------------------------------------------------------------------------------------
    # Looping over auto-regressive tasks
    # ------------------------------------------------------------------------------------------------------------------

    output_dir = Path(config["persistence_settings"]["output_dir"]) / args.split

    print(f"Start time: {datetime.now()}\n")

    for task_ in config["persistence_settings"]["ar_tasks"]:
        print("---------------------------------------------")
        print(f"Running persistence pipeline for {task_}, model `{args.persistence_model}`...\n")
        run_persistence_pipeline(task=task_, split=args.split, pipeline_config=config)
        print(f"\nFinished `{args.persistence_model}` pipeline for {task_} [{datetime.now()}].")
        print("=========================================================\n")

    if config["persistence_settings"]["compute_summary_metrics"]:
        print("---------------------------------------------------------")
        print(f"Computing all metrics for `{args.persistence_model}` pipeline....\n")
        compute_summary_metrics(output_dir=output_dir)
        print(f"\nFinished computation of all metrics for `{args.persistence_model}` pipeline [{datetime.now()}].")
        print("=========================================================\n\n")

    print(f"\nAll done. Results saved under the target directory `{output_dir}` [{datetime.now()}].")

    # ------------------------------------------------------------------------------------------------------------------
