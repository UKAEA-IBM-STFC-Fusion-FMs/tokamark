"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html

Benchmark evaluation utilities for MAST tasks.

Typical usage:
1. During inference, create `WindowMetricsAccumulator(task)` and call `add_batch(...)` for each model batch.
2. After one task is complete, call:
   `compute_metrics(task, output_dir, window_metrics_accumulator, ...)`.
   Normalization is disabled by default. To also compute NRMSE/NMAE, pass
   `normalization=True, normalization_path=<reference statistics YAML>`.
   The caller selects the reference; the evaluator never infers a split or a default statistics path.
   Default save flags are:
   - `save_windows_metrics=False`
   - `save_shot_metrics=False`
   - `save_task_metrics=True`
   With defaults, only `<output_dir>/<task>/task_metrics.csv` is saved. To enable other files, pass:
   - `save_windows_metrics=True` for `<output_dir>/<task>/windows_metrics.csv`
   - `save_shot_metrics=True` for `<output_dir>/<task>/shots_metrics.csv` (you can enable any combination of the three
     flags).
3. After all tasks are complete, call
   `compute_summary_metrics(output_dir, source=...)` to produce `signals_metrics.csv` and `groups_metrics.csv` in
   `output_dir`.
   Supported `source` values are:
   - `source="task_metrics"` (default): reads `task_metrics.csv` per task
   - `source="windows_metrics"`: reads `windows_metrics.csv` per task
   - `source="shots_metrics"`: reads `shots_metrics.csv` per task
   Window CSVs can be normalized with the same explicit options as `compute_metrics`.
   Task/shot CSVs retain their stored metrics; normalization options must be omitted for those sources.

Aggregation follows the benchmark hierarchy:
- signal rows: equal-weight mean/std across shots
- task rows: equal-weight mean/std across shots
- group rows: equal-weight mean of task means and mean of task stds
"""

import os.path
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from torch import Tensor as TorchTensor

from tokamark.tasks import GROUP_TASKS, TASKS_CONFIGS_MAP, get_signals_metadata

# ----------------------------------------------------------------------------------------------------------------------

ID_COLUMNS = ["shot_id", "window_index", "feature_name"]
METRIC_COLUMNS = ["RMSE", "MAE", "nan_fraction"]
COLUMNS = ID_COLUMNS + METRIC_COLUMNS
TASK_METRICS = ("NRMSE", "NMAE", "RMSE", "MAE", "nan_fraction")

METRIC_MEAN_COLUMNS = tuple(f"{m}_mean" for m in TASK_METRICS)
METRIC_STD_COLUMNS = tuple(f"{m}_std_pop" for m in TASK_METRICS)
RAW_TO_MEAN = dict(zip(TASK_METRICS, METRIC_MEAN_COLUMNS))
MEAN_TO_RAW = dict(zip(METRIC_MEAN_COLUMNS, TASK_METRICS))

TASK_SUMMARY_COLUMNS = ("n_shots",) + tuple(c for pair in zip(METRIC_MEAN_COLUMNS, METRIC_STD_COLUMNS) for c in pair)
SIGNAL_STD_COLUMNS = METRIC_STD_COLUMNS
SIGNAL_VALUE_COLUMNS = METRIC_MEAN_COLUMNS
SHOT_SIGNAL_COLUMNS = ("shot_id", "feature_name", "n_windows") + METRIC_MEAN_COLUMNS

WINDOW_METRICS_FILENAME = "windows_metrics.csv"
SHOT_METRICS_FILENAME = "shots_metrics.csv"
SIGNAL_METRICS_FILENAME = "signals_metrics.csv"
TASK_METRICS_FILENAME = "task_metrics.csv"
GROUP_METRICS_FILENAME = "groups_metrics.csv"


# ----------------------------------------------------------------------------------------------------------------------
def compute_windows_metrics(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    shot_ids: np.ndarray | TorchTensor,
    window_indices: np.ndarray | TorchTensor,
    feature_name: str,
) -> pd.DataFrame:
    """
    Compute RMSE/MAE per window for one feature and return a dataframe chunk.

    Parameters
    ----------
    y_target : np.ndarray
        Input target data in np.ndarray format.
    y_pred : np.ndarray
        Input predicted data in np.ndarray format.
    shot_ids : Union[np.ndarray, TorchTensor]
        Torch tensor with shot IDs.
    window_indices : Union[np.ndarray, TorchTensor]
        Torch tensor with window indices.
    feature_name : str
        Name of target feature.

    Returns
    -------
    pd.DataFrame
        Windows metrics dataframe.

    """

    # Errors (ignore NaNs in calculations)
    diff = y_target - y_pred

    rmse_per_sample = np.sqrt(np.nanmean(diff**2, axis=1))
    mae_per_sample = np.nanmean(np.abs(diff), axis=1)

    # NaN percentage per sample
    nan_mask = np.isnan(y_target)
    nan_pct_per_sample = np.mean(nan_mask, axis=1)  # fraction of NaNs (0–1)

    # Build columns explicitly to preserve numeric dtypes (avoid mixed-type upcast).
    return pd.DataFrame(
        {
            "shot_id": np.asarray(shot_ids),
            "window_index": np.asarray(window_indices),
            "feature_name": np.asarray([feature_name] * len(rmse_per_sample), dtype=object),
            "RMSE": np.asarray(rmse_per_sample, dtype=float),
            "MAE": np.asarray(mae_per_sample, dtype=float),
            "nan_fraction": np.asarray(nan_pct_per_sample, dtype=float),
        }
    )


# ======================================================================================================================
class WindowMetricsAccumulator:
    """
    In-memory collector of per-window metrics for one task.

    Attributes
    ----------
    self.task : str
        Selected task.
    self._chunks : list[pd.DataFrame]
        Supporting variable to hold list of chunks.

    Methods
    -------
    add_batch(y_target, y_pred, shot_ids, window_indices, feature_name)
        Compute one metrics chunk and append it to the accumulator.
    is_empty()
        Return True when no window chunks were collected.
    to_dataframe()
        Concatenate all collected chunks into one windows dataframe.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, task: str) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        task : str
            Selected task.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        self.task = str(task)
        self._chunks: list[pd.DataFrame] = []

    # ------------------------------------------------------------------------------------------------------------------
    def add_batch(
        self,
        y_target: np.ndarray,
        y_pred: np.ndarray,
        shot_ids: np.ndarray | TorchTensor,
        window_indices: np.ndarray | TorchTensor,
        feature_name: str,
    ) -> None:
        """
        Compute one metrics chunk and append it to the accumulator.

        Parameters
        ----------
        y_target : np.ndarray
            Input target data in np.ndarray format.
        y_pred : np.ndarray
            Input predicted data in np.ndarray format.
        shot_ids : Union[np.ndarray, TorchTensor]
            Torch tensor with shot IDs.
        window_indices : Union[np.ndarray, TorchTensor]
            Torch tensor with window indices.
        feature_name : str
            Name of target feature.

        Returns
        -------
        None

        """
        self._chunks.append(
            compute_windows_metrics(
                y_target=y_target,
                y_pred=y_pred,
                shot_ids=shot_ids,
                window_indices=window_indices,
                feature_name=feature_name,
            )
        )

    # ------------------------------------------------------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Return True when no window chunks were collected."""
        return len(self._chunks) == 0

    # ------------------------------------------------------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate all collected chunks into one windows dataframe."""
        if self.is_empty():
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat(self._chunks, ignore_index=True)

    # ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def _build_task_metrics_df(task: str, df_signals: pd.DataFrame, df_task_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Build the task dataframe (signal rows + one task summary row).

    Parameters
    ----------
    task : str
        Target signal.
    df_signals : pd.DataFrame
        Dataframe with signals scores across shots.
    df_task_shots : pd.DataFrame
        Dataframe with task-level score per shot.

    Returns
    -------
    pd.DataFrame
        Dataframe with task metrics.

    """

    df_task_metrics = df_signals.reset_index().rename(columns={"index": "feature_name"})
    df_task_metrics["feature_name"] = df_task_metrics["feature_name"].astype(str)

    summary = _build_task_summary_row_from_shots(df_task_shots=df_task_shots)
    task_row_df = pd.DataFrame([{"feature_name": task, **summary}])

    for col in TASK_SUMMARY_COLUMNS:
        if col not in df_task_metrics.columns:
            df_task_metrics[col] = np.nan

    df_task_metrics = pd.concat([df_task_metrics, task_row_df], ignore_index=True)

    ordered_cols = ["feature_name"] + list(TASK_SUMMARY_COLUMNS)
    remaining_cols = [c for c in df_task_metrics.columns if c not in ordered_cols]

    return df_task_metrics[ordered_cols + remaining_cols]


# ----------------------------------------------------------------------------------------------------------------------
def _build_task_summary_row_from_shots(df_task_shots: pd.DataFrame) -> dict[str, Any]:
    """
    Build one task-level summary row from shot-level metrics.

    Parameters
    ----------
    df_task_shots : pd.DataFrame
        Input dataframe with shot-level metrics.

    Returns
    -------
    dict[str, Any]
        Dictionary with one task-level summary row from shot-level metrics.

    """

    summary = {"n_shots": float(len(df_task_shots))}
    for metric in TASK_METRICS:
        if metric in df_task_shots.columns:
            vals = pd.to_numeric(df_task_shots[metric], errors="coerce").dropna()
        else:
            vals = pd.Series(dtype=float)

        if len(vals) == 0:
            mean_val = np.nan
            std_val = np.nan
        else:
            mean_val = float(vals.mean())
            std_val = float(vals.std(ddof=0))

        summary[f"{metric}_mean"] = mean_val
        summary[f"{metric}_std_pop"] = std_val

    return summary


# ----------------------------------------------------------------------------------------------------------------------
def _build_group_summary_from_task_summaries(df_tasks: pd.DataFrame) -> dict[str, Any]:
    """
    Build one group summary row from equal-weight task means/stds.

    Parameters
    ----------
    df_tasks : pd.DataFrame
        Dataframe with task summaries.

    Returns
    -------
    dict[str, Any]
        Dataframe with group task summary.

    """

    group = {"n_shots": float(pd.to_numeric(df_tasks["n_shots"], errors="coerce").sum())}

    for metric in TASK_METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std_pop"
        if mean_col not in df_tasks.columns:
            group[mean_col] = np.nan
            group[std_col] = np.nan
            continue

        tmp = df_tasks[[mean_col, std_col]].copy()
        tmp[mean_col] = pd.to_numeric(tmp[mean_col], errors="coerce")
        tmp[std_col] = pd.to_numeric(tmp[std_col], errors="coerce")
        tmp = tmp.dropna()

        if len(tmp) == 0:
            group[mean_col] = np.nan
            group[std_col] = np.nan
            continue

        means = tmp[mean_col].to_numpy(dtype=float)
        stds = tmp[std_col].to_numpy(dtype=float)

        group[mean_col] = float(np.mean(means))
        group[std_col] = float(np.mean(stds))

    return group


# ----------------------------------------------------------------------------------------------------------------------
def _extract_task_summary_from_task_metrics(
    task: str, output_dir: str | Path
) -> tuple[None, None] | tuple[dict[str, Any], pd.DataFrame]:
    """
    Return one task summary row along with signal rows from the task metrics CSV file defined in
    `TASK_METRICS_FILENAME`.

    Parameters
    ----------
    task : str
        Target task.
    output_dir : Union[str, Path]
        Target output directory.

    Returns
    -------
    Union[tuple[None, None], tuple[dict[str, Any], pd.DataFrame]]
        Task summary and signal rows from task metrics file if possible, otherwise (None, None).

    """

    file_path = Path(output_dir) / task / TASK_METRICS_FILENAME
    if not file_path.exists():
        print(f"Warning: task {task} was yet evaluated, the corresponding files were not found.")
        return None, None

    df = pd.read_csv(str(file_path))
    df = cast(pd.DataFrame, df)
    if "feature_name" not in df.columns:
        print(f"Warning: {file_path} has no feature_name column. Skipping task {task}.")
        return None, None

    df["feature_name"] = df["feature_name"].astype(str)
    task_rows = df[df["feature_name"] == task]
    if len(task_rows) == 0:
        print(f"Warning: {file_path} has no task row '{task}'. Skipping task.")
        return None, None

    row = task_rows.iloc[0]
    summary: dict[str, Any] = {"task": task}
    for col in TASK_SUMMARY_COLUMNS:
        summary[col] = row[col] if col in row.index else np.nan

    signal_rows = df[df["feature_name"] != task].copy()
    for col in SIGNAL_STD_COLUMNS:
        if col not in signal_rows.columns:
            signal_rows[col] = np.nan

    ordered_signal_cols = (
        ["feature_name"]
        + [c for c in SIGNAL_VALUE_COLUMNS if c in signal_rows.columns]
        + [c for c in SIGNAL_STD_COLUMNS if c in signal_rows.columns]
    )
    signal_rows = signal_rows[ordered_signal_cols]
    signal_rows["task"] = task

    return summary, signal_rows


# ----------------------------------------------------------------------------------------------------------------------
def _extract_task_summary_from_shots_metrics(task: str, output_dir: str | Path) -> dict[str, Any] | None:
    """
    Load one task summary row from the shots metrics CSV file defined in `SHOT_METRICS_FILENAME`.

    Parameters
    ----------
    task : str
        Target task.
    output_dir : Union[str, Path]
        Target output directory.

    Returns
    -------
    Optional[dict[str, Any]]
        Dictionary with task summary from shot metrics file if possible, otherwise None.

    """

    file_path = Path(output_dir) / task / SHOT_METRICS_FILENAME
    if not file_path.exists():
        print(f"Warning: task {task} was yet evaluated, the corresponding files were not found.")
        return None

    df = pd.read_csv(str(file_path))
    df = cast(pd.DataFrame, df)
    if "shot_id" not in df.columns:
        print(f"Warning: data loaded from {file_path} has no shot_id column. Skipping task {task}.")
        return None

    # Rename _mean columns back to bare names, then average across signals per shot.
    df_renamed = df.rename(columns=MEAN_TO_RAW)
    for metric in TASK_METRICS:
        if metric not in df_renamed.columns:
            df_renamed[metric] = np.nan
        df_renamed[metric] = pd.to_numeric(df_renamed[metric], errors="coerce")

    df_task_shots = df_renamed.groupby("shot_id")[list(TASK_METRICS)].mean()

    return {"task": task, **_build_task_summary_row_from_shots(df_task_shots=df_task_shots)}


# ----------------------------------------------------------------------------------------------------------------------
def _extract_task_summary_from_windows_metrics(
    task: str, output_dir: str | Path, signal_std: Mapping[str, Any] | None = None
) -> tuple[None, None] | tuple[dict[str, Any], pd.DataFrame]:
    """
    Return one task summary row along with signal rows from the windows metrics CSV file defined in
    `WINDOW_METRICS_FILENAME`.

    Parameters
    ----------
    task : str
        Target task.
    output_dir : Union[str, Path]
        Target output directory.
    signal_std : Mapping[str, Any] | None
        Loaded reference statistics, or None to leave normalized metrics as NaN.
        Optional. Default: None.

    Returns
    -------
    Union[tuple[None, None], tuple[dict[str, str], pd.DataFrame]]
        Task summary and signal rows from windows metrics file if possible, otherwise (None, None).

    """

    file_path = Path(output_dir) / task / WINDOW_METRICS_FILENAME
    if not file_path.exists():
        print(f"Warning: task {task} was yet evaluated, the corresponding files were not found.")
        return None, None

    df = pd.read_csv(str(file_path))
    df = cast(pd.DataFrame, df)
    df_signals, df_task_shots, _ = _aggregate_windows_metrics(df=df, signal_std=signal_std)
    summary = {"task": task, **_build_task_summary_row_from_shots(df_task_shots=df_task_shots)}
    signal_rows = df_signals.reset_index().assign(task=task)

    return summary, signal_rows


# ----------------------------------------------------------------------------------------------------------------------
def _evaluation_signal_stats(normalization: bool, normalization_path: str | Path | None) -> Mapping[str, Any] | None:
    """
    Load an explicit evaluation reference only when normalization is enabled.

    Parameters
    ----------
    normalization : bool
        Whether normalized metrics are requested.
    normalization_path : str | Path | None
        Reference statistics YAML. Required when normalization is enabled;
        must be None otherwise.

    Returns
    -------
    Mapping[str, Any] | None
        Signal statistics, or None without reading metadata when normalization is disabled.

    Raises
    ------
    TypeError
        If normalization is not a boolean or the loaded statistics are not a mapping.
    ValueError
        If the path is missing when required or provided with normalization disabled.

    """
    if not isinstance(normalization, bool):
        raise TypeError("normalization must be a boolean.")
    if not normalization:
        if normalization_path is not None:
            raise ValueError("normalization_path must be omitted when normalization=False.")
        return None
    if normalization_path is None or not str(normalization_path).strip():
        raise ValueError("normalization=True requires an explicit normalization_path.")
    stats = get_signals_metadata(file_path=str(normalization_path))
    if not isinstance(stats, Mapping):
        raise TypeError("Normalization statistics must be a mapping of signal names to statistics.")
    return stats


# ----------------------------------------------------------------------------------------------------------------------
def aggregate_windows_metrics(
    df: pd.DataFrame,
    *,
    normalization: bool = False,
    normalization_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate window rows into signal-level and shot-level dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with windows metrics.
    normalization : bool
        Whether to compute NRMSE/NMAE in addition to physical-unit RMSE/MAE. Default: False.
    normalization_path : str | Path | None
        Explicit statistics YAML with a finite positive scalar std for every evaluated signal.
        Required when normalization=True; forbidden otherwise. Keep the reference fixed across
        reduced-data runs. These are evaluation references, not model preprocessing statistics.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple of (df_signals, df_task_shots, df_signals_shots) aggregated metrics in pd.DataFrame format.

    """

    signal_std = _evaluation_signal_stats(normalization, normalization_path)
    df_signals, df_task_shots, df_signals_shots = _aggregate_windows_metrics(df, signal_std)
    return df_signals, df_task_shots, df_signals_shots


# ----------------------------------------------------------------------------------------------------------------------
def _aggregate_windows_metrics(
    df: pd.DataFrame, signal_std: Mapping[str, Any] | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate window metrics using an already loaded evaluation reference.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with window metrics.
    signal_std : Mapping[str, Any] | None
        Reference statistics, or None to leave normalized metrics as NaN.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Signal, task-per-shot and signal-per-shot metrics.

    """
    df = df.copy()

    # Be robust to string/object inputs from in-memory paths and CSV readers.
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    df = df.dropna(subset=["RMSE", "MAE"])

    # Compute signal level score within each shot.
    if "RMSE" in df.columns:
        # Convert RMSE -> MSE so the mean across windows is a valid mean-squared-error.
        df["RMSE"] = df["RMSE"] ** 2
    df_signals_shots = df.drop(columns="window_index").groupby(by=["shot_id", "feature_name"]).mean().reset_index()
    if "RMSE" in df_signals_shots.columns:
        # Back to RMSE: sqrt(mean(MSE_w)) gives the correct aggregated RMSE.
        df_signals_shots["RMSE"] = df_signals_shots["RMSE"] ** 0.5

    df_signals_shots["NRMSE"] = np.nan
    df_signals_shots["NMAE"] = np.nan

    # Normalize each signal before aggregating different physical quantities.
    if signal_std is not None:
        for name in df_signals_shots["feature_name"].unique():
            if name not in signal_std or not isinstance(signal_std[name], Mapping) or "std" not in signal_std[name]:
                raise ValueError(f"Missing evaluation standard deviation for signal {name!r}.")
            value = signal_std[name]["std"]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Evaluation standard deviation for {name!r} must be finite and positive.")
        std = df_signals_shots["feature_name"].apply(lambda x: signal_std[x]["std"])
        df_signals_shots["NRMSE"] = df_signals_shots["RMSE"] / std
        df_signals_shots["NMAE"] = df_signals_shots["MAE"] / std

    # Average signals scores across shots.
    n_shots_per_signal = df_signals_shots.groupby("feature_name")["shot_id"].nunique().rename("n_shots")
    df_signals_mean = (
        df_signals_shots.drop(columns="shot_id")
        .groupby(by=["feature_name"])
        .mean()
        .rename(columns=lambda c: f"{c}_mean")
    )
    df_signals_std = (
        df_signals_shots.drop(columns="shot_id")
        .groupby(by=["feature_name"])
        .std(ddof=0)
        .fillna(0.0)
        .rename(columns=lambda c: f"{c}_std_pop")
    )
    df_signals = df_signals_mean.join(df_signals_std).join(n_shots_per_signal)
    if signal_std is None:
        df_signals[["NRMSE_std_pop", "NMAE_std_pop"]] = np.nan

    # Compute task-level score within each shot.
    df_task_shots = df_signals_shots.drop(columns=["feature_name"]).groupby(by=["shot_id"]).mean()

    return df_signals, df_task_shots, df_signals_shots


# ----------------------------------------------------------------------------------------------------------------------
def compute_metrics(
    task: str,
    output_dir: Path | str,
    window_metrics_accumulator: WindowMetricsAccumulator,
    save_windows_metrics: bool = False,
    save_shot_metrics: bool = False,
    save_task_metrics: bool = True,
    *,
    normalization: bool = False,
    normalization_path: str | Path | None = None,
):
    """
    Compute one task metrics from an in-memory window metrics accumulator.

    Parameters
    ----------
    task : str
        Target task.
    output_dir : Union[Path, str]
        Taget output directory.
    window_metrics_accumulator : WindowMetricsAccumulator
        Instance of WindowMetricsAccumulator class used for computation of metrics.
    save_windows_metrics : bool
        If True, save windows metrics as pandas dataframe in the provided `output_dir` directory.
        Optional. Default: False.
    save_shot_metrics : bool
        If True, save shot metrics as pandas dataframe in the provided `output_dir` directory.
        Optional. Default: False.
    save_task_metrics: bool
        If True, save task metrics as pandas dataframe in the provided `output_dir` directory.
        Optional. Default: True.
    normalization : bool
        Whether to compute NRMSE/NMAE in addition to physical-unit RMSE/MAE. Default: False.
    normalization_path : str | Path | None
        Explicit statistics YAML with a finite positive scalar std for every evaluated signal.
        Required when normalization=True; forbidden otherwise. Keep the reference fixed across
        reduced-data runs. These are evaluation references, not model preprocessing statistics.

    Returns
    -------
    pd.DataFrame
        Tasks metrics in pandas dataframe format.

    Raises
    ------
    ValueError
        If `window_metrics_accumulator` is None.
        If `window_metrics_accumulator.task` task does not match compute_metrics `task`.
        If no window metrics are collected for the passed `task`.
        If `window_metrics_accumulator.to_dataframe()` misses required columns.
    TypeError
        If `window_metrics_accumulator` is not a WindowMetricsAccumulator instance.

    """

    if task not in TASKS_CONFIGS_MAP:
        print(f"Warning: task {task} is not known. Available tasks are: {TASKS_CONFIGS_MAP.keys()!s}.")

    output_dir = Path(output_dir)
    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    if window_metrics_accumulator is None:
        raise ValueError("`window_metrics_accumulator` cannot be None in tokamark.evaluator.compute_metrics().")

    if not isinstance(window_metrics_accumulator, WindowMetricsAccumulator):
        raise TypeError("`window_metrics_accumulator` must be a WindowMetricsAccumulator instance.")

    if window_metrics_accumulator.task != str(task):
        raise ValueError(
            f"`window_metrics_accumulator.task` task does not match compute_metrics task: "
            f"{window_metrics_accumulator.task} vs {task}."
        )

    windows_df = window_metrics_accumulator.to_dataframe()
    if len(windows_df) == 0:
        raise ValueError(f"No window metrics were collected for task {task}.")

    missing = [c for c in COLUMNS if c not in windows_df.columns]
    if len(missing) > 0:
        raise ValueError(
            "`window_metrics_accumulator.to_dataframe()` is missing required columns: " + ", ".join(missing)
        )

    df = windows_df[COLUMNS].copy()

    # Compute signal and task level score within each shot.
    df_signals, df_task_shots, df_signals_shots = aggregate_windows_metrics(
        df=df, normalization=normalization, normalization_path=normalization_path
    )
    df_task_metrics = _build_task_metrics_df(task=task, df_signals=df_signals, df_task_shots=df_task_shots)

    if save_windows_metrics:
        df.to_csv(task_output_dir / WINDOW_METRICS_FILENAME, index=False)

    if save_shot_metrics:
        n_windows = (
            df.groupby(["shot_id", "feature_name"], as_index=False)["window_index"]
            .nunique()
            .rename(columns={"window_index": "n_windows"})
        )
        df_shots = df_signals_shots.rename(columns=RAW_TO_MEAN).merge(
            n_windows, on=["shot_id", "feature_name"], how="left"
        )
        df_shots = df_shots[list(SHOT_SIGNAL_COLUMNS)]
        df_shots.to_csv(task_output_dir / SHOT_METRICS_FILENAME, index=False)

    if save_task_metrics:
        df_task_metrics.to_csv(task_output_dir / TASK_METRICS_FILENAME, index=False)

    return df_task_metrics.set_index("feature_name")


# ----------------------------------------------------------------------------------------------------------------------
def compute_summary_metrics(  # NOSONAR - Ignore cognitive complexity
    output_dir: Path | str = ".",
    source: Literal["task_metrics", "windows_metrics", "shots_metrics"] = "task_metrics",
    save_locally: bool = True,
    *,
    normalization: bool = False,
    normalization_path: str | Path | None = None,
):
    """
    Aggregate all tasks into signal/group metrics from task/window/shot files.

    Parameters
    ----------
    output_dir : Union[Path, str]
        Target output directory.
        Optional. Default: ".".
    source : Literal["task_metrics", "windows_metrics", "shots_metrics"]
        Type of metrics to be computed. Valid options: ["task_metrics", "windows_metrics", "shots_metrics"].
        Optional. Default: "task_metrics".
    save_locally: bool
        If True, save metrics as pandas dataframes in the provided `output_dir` directory.
        Optional. Default: True.
    normalization : bool
        Enable normalization when recomputing source="windows_metrics". Default: False.
        Task/shot sources preserve the metrics already present in their files and cannot be renormalized.
    normalization_path : str | Path | None
        Explicit reference YAML, required when normalization=True and forbidden otherwise.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Signals and groups metrics in pandas dataframe format, respectively.

    Raises
    ------
    ValueError
        If `source` not in ["task_metrics", "windows_metrics", "shots_metrics"].

    """

    output_dir = Path(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(name=output_dir, exist_ok=True)

    all_signals = []
    all_groups = []

    if source not in ["task_metrics", "windows_metrics", "shots_metrics"]:
        raise ValueError("`source` must be one of: 'task_metrics', 'windows_metrics', 'shots_metrics'.")

    if source != "windows_metrics" and (normalization or normalization_path is not None):
        raise ValueError("Task/shot files are already aggregated; use source='windows_metrics' to normalize.")
    signal_std = None
    if source == "windows_metrics":
        signal_std = _evaluation_signal_stats(normalization, normalization_path)

    for group_id in GROUP_TASKS:
        task_summaries = []

        for task in GROUP_TASKS[group_id]:
            if source == "windows_metrics":
                task_summary, signal_rows = _extract_task_summary_from_windows_metrics(
                    task=task, output_dir=output_dir, signal_std=signal_std
                )
                if task_summary is None:
                    continue
                task_summaries.append(task_summary)
                if (signal_rows is not None) and (len(signal_rows) > 0):
                    all_signals.append(signal_rows)
            elif source == "shots_metrics":
                task_summary = _extract_task_summary_from_shots_metrics(task=task, output_dir=output_dir)
                if task_summary is None:
                    continue
                task_summaries.append(task_summary)
            else:
                task_summary, signal_rows = _extract_task_summary_from_task_metrics(task=task, output_dir=output_dir)
                if task_summary is None:
                    continue
                task_summaries.append(task_summary)
                if (signal_rows is not None) and (len(signal_rows) > 0):
                    all_signals.append(signal_rows)

        if len(task_summaries) > 0:
            df_tasks = pd.DataFrame(task_summaries)
            df_group_values = _build_group_summary_from_task_summaries(df_tasks=df_tasks)
            df_group_row: dict[str, object] = {"task": f"group_{group_id}"}
            df_group_row.update(df_group_values)

            all_groups.append(pd.DataFrame([df_group_row]))
            all_groups.append(df_tasks)

    df_signals = pd.concat(all_signals, ignore_index=True) if len(all_signals) > 0 else pd.DataFrame()
    if (len(df_signals) > 0) and ("task" in df_signals.columns):
        ordered_cols = ["task"] + [c for c in df_signals.columns if c != "task"]
        df_signals = df_signals[ordered_cols]

    df_groups = pd.concat(all_groups, ignore_index=True) if len(all_groups) > 0 else pd.DataFrame()
    if (len(df_groups) > 0) and ("task" in df_groups.columns):
        ordered_cols = ["task"] + [c for c in df_groups.columns if c != "task"]
        df_groups = df_groups[ordered_cols]

    if save_locally:
        if len(df_signals) > 0:
            df_signals.to_csv(output_dir / SIGNAL_METRICS_FILENAME, index=False)
        if len(df_groups) > 0:
            df_groups.to_csv(output_dir / GROUP_METRICS_FILENAME, index=False)

    return df_signals, df_groups


# ======================================================================================================================
if __name__ == "__main__":
    print("Example use of this module is described below.\n")

    print("First, some preliminary steps:\n")

    print(
        "    output_dir = <output dir>\n"
        "    target_task = <target_task>\n"
        "    wma = WindowMetricsAccumulator(\n"
        "        task=<target_task>\n"
        "    )\n"
    )

    print(
        "From here, the WindowMetricsAccumulator instance needs to be populated for its use to compute metrics."
        "This involves a loop as follows:\n"
    )

    print(
        "    dataloader = <DataLoader instance>\n"
        "    for batch in <dataloader>:\n"
        "        y_true, y_pred, y_mask, shot_ids, window_indices  = <function(batch)>\n"
        "        for <feature_name> in <features>:\n"
        "            wma.add_batch(\n"
        "                y_target=<function(y_true)>,\n"
        "                y_pred=<function(y_pred)>,\n"
        "                shot_ids=<shot_ids[idx]>,\n"
        "                window_indices=<window_indices[idx]>,\n"
        "                feature_name=<feature_name>,\n"
        "            )\n"
    )

    print("After this, the metrics computation can be done as follows:\n")

    print(
        "    compute_metrics(\n"
        "        task=target_task,\n"
        "        window_metrics_accumulator=wma,\n"
        "        output_dir=<output_dir>\n"
        "    )\n"
        "    compute_summary_metrics(output_dir=output_dir_)\n"
    )
