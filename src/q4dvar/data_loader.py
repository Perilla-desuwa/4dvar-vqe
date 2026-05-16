from __future__ import annotations

import csv
from dataclasses import dataclass
from os import PathLike
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class Lorenz96Dataset:
    """Official Lorenz96 CSV data arranged as time-by-dimension arrays."""

    time_steps: NDArray[np.int_]
    truth: Array
    observed: Array
    observed_mask: NDArray[np.bool_]
    source_path: str

    @property
    def n_times(self) -> int:
        return int(self.time_steps.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observed.shape[1])

    @property
    def has_truth(self) -> bool:
        return bool(np.isfinite(self.truth).any())


def load_lorenz96_csv(path: str | PathLike[str], state_dim: int = 40) -> Lorenz96Dataset:
    """Load an official Lorenz96 train/test CSV into dense arrays.

    The competition files are long-form tables with one row per
    ``time_step, dimension`` pair. Missing or blank ``true_value`` entries are
    preserved as NaN so the same loader can be used for hidden test files.
    """

    with open(path, newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        _require_columns(rows.fieldnames, ("time_step", "dimension", "observed_value"))
        raw_records = list(_iter_records(rows, state_dim))
    if not raw_records:
        raise ValueError(f"No Lorenz96 records found in {path}.")

    time_steps = np.asarray(sorted({record.time_step for record in raw_records}), dtype=np.int_)
    time_index = {time_step: index for index, time_step in enumerate(time_steps)}

    truth = np.full((len(time_steps), state_dim), np.nan, dtype=np.float64)
    observed = np.full((len(time_steps), state_dim), np.nan, dtype=np.float64)
    observed_mask = np.zeros((len(time_steps), state_dim), dtype=np.bool_)
    seen: set[tuple[int, int]] = set()

    for record in raw_records:
        key = (record.time_step, record.dimension)
        if key in seen:
            raise ValueError(f"Duplicate row for time_step={record.time_step}, dimension={record.dimension}.")
        seen.add(key)

        row_index = time_index[record.time_step]
        truth[row_index, record.dimension] = record.true_value
        observed[row_index, record.dimension] = record.observed_value
        observed_mask[row_index, record.dimension] = np.isfinite(record.observed_value)

    _validate_complete_dimensions(time_steps, observed, state_dim)

    return Lorenz96Dataset(
        time_steps=time_steps,
        truth=truth,
        observed=observed,
        observed_mask=observed_mask,
        source_path=str(path),
    )


def _require_columns(fieldnames: Iterable[str] | None, required: tuple[str, ...]) -> None:
    available = set(fieldnames or ())
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}.")


@dataclass(frozen=True)
class _Record:
    time_step: int
    dimension: int
    true_value: float
    observed_value: float


def _iter_records(rows: csv.DictReader[str], state_dim: int) -> Iterable[_Record]:
    has_truth = "true_value" in (rows.fieldnames or ())

    for line_number, row in enumerate(rows, start=2):
        time_step = _parse_int(row.get("time_step"), "time_step", line_number)
        dimension = _parse_int(row.get("dimension"), "dimension", line_number)
        if not 0 <= dimension < state_dim:
            raise ValueError(
                f"Invalid dimension at line {line_number}: {dimension}. "
                f"Expected 0 <= dimension < {state_dim}."
            )
        yield _Record(
            time_step=time_step,
            dimension=dimension,
            true_value=_parse_optional_float(row.get("true_value") if has_truth else None),
            observed_value=_parse_optional_float(row.get("observed_value")),
        )


def _parse_int(value: str | None, column: str, line_number: int) -> int:
    if value is None or value == "":
        raise ValueError(f"Missing {column} at line {line_number}.")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {column} at line {line_number}: {value!r}.") from exc


def _parse_optional_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _validate_complete_dimensions(time_steps: NDArray[np.int_], observed: Array, state_dim: int) -> None:
    missing_rows = np.where(np.isnan(observed).all(axis=1))[0]
    if len(missing_rows) > 0:
        missing_times = ", ".join(str(int(time_steps[index])) for index in missing_rows[:5])
        raise ValueError(f"Missing all observed values for time_step(s): {missing_times}.")

    incomplete = np.where(np.isnan(observed).sum(axis=1) > 0)[0]
    if len(incomplete) > 0:
        first = int(incomplete[0])
        missing_dims = np.where(np.isnan(observed[first]))[0]
        dims = ", ".join(str(int(dim)) for dim in missing_dims[:8])
        raise ValueError(
            f"Incomplete observed state at time_step={int(time_steps[first])}; "
            f"missing dimension(s): {dims}. Expected {state_dim} dimensions."
        )
