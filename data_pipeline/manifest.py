"""
Manifest IO for the data pipeline.

Provides dataclasses and JSON serialization for:
- PartitionIndices: train/valid/test split boundaries with lookback and horizon
- VMDParams: Variational Mode Decomposition configuration parameters
- PartitionManifest: read/write partition_indices_l{L}_h{H}.json
- VMDManifest: read/write vmd_params.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class PartitionIndices:
    """Integer row boundaries for chronological train/valid/test split.

    Each boundary is a tuple (start, end) representing a half-open interval [start, end).
    """

    train: tuple[int, int]
    valid: tuple[int, int]
    test: tuple[int, int]
    lookback: int
    horizon: int


@dataclass
class VMDParams:
    """Parameters for Variational Mode Decomposition.

    Defaults match the design document specification.
    """

    K: int = 5
    alpha: float = 2000.0
    tau: float = 0.0
    DC: int = 0
    init: int = 1
    tol: float = 1e-7
    max_iter: int = 500


class PartitionManifest:
    """Read/write JSON manifests for partition indices.

    File naming convention: partition_indices_l{lookback}_h{horizon}.json

    JSON schema:
    {
        "lookback": int,
        "horizon": int,
        "n_total_rows": int,
        "train": {"start": int, "end": int},
        "valid": {"start": int, "end": int},
        "test": {"start": int, "end": int}
    }
    """

    @staticmethod
    def filename(lookback: int, horizon: int) -> str:
        """Generate the manifest filename for given lookback and horizon."""
        return f"partition_indices_l{lookback}_h{horizon}.json"

    @staticmethod
    def write(
        path: str | Path,
        indices: PartitionIndices,
        n_total_rows: int,
    ) -> Path:
        """Write partition indices to a JSON manifest file.

        Parameters
        ----------
        path : str or Path
            Directory or full file path. If a directory, the filename is
            generated from lookback and horizon values.
        indices : PartitionIndices
            The partition boundary indices to persist.
        n_total_rows : int
            Total number of rows in the dataset after preprocessing.

        Returns
        -------
        Path
            The path to the written manifest file.
        """
        path = Path(path)
        if path.is_dir() or not path.suffix:
            # path is a directory; generate filename
            os.makedirs(path, exist_ok=True)
            filename = PartitionManifest.filename(indices.lookback, indices.horizon)
            filepath = path / filename
        else:
            os.makedirs(path.parent, exist_ok=True)
            filepath = path

        manifest = {
            "lookback": indices.lookback,
            "horizon": indices.horizon,
            "n_total_rows": n_total_rows,
            "train": {"start": indices.train[0], "end": indices.train[1]},
            "valid": {"start": indices.valid[0], "end": indices.valid[1]},
            "test": {"start": indices.test[0], "end": indices.test[1]},
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return filepath

    @staticmethod
    def read(path: str | Path) -> PartitionIndices:
        """Read partition indices from a JSON manifest file.

        Parameters
        ----------
        path : str or Path
            Full path to the manifest JSON file.

        Returns
        -------
        PartitionIndices
            The deserialized partition indices.

        Raises
        ------
        FileNotFoundError
            If the manifest file does not exist.
        KeyError
            If required fields are missing from the JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Partition manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PartitionIndices(
            train=(data["train"]["start"], data["train"]["end"]),
            valid=(data["valid"]["start"], data["valid"]["end"]),
            test=(data["test"]["start"], data["test"]["end"]),
            lookback=data["lookback"],
            horizon=data["horizon"],
        )


class VMDManifest:
    """Read/write JSON manifests for VMD parameters.

    File naming convention: vmd_params.json

    JSON schema:
    {
        "K": int,
        "alpha": float,
        "tau": float,
        "DC": int,
        "init": int,
        "tol": float,
        "max_iter": int,
        "library": str,
        "fit_seed": int,
        "fit_n_samples": int,
        "imf_shape": [int, int]
    }
    """

    DEFAULT_FILENAME = "vmd_params.json"

    @staticmethod
    def write(
        path: str | Path,
        params: VMDParams,
        *,
        library: str = "vmdpy",
        fit_seed: int = 42,
        fit_n_samples: int = 0,
        imf_shape: tuple[int, int] = (0, 0),
    ) -> Path:
        """Write VMD parameters to a JSON manifest file.

        Parameters
        ----------
        path : str or Path
            Directory or full file path. If a directory, uses default filename.
        params : VMDParams
            The VMD configuration parameters to persist.
        library : str
            The VMD library used (e.g., "vmdpy==0.2").
        fit_seed : int
            The seed used when fitting VMD on the training partition.
        fit_n_samples : int
            Number of training samples used for VMD fitting.
        imf_shape : tuple[int, int]
            Shape of the full IMF array (n_total_samples, K).

        Returns
        -------
        Path
            The path to the written manifest file.
        """
        path = Path(path)
        if path.is_dir() or not path.suffix:
            os.makedirs(path, exist_ok=True)
            filepath = path / VMDManifest.DEFAULT_FILENAME
        else:
            os.makedirs(path.parent, exist_ok=True)
            filepath = path

        manifest = {
            "K": params.K,
            "alpha": params.alpha,
            "tau": params.tau,
            "DC": params.DC,
            "init": params.init,
            "tol": params.tol,
            "max_iter": params.max_iter,
            "library": library,
            "fit_seed": fit_seed,
            "fit_n_samples": fit_n_samples,
            "imf_shape": list(imf_shape),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return filepath

    @staticmethod
    def read(path: str | Path) -> tuple[VMDParams, dict]:
        """Read VMD parameters from a JSON manifest file.

        Parameters
        ----------
        path : str or Path
            Full path to the VMD manifest JSON file.

        Returns
        -------
        tuple[VMDParams, dict]
            A tuple of (VMDParams dataclass, metadata dict) where metadata
            contains library, fit_seed, fit_n_samples, and imf_shape.

        Raises
        ------
        FileNotFoundError
            If the manifest file does not exist.
        KeyError
            If required fields are missing from the JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VMD manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = VMDParams(
            K=data["K"],
            alpha=data["alpha"],
            tau=data["tau"],
            DC=data["DC"],
            init=data["init"],
            tol=data["tol"],
            max_iter=data["max_iter"],
        )

        metadata = {
            "library": data.get("library", "vmdpy"),
            "fit_seed": data.get("fit_seed", 42),
            "fit_n_samples": data.get("fit_n_samples", 0),
            "imf_shape": tuple(data.get("imf_shape", [0, 0])),
        }

        return params, metadata
