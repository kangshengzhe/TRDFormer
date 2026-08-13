"""Run record dataclass and append-only JSONL writer.

Each completed (or failed/deferred) experiment run produces a RunRecord that is
serialized as a single JSON line and appended to `run_records.jsonl`.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from reproducibility.environment import EnvSnapshot


@dataclass
class RunRecord:
    """Complete record of a single experiment run.

    All fields are populated by the experiment runner upon completion (or
    failure/deferral) and persisted to the JSONL log.
    """

    run_id: str
    model_name: str
    horizon: int
    lookback: int
    seed: int
    metrics: dict  # {'mae', 'rmse', 'r2', 'mbe', 'smape'} in kW
    val_metrics: dict
    train_losses: str  # path to .npz file
    checkpoint: str  # path to .pt file
    config_yaml: str  # path to merged config snapshot
    partition_path: str  # path to partition_indices.json
    scaler_path: str
    vmd_params_path: Optional[str]
    env: EnvSnapshot
    started_at: str  # ISO 8601 timestamp
    finished_at: str  # ISO 8601 timestamp
    wall_clock_seconds: float
    status: str  # 'success' | 'failed' | 'deferred'
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        d = asdict(self)
        # EnvSnapshot is already converted by asdict, but ensure it's a dict
        if isinstance(d.get("env"), EnvSnapshot):
            d["env"] = d["env"].to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        """Reconstruct a RunRecord from a dictionary (e.g., parsed JSON line)."""
        env_data = d.pop("env", {})
        env = EnvSnapshot.from_dict(env_data) if env_data else None
        return cls(env=env, **d)

    def to_json_line(self) -> str:
        """Serialize to a single JSON line (no trailing newline)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def append_run_record(record: RunRecord, path: str = "outputs/runs/run_records.jsonl") -> None:
    """Append a RunRecord as a single JSON line to the specified JSONL file.

    Creates the file and any parent directories if they do not exist.
    Uses append mode to ensure existing records are never overwritten.

    Args:
        record: The RunRecord to persist.
        path: File path for the JSONL log. Defaults to
            'outputs/runs/run_records.jsonl'.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(record.to_json_line() + "\n")


def load_run_records(path: str = "outputs/runs/run_records.jsonl") -> list[RunRecord]:
    """Load all RunRecords from a JSONL file.

    Args:
        path: File path for the JSONL log.

    Returns:
        List of RunRecord instances in file order.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Run records file not found: {path}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                records.append(RunRecord.from_dict(d))
    return records
