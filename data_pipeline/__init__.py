"""
Data pipeline module for wind power forecasting experiments.

Provides:
- PartitionIndices: dataclass defining train/valid/test split boundaries
- VMDParams: dataclass defining Variational Mode Decomposition parameters
- PartitionManifest: JSON manifest IO for partition indices
- VMDManifest: JSON manifest IO for VMD parameters
- physical_rule_clean: physical-rule outlier cleaning function
- chronological_split: deterministic chronological train/valid/test split
- persist_partition_indices: persist split boundaries to JSON manifest
- FeatureScaler: StandardScaler wrapper fitted only on training data
- fit_vmd_on_train: run VMD on training partition target signal
- apply_vmd_to_partition: apply VMD with same parameters to val/test partitions
- persist_vmd_params: persist VMD parameters to vmd_params.json via manifest
- WindowedSeriesDataset: sliding-window torch Dataset (lookback, F_in) / (horizon,)
"""

from data_pipeline.manifest import (
    PartitionIndices,
    VMDParams,
    PartitionManifest,
    VMDManifest,
)
from data_pipeline.cleaning import physical_rule_clean
from data_pipeline.splits import chronological_split, persist_partition_indices
from data_pipeline.scaling import FeatureScaler
from data_pipeline.vmd import (
    fit_vmd_on_train,
    apply_vmd_to_partition,
    persist_vmd_params,
)
from data_pipeline.windowing import WindowedSeriesDataset

__all__ = [
    "PartitionIndices",
    "VMDParams",
    "PartitionManifest",
    "VMDManifest",
    "physical_rule_clean",
    "chronological_split",
    "persist_partition_indices",
    "FeatureScaler",
    "fit_vmd_on_train",
    "apply_vmd_to_partition",
    "persist_vmd_params",
    "WindowedSeriesDataset",
]
