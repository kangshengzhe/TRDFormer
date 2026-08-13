"""
为全部 horizon 生成分区清单。

划分边界（train/valid/test）只取决于数据总行数和比例，与 horizon 无关，
因此可以从已有的任一 partition_indices 复制边界，只改 horizon 字段，
为 {1, 6, 12, 24} 各生成一份清单文件。

这样批量运行时每个 horizon 都能找到对应的 partition_indices_l144_h{H}.json。
"""
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from data_pipeline.manifest import PartitionManifest, PartitionIndices

MANIFEST_DIR = Path("outputs/manifests")
LOOKBACK = 144
HORIZONS = [1, 6, 12, 24]


def main() -> int:
    # 找一份已存在的清单作为边界来源
    existing = sorted(MANIFEST_DIR.glob("partition_indices_l*.json"))
    if not existing:
        print("ERROR: 没有找到任何 partition_indices 清单，请先运行 preprocess_cli.py")
        return 1

    src_path = existing[0]
    src = PartitionManifest.read(str(src_path))
    print(f"以 {src_path.name} 的边界为基准:")
    print(f"  train={src.train}  valid={src.valid}  test={src.test}")

    # n_total_rows = test.end（最后一个分区的结束索引）
    n_total = src.test[1]

    for h in HORIZONS:
        indices = PartitionIndices(
            train=src.train, valid=src.valid, test=src.test,
            lookback=LOOKBACK, horizon=h,
        )
        out = PartitionManifest.write(str(MANIFEST_DIR), indices, n_total)
        print(f"  写入: {Path(out).name}")

    print("全部 horizon 分区清单已生成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
