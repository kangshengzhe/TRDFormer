"""
生成 DWT 分解文件（替代因果 VMD）

用法：
    python scripts/gen_dwt_imfs.py

输出：
    outputs/manifests/dwt_imfs.npz  — shape (N, 5)，与 vmd_imfs.npz 完全同布局
    可直接替换 vmd_imfs.npz 使用（或在 run_batch 中指定 imf_path）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.dwt import generate_dwt_imfs_all_horizons

def main():
    print("=== 生成 DWT 分解文件 ===")
    imfs = generate_dwt_imfs_all_horizons(
        csv_path="data/wind/sdwpf_turb1_cleaned_final.csv",
        manifest_dir="outputs/manifests",
        output_path="outputs/manifests/dwt_imfs.npz",
        wavelet="db4",
        max_level=4,
        target_col="Patv",
    )
    print(f"\n完成！DWT IMF shape: {imfs.shape}")
    print(f"各通道std: {imfs.std(axis=0)}")
    # 原先这里打印的是 abs(imfs.sum(axis=1).max())，即重构信号的最大值
    # （约 1.96），而不是重构误差，看上去像“误差极大”的假警报。真正的
    # 逐分区重构误差由 data_pipeline.dwt 在上面按分区打印（float32 存储
    # 精度下约 1e-7）。
    print("逐分区重构误差见上方 'reconstruction_error='（float32 存储下约 1e-7）")
    print("提示：训练脚本读取的是 outputs/manifests/vmd_imfs.npz，"
          "需将本文件复制过去才会生效：")
    print("  cp outputs/manifests/dwt_imfs.npz outputs/manifests/vmd_imfs.npz")

if __name__ == "__main__":
    main()
