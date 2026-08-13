# -*- coding: utf-8 -*-
"""
为多风机泛化验证批量生成统一清洗数据(10 台,含 turb1),放到 data/wind/multiturb/。
统一清洗逻辑(完全可复现,方法节可写):
  1. 从全量抽取该 TurbID,按 (Day, Tmstamp) 时间排序
  2. 选 8 列 [TurbID, Day, Tmstamp, Wspd, Wdir, Etmp, Itmp, Patv]
  3. Wdir = Wdir % 360
  4. physical_rule_clean(enable=True)
  5. 线性插值(默认 forward,不回填边界) + dropna  → 与官方一致地删掉起始全 NaN 行
  6. 保存 data/wind/multiturb/sdwpf_turb{ID}_cleaned_final.csv
主实验用的官方 data/wind/sdwpf_turb1_cleaned_final.csv 不受影响。
"""
import sys, traceback, shutil
from pathlib import Path
REPO = Path(r"C:\Users\kangs\Desktop\windpower_model\iTansformer_LSTM_CA_KAN-master")
sys.path.insert(0, str(REPO))
OUT = REPO / "tools" / "build_multi_turbine.txt"

try:
    import pandas as pd, numpy as np
    from data_pipeline.cleaning import physical_rule_clean

    FULL = r"C:\Users\kangs\Desktop\windpower_model\wtbdata_245days.csv"
    MT_DIR = REPO / "data" / "wind" / "multiturb"
    MT_DIR.mkdir(parents=True, exist_ok=True)
    FEAT = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
    ALL_TURBINES = [1, 2, 13, 55, 70, 83, 86, 88, 94, 99]

    o = []
    def w(s=""): o.append(str(s))

    # 清掉上一版误放到 data/wind/ 根目录的 9 个文件
    for tid in [2, 13, 55, 70, 83, 86, 88, 94, 99]:
        stray = REPO / "data" / "wind" / f"sdwpf_turb{tid}_cleaned_final.csv"
        if stray.exists():
            stray.unlink()
            w("removed stray: %s" % stray.name)

    full = pd.read_csv(FULL)

    def clean_one(tid):
        sub = full[full["TurbID"] == tid].copy()
        ref = pd.Timestamp("2020-01-01")
        ts = pd.to_datetime(
            (ref + pd.to_timedelta(sub["Day"].astype(int) - 1, unit="D")).astype(str)
            + " " + sub["Tmstamp"].astype(str), format="%Y-%m-%d %H:%M")
        sub = sub.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
        keep = sub[["TurbID", "Day", "Tmstamp", "Wspd", "Wdir", "Etmp", "Itmp", "Patv"]].copy()
        keep["Wdir"] = keep["Wdir"] % 360
        cleaned, rep = physical_rule_clean(keep, enable=True)
        cleaned[FEAT] = cleaned[FEAT].interpolate(method="linear")   # 默认 forward,不回填首部
        cleaned = cleaned.dropna(subset=FEAT).reset_index(drop=True)
        # 插值后一致性修正(插值可能重新引入物理不一致的组合)
        cleaned["Patv"] = cleaned["Patv"].clip(lower=0)
        cleaned.loc[(cleaned["Patv"] > 0) & (cleaned["Wspd"] < 3), "Patv"] = 0.0
        return cleaned, rep

    w("\n=== generating unified cleaned data (10 turbines) -> multiturb/ ===")
    stats = []
    for tid in ALL_TURBINES:
        cleaned, rep = clean_one(tid)
        path = MT_DIR / f"sdwpf_turb{tid}_cleaned_final.csv"
        cleaned.to_csv(path, index=False)
        # 校验:无 NaN、无残留物理异常
        assert cleaned[FEAT].isna().sum().sum() == 0, "NaN remains in turb%d" % tid
        assert (cleaned["Patv"] < 0).sum() == 0
        assert ((cleaned["Patv"] > 0) & (cleaned["Wspd"] < 3)).sum() == 0
        stats.append((tid, len(cleaned), cleaned["Patv"].mean(), cleaned["Patv"].std()))
        w("turb%-3d rows=%d  patv_mean=%.1f  patv_std=%.1f  (clip_neg=%d below_cutin=%d)" %
          (tid, len(cleaned), cleaned["Patv"].mean(), cleaned["Patv"].std(),
           rep["n_clipped_negative_patv"], rep["n_marked_below_cutin_with_power"]))

    rows = set(s[1] for s in stats)
    w("\nrow counts across turbines: %s (all equal: %s)" % (sorted(rows), len(rows) == 1))

    # 与官方 turb1 对比(键对齐,正确方式)
    off_path = REPO / "data" / "wind" / "sdwpf_turb1_cleaned_final.csv"
    if off_path.exists():
        off = pd.read_csv(off_path)
        my1 = pd.read_csv(MT_DIR / "sdwpf_turb1_cleaned_final.csv")
        m = pd.merge(my1, off, on=["Day", "Tmstamp"], suffixes=("_my", "_off"))
        w("\n=== unified turb1 vs official (key-aligned, merged rows=%d) ===" % len(m))
        for col in FEAT:
            d = (m[col + "_my"] - m[col + "_off"]).abs()
            w("  %-5s rows_diff>0.01=%5d  max=%.3f  mean=%.5f" %
              (col, int((d > 0.01).sum()), d.max(), d.mean()))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(o))
    print("done")
except Exception:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("ERROR:\n" + traceback.format_exc())
    print("error")
