# -*- coding: utf-8 -*-
"""134 台风机画像:出力/波动/缺失/异常,用于分层挑选代表风机。纯向量化。"""
import traceback
OUT = r"C:\Users\kangs\Desktop\windpower_model\iTansformer_LSTM_CA_KAN-master\tools\turbine_profile.txt"
try:
    import pandas as pd, numpy as np
    FULL = r"C:\Users\kangs\Desktop\windpower_model\wtbdata_245days.csv"
    df = pd.read_csv(FULL, usecols=["TurbID", "Wspd", "Patv"])

    # 预先建异常标记列(向量化,避免 groupby.apply)
    df["_neg"]   = (df["Patv"] < 0)
    df["_stall"] = (df["Patv"] <= 0) & (df["Wspd"] >= 3)
    df["_pmiss"] = df["Patv"].isna()
    df["_wmiss"] = df["Wspd"].isna()

    g = df.groupby("TurbID")
    rep = pd.DataFrame({
        "n":            g.size(),
        "patv_missing": g["_pmiss"].mean(),
        "wspd_missing": g["_wmiss"].mean(),
        "patv_mean":    g["Patv"].mean(),
        "patv_std":     g["Patv"].std(),
        "neg_rate":     g["_neg"].mean(),
        "stall_rate":   g["_stall"].mean(),
    })
    rep["cv"] = rep["patv_std"] / rep["patv_mean"].replace(0, np.nan)

    out = []
    def w(s=""): out.append(str(s))
    w("=== 134-turbine profile (summary) ===")
    w(rep.describe().round(3).to_string())
    w("\n=== worst 15 by patv_missing ===")
    w(rep.sort_values("patv_missing", ascending=False).head(15)[
        ["patv_missing","wspd_missing","stall_rate","patv_mean"]].round(3).to_string())

    clean = rep[(rep["patv_missing"] < 0.05) & (rep["stall_rate"] < 0.10)].copy()
    w("\n=== clean pool (patv_missing<5%% & stall_rate<10%%): %d turbines ===" % len(clean))

    clean["out_tier"] = pd.qcut(clean["patv_mean"], 3, labels=["low","mid","high"])
    w("\n=== stratified representative candidates ===")
    picks = []
    for tier in ["low","mid","high"]:
        sub = clean[clean["out_tier"] == tier].sort_values("cv")
        if len(sub) >= 3:
            idx = [int(sub.index[0]), int(sub.index[len(sub)//2]), int(sub.index[-1])]
        else:
            idx = [int(i) for i in sub.index]
        picks += idx
        w("tier=%s (n=%d): low-cv=%d mid-cv=%d high-cv=%d  (patv_mean %.0f..%.0f)" %
          (tier, len(sub), idx[0], idx[1], idx[-1], sub["patv_mean"].min(), sub["patv_mean"].max()))
    w("\nSUGGESTED PICKS (%d): %s" % (len(picks), sorted(set(picks))))

    rep.round(4).to_csv(r"C:\Users\kangs\Desktop\windpower_model\iTansformer_LSTM_CA_KAN-master\tools\turbine_profile.csv")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("done")
except Exception:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("ERROR:\n" + traceback.format_exc())
    print("error captured")
