"""核对论文正文中的每个关键数字与实验记录是否一致。

为什么需要这个：正文经过多轮压缩重写，任何一次编辑都可能让某个数字与它
所依据的 run_records.jsonl 脱钩。审稿人复现时对不上，代价远大于现在花几
秒核对。

逐项从 outputs/ 重算，与硬编码在 main.tex 里的值比对，超出容差即报错。
探针与线性天花板的数字来自训练主机上的一次性诊断脚本（_leak_probe.py /
_linear_ceiling.py），无法在本地重算，因此只做"论文里确实写着这个值"的
存在性检查，并标注来源。

用法：
    python tools/verify_manuscript_numbers.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

TEX = pathlib.Path("manuscript/main.tex")

# 变体 -> (run 目录, 论文中的 MAE, 论文中的 std)
RUNS = {
    "offline (proposed)":  ("outputs/v2_full",           58.18, 1.13),
    "lag k=3":             ("outputs/lag3_dwt_h12",       71.58, 1.31),
    "lag k=12":            ("outputs/lag12_dwt_h12",      81.94, 3.12),
    "atrous causal":       ("outputs/atrous_h12",         91.10, 4.37),
    "atrous rescaled":     ("outputs/atrous_norm_h12",     91.54, 3.95),
    "trailing symmetric":  ("outputs/causal_dwt_h12",     96.52, 5.72),
    "trailing reflect":    ("outputs/causal_reflect_h12", 97.87, 6.95),
}
# ablation_v2 里按 run_id 子串筛选
ABLATION = {
    "w/o DWT":        ("no_dwt",       81.95, 2.11),
    "w/o trend":      ("no_trend",     76.24, 1.20),
    "w/o iTrans":     ("no_itrans",    74.62, 0.85),
    "w/o LSTM":       ("no_lstm",      59.31, 1.16),
    "fusion cross":   ("fusion_cross", 63.65, 3.24),
    "head linear":    ("head_linear",  58.63, 1.53),
    "head MLP":       ("head_mlp",     57.03, 1.23),
}
# 只做存在性检查的数字（来自训练主机上的诊断脚本）
DIAGNOSTIC = [
    ("probe offline lead-1 gain",     "16.61"),
    ("probe causal atrous lead-1",    "1.50"),
    ("probe causal trailing lead-1",  "0.68"),
    ("probe offline lead-6 gain",     "38.77"),
    ("probe offline lead-12 gain",    "32.39"),
    ("probe persistence lead-1",      "40.52"),
    ("probe lag3 lead-1 penalty",     "14.43"),
    ("ridge ceiling with bands",      "58.06"),
    ("ridge ceiling without bands",   "106.09"),
    ("ridge encoder worth",           "24"),
    # 2026-08: even-parity 值曾误写为 0.71。实测 0.7046，两位小数应为 0.70。
    # 本列表只做"论文里是否出现该字符串"的存在性检查，所以写错了也会通过 ——
    # 这类舍入错误要靠 tools/probe_perturbation.py 的实测输出来核对。
    ("perturbation D1 odd",           "2.29"),
    ("perturbation D1 even",          "0.70"),
    ("additivity of atrous",          r"9\\times10\^\{-16\}"),
    ("additivity under perturbation", r"8\\times10\^\{-16\}"),
    ("survey total",                  "50"),
    ("survey with decomposition",     "44"),
    ("survey raising leakage",        "17"),
    ("total runs",                    r"1\{,\}108"),
]

TOL_MEAN, TOL_STD = 0.02, 0.03


def mae_from(run_dir: str, id_filter: str | None = None):
    p = pathlib.Path(run_dir) / "outputs" / "runs" / "run_records.jsonl"
    if not p.exists():
        return None
    vals = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "success" or r.get("horizon") != 12:
            continue
        if id_filter and id_filter not in r.get("run_id", ""):
            continue
        m = r.get("metrics") or {}
        if m.get("mae") is not None:
            vals.append(float(m["mae"]))
    if not vals:
        return None
    a = np.asarray(vals)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    bad = 0

    print("=== 因果化变体 (h=12) ===")
    for label, (d, want_m, want_s) in RUNS.items():
        got = mae_from(d)
        if got is None:
            print(f"  ?? {label:22s} 无记录 ({d})")
            bad += 1
            continue
        m, s, n = got
        ok_m = abs(m - want_m) <= TOL_MEAN
        ok_s = abs(s - want_s) <= TOL_STD
        flag = "OK " if (ok_m and ok_s) else "BAD"
        if not (ok_m and ok_s):
            bad += 1
        print(f"  {flag} {label:22s} 实测 {m:6.2f}+-{s:4.2f} (n={n})  "
              f"论文 {want_m:6.2f}+-{want_s:4.2f}")

    print("\n=== 消融变体 (h=12) ===")
    for label, (idf, want_m, want_s) in ABLATION.items():
        got = mae_from("outputs/ablation_v2", idf)
        if got is None:
            print(f"  ?? {label:22s} 无记录")
            bad += 1
            continue
        m, s, n = got
        ok = abs(m - want_m) <= TOL_MEAN and abs(s - want_s) <= TOL_STD
        if not ok:
            bad += 1
        print(f"  {'OK ' if ok else 'BAD'} {label:22s} 实测 {m:6.2f}+-{s:4.2f} "
              f"(n={n})  论文 {want_m:6.2f}+-{want_s:4.2f}")

    print("\n=== 诊断数字：检查论文中确实写着（无法本地重算）===")
    for label, pat in DIAGNOSTIC:
        hit = re.search(pat, tex) is not None
        if not hit:
            bad += 1
        print(f"  {'OK ' if hit else 'BAD'} {label:32s} /{pat}/")

    print("\n=== 一致性断言 ===")
    lag12 = mae_from("outputs/lag12_dwt_h12")
    nodwt = mae_from("outputs/ablation_v2", "no_dwt")
    if lag12 and nodwt:
        gap = abs(lag12[0] - nodwt[0])
        ok = gap <= 0.02
        if not ok:
            bad += 1
        print(f"  {'OK ' if ok else 'BAD'} lag12 == w/o DWT: "
              f"{lag12[0]:.2f} vs {nodwt[0]:.2f}, 差 {gap:.4f} kW "
              f"(论文声称 0.01)")

    print(f"\n{'全部通过' if bad == 0 else f'{bad} 项不一致'}")
    return 1 if bad else 0


if __name__ == "__main__":
    import os

    os.chdir(pathlib.Path(__file__).resolve().parents[1])
    raise SystemExit(main())
