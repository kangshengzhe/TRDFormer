"""
因果 VMD 生成器（消除数据泄漏）
================================
现状：对整个分区一次性做 VMD，每点 IMF 含未来信息 -> 泄漏。
修复：全序列逐点滑窗 VMD——点 i 的 IMF 只用历史窗口 [i-W+1, i]，
      取该段末点的 IMF 值。严格因果，无未来信息。

输出：vmd_imfs_causal.npz，含 all_imfs=(N, K)，与原 vmd_imfs.npz 同布局，
      可直接替换供 experiments/runner.py 使用。
"""
from __future__ import annotations
import argparse, time
from multiprocessing import Pool
import numpy as np
from vmdpy import VMD

# 与 vmd_params.json 完全一致
K, ALPHA, TAU, DC, INIT, TOL = 5, 2000.0, 0.0, 0, 1, 1e-7

_SIGNAL = None
_W = None
_MIN = None


def _init(signal, window, min_hist):
    global _SIGNAL, _W, _MIN
    _SIGNAL, _W, _MIN = signal, window, min_hist


def _causal_imf_at(i: int) -> np.ndarray:
    """点 i 的因果 IMF：用历史窗 [i-W+1, i] 做 VMD，取末点 IMF。"""
    lo = max(0, i - _W + 1)
    seg = _SIGNAL[lo:i + 1]
    if len(seg) < _MIN:
        return np.full(K, seg.mean() / K if len(seg) else 0.0, dtype=np.float64)
    u, _, _ = VMD(seg, ALPHA, TAU, K, DC, INIT, TOL)
    imfs = np.asarray(u)          # (K, len_even)
    return imfs[:, -1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--min-hist", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.imf_in)
    global_imfs = z["all_imfs"]
    signal = global_imfs.sum(axis=1).astype(np.float64)
    N = len(signal)
    n_do = args.limit if args.limit > 0 else N
    print(f"signal length N={N}, computing causal IMF for {n_do} points, "
          f"window={args.window}, workers={args.workers}", flush=True)

    t0 = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(signal, args.window, args.min_hist)) as pool:
        results = pool.map(_causal_imf_at, range(n_do), chunksize=64)
    causal = np.asarray(results, dtype=np.float32)
    dt = time.time() - t0
    print(f"done in {dt:.1f}s  ({dt/n_do*1000:.1f} ms/point)", flush=True)

    if args.limit > 0:
        rel = np.linalg.norm(causal - global_imfs[:n_do], axis=1) / (signal.std() + 1e-9)
        print(f"LIMIT mode (no save). causal-vs-global rel diff: "
              f"mean={rel.mean():.3f} median={np.median(rel):.3f} max={rel.max():.3f}", flush=True)
        return

    np.savez_compressed(args.out, all_imfs=causal)
    print(f"saved causal IMF to {args.out}, shape={causal.shape}", flush=True)


if __name__ == "__main__":
    main()
