#!/bin/bash
# ================================================================
# V2 多风机泛化实验
# 10台风机 x [proposed_v2, dlinear, itransformer, patchtst] x 4步长 x 3种子
# = 10 x 4 x 4 x 3 = 480 runs
# ================================================================
cd ~/windpower_model/iTansformer_LSTM_CA_KAN-master

echo "=== V2 Multi-turbine Generalization Experiment ==="
echo "Turbines: 1,2,13,55,70,83,86,88,94,99"
echo "Models: proposed_v2, dlinear, itransformer, patchtst"
echo "Horizons: 1,6,12,24  Seeds: 42,43,44"
echo ""

# Step 1: Generate DWT IMFs for all 10 turbines (per-partition, scaled)
echo "=== Step 1: Generate DWT for all turbines ==="
.venv/bin/python << 'EOF'
import sys, os
sys.path.insert(0, '.')
from pathlib import Path
from data_pipeline.dwt import generate_dwt_imfs
from scripts.preprocess_cli import run_pipeline
import json

TURBINES = [1, 2, 13, 55, 70, 83, 86, 88, 94, 99]
HORIZONS = [1, 6, 12, 24]
LOOKBACK = 144

for tid in TURBINES:
    csv_path = f"data/wind/multiturb/sdwpf_turb{tid}_cleaned_final.csv"
    if not Path(csv_path).exists():
        print(f"  [turb{tid}] SKIP - no data file")
        continue
    
    for h in HORIZONS:
        mdir = Path(f"outputs/multiturb_v2/manifests/turb{tid}/h{h}")
        partition = mdir / f"partition_indices_l{LOOKBACK}_h{h}.json"
        dwt_imf = mdir / "vmd_imfs.npz"  # named vmd_imfs for compatibility with runner
        scaler = mdir / "scaler.pkl"
        
        if partition.exists() and dwt_imf.exists() and scaler.exists():
            continue  # already preprocessed
        
        mdir.mkdir(parents=True, exist_ok=True)
        
        # Run standard pipeline (creates partition + scaler + VMD)
        cfg = {
            "csv_path": csv_path,
            "manifest_dir": str(mdir),
            "lookback": LOOKBACK,
            "horizon": h,
            "vmd_k": 5,
            "vmd_off": True,  # skip VMD, we'll do DWT instead
            "cleaning": True,
            "vmd_alpha": 2000.0, "vmd_tau": 0.0, "vmd_DC": 0, "vmd_init": 1, "vmd_tol": 1e-7,
        }
        try:
            run_pipeline(cfg)
        except Exception as e:
            print(f"  [turb{tid} h{h}] pipeline error: {e}")
            continue
        
        # Now generate DWT on scaled signal
        if partition.exists() and scaler.exists():
            try:
                generate_dwt_imfs(
                    csv_path=csv_path,
                    partition_path=str(partition),
                    output_path=str(dwt_imf),
                    scaler_path=str(scaler),
                    wavelet="db4",
                    max_level=4,
                    target_col="Patv",
                )
            except Exception as e:
                print(f"  [turb{tid} h{h}] DWT error: {e}")
    
    print(f"  [turb{tid}] preprocessed")

print("\nAll turbines preprocessed.")
EOF

echo ""
echo "=== Step 2: Launch V2 multi-turbine experiments (dual GPU) ==="

# Shard 0: turbines 1,2,13,55,70 on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup stdbuf -oL .venv/bin/python -u scripts/run_multiturb_v2.py \
    --turbines 1 2 13 55 70 \
    --epochs 150 \
    > /tmp/multiturb_v2_s0.log 2>&1 &
echo "Shard0 PID: $!"

# Shard 1: turbines 83,86,88,94,99 on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup stdbuf -oL .venv/bin/python -u scripts/run_multiturb_v2.py \
    --turbines 83 86 88 94 99 \
    --epochs 150 \
    > /tmp/multiturb_v2_s1.log 2>&1 &
echo "Shard1 PID: $!"

sleep 5
echo ""
echo "=== Verify ==="
ps aux | grep run_multiturb_v2 | grep -v grep
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "Monitor: tail -f /tmp/multiturb_v2_s0.log"
