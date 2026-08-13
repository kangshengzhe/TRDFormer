#!/bin/bash
# ================================================================
# V2 模型实验（DWT + 趋势残差学习）
# 前置：需要 DWT 实验跑完、确认 dwt_imfs.npz 已替换为 vmd_imfs.npz
# ================================================================
cd ~/windpower_model/iTansformer_LSTM_CA_KAN-master

echo "=== 确认 DWT IMFs 已就位 ==="
.venv/bin/python -c "
import numpy as np
d = np.load('outputs/manifests/vmd_imfs.npz')
print('IMF key:', list(d.files), 'shape:', d['all_imfs'].shape)
print('std per channel:', d['all_imfs'].std(axis=0))
# DWT版应该有[~50, ~58, ~77, ~100, ~503]的std
"

echo ""
echo "=== 启动 V2 实验（双卡）==="
# V2 只跑 proposed_v2 模型，因为 ablation 版仍沿用 V1 的结果
MODELS="proposed_v2"
HORIZONS="1 6 12 24"
SEEDS="42 43 44 45 46 47 48 49 50 51"
EPOCHS=150
OUTDIR=outputs/v2_full

mkdir -p $OUTDIR

# Shard 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup stdbuf -oL .venv/bin/python -u scripts/run_batch_v2.py \
    --horizons $HORIZONS \
    --seeds $SEEDS \
    --epochs $EPOCHS \
    --out-dir $OUTDIR \
    --num-shards 2 --shard-index 0 \
    > /tmp/v2_s0.log 2>&1 &
echo "Shard0 PID: $!"

# Shard 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup stdbuf -oL .venv/bin/python -u scripts/run_batch_v2.py \
    --horizons $HORIZONS \
    --seeds $SEEDS \
    --epochs $EPOCHS \
    --out-dir $OUTDIR \
    --num-shards 2 --shard-index 1 \
    > /tmp/v2_s1.log 2>&1 &
echo "Shard1 PID: $!"

sleep 5
echo ""
echo "=== 进程确认 ==="
ps aux | grep run_batch_v2 | grep -v grep
echo ""
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
