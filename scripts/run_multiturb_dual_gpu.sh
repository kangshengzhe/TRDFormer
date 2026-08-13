#!/usr/bin/env bash
# 多风机泛化验证 —— 双卡并行(按风机分片,每卡包干一半风机)
#
# 用法:
#   bash scripts/run_multiturb_dual_gpu.sh [run_multiturb.py 的其余参数...]
#
# 原理:
#   GPU0: CUDA_VISIBLE_DEVICES=0 python -m scripts.run_multiturb --num-shards 2 --shard-index 0
#   GPU1: CUDA_VISIBLE_DEVICES=1 python -m scripts.run_multiturb --num-shards 2 --shard-index 1
#   两进程各处理互不重叠的一半风机(预处理+训练在同进程内完成,无 manifest 写冲突),
#   共同 append 到 outputs/multiturb/outputs/runs/run_records.jsonl(Linux append 原子)。
#
# 日志: logs/mt_gpu0.log 和 logs/mt_gpu1.log

set -e
cd "$(dirname "$0")/.."   # 切到仓库根目录
mkdir -p logs

PY=.venv/bin/python   # 使用仓库自带 venv(含 numpy/torch/vmdpy),而非系统 python3

echo "启动 GPU0 -> logs/mt_gpu0.log"
CUDA_VISIBLE_DEVICES=0 nohup $PY -m scripts.run_multiturb \
    --num-shards 2 --shard-index 0 "$@" \
    > logs/mt_gpu0.log 2>&1 &
PID0=$!
echo "  PID: $PID0"

echo "启动 GPU1 -> logs/mt_gpu1.log"
CUDA_VISIBLE_DEVICES=1 nohup $PY -m scripts.run_multiturb \
    --num-shards 2 --shard-index 1 "$@" \
    > logs/mt_gpu1.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

echo ""
echo "两进程已后台启动。查看进度:"
echo "  tail -f logs/mt_gpu0.log"
echo "  tail -f logs/mt_gpu1.log"
echo "检查是否在跑:  ps -p $PID0 $PID1"
echo "$PID0 $PID1" > logs/mt_pids.txt
