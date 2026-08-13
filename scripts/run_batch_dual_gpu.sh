#!/usr/bin/env bash
# 双卡并行批量运行器 —— 同时用 2 张 GPU 各跑一半任务
#
# 用法:
#   bash scripts/run_batch_dual_gpu.sh [run_batch.py 的其余参数...]
#
# 示例:
#   bash scripts/run_batch_dual_gpu.sh --models proposed lstm transformer informer \
#       fedformer dlinear patchtst itransformer timesnet autoformer \
#       nonstationary_transformer timexer --horizons 6
#
# 原理:
#   GPU0 进程: CUDA_VISIBLE_DEVICES=0 python -m scripts.run_batch --num-shards 2 --shard-index 0 ...
#   GPU1 进程: CUDA_VISIBLE_DEVICES=1 python -m scripts.run_batch --num-shards 2 --shard-index 1 ...
#   两个进程各跑互不重叠的一半任务，同时写入同一个 run_records.jsonl（Linux 上小体积
#   append 写是原子的，不会互相覆盖）。
#
# 日志分别写到 logs/gpu0.log 和 logs/gpu1.log，方便分别查看进度。

set -e
cd "$(dirname "$0")/.."   # 切到仓库根目录

mkdir -p logs

echo "启动 GPU0 进程 -> logs/gpu0.log"
CUDA_VISIBLE_DEVICES=0 nohup python3 -m scripts.run_batch \
    --num-shards 2 --shard-index 0 "$@" \
    > logs/gpu0.log 2>&1 &
PID0=$!
echo "  PID: $PID0"

echo "启动 GPU1 进程 -> logs/gpu1.log"
CUDA_VISIBLE_DEVICES=1 nohup python3 -m scripts.run_batch \
    --num-shards 2 --shard-index 1 "$@" \
    > logs/gpu1.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

echo ""
echo "两个进程已在后台启动。"
echo "查看进度:"
echo "  tail -f logs/gpu0.log"
echo "  tail -f logs/gpu1.log"
echo ""
echo "检查是否还在运行:"
echo "  ps -p $PID0 $PID1"
echo ""
echo "PID 已写入 logs/pids.txt，方便后续管理"
echo "$PID0 $PID1" > logs/pids.txt
