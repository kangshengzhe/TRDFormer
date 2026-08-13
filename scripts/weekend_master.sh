#!/usr/bin/env bash
# ============================================================================
# 周末无人值守主控脚本
# ----------------------------------------------------------------------------
# 阶段 1：等待当前正在跑的 160 个任务（proposed-gated + 消融）完成。
# 阶段 2：聚合出 5 种子版表格快照（即使后续阶段失败，周一也有可用结果）。
# 阶段 3：双卡补跑 seeds 47-51，把全矩阵从 5 种子扩到 10 种子，增强显著性统计。
# 阶段 4：聚合出 10 种子最终表格。
#
# 全程写日志到 logs/weekend.log；每一步都是断点续跑安全、非破坏性的。
# 通过 nohup 启动，SSH 断开后继续运行。
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/weekend.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "================ 周末主控启动 ================"

# ── 阶段 1：等待当前批次结束 ────────────────────────────────────────────────
if [ -f logs/pids.txt ]; then
    PIDS=$(cat logs/pids.txt)
    log "阶段1：等待当前批次 PID: $PIDS"
    for pid in $PIDS; do
        while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    done
fi
log "阶段1完成：当前 160 个任务已结束"

# ── 阶段 2：5 种子表格快照 ─────────────────────────────────────────────────
log "阶段2：聚合 5 种子表格快照"
python3 -m scripts.aggregate_cli >> "$LOG" 2>&1 && log "阶段2：聚合成功" || log "阶段2：聚合失败(不影响后续)"
rm -rf outputs/tables_5seed_snapshot
cp -r outputs/tables outputs/tables_5seed_snapshot 2>/dev/null && log "阶段2：已保存 5 种子快照到 outputs/tables_5seed_snapshot"

# ── 阶段 3：双卡补跑 seeds 47-51（全矩阵，10 种子） ────────────────────────
log "阶段3：启动双卡补跑 seeds 47-51（全 21 模型 × 4 步长 × 5 新种子 = 420 组）"
CUDA_VISIBLE_DEVICES=0 nohup python3 -m scripts.run_batch \
    --num-shards 2 --shard-index 0 --seeds 47 48 49 50 51 \
    > logs/gpu0_seeds.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 nohup python3 -m scripts.run_batch \
    --num-shards 2 --shard-index 1 --seeds 47 48 49 50 51 \
    > logs/gpu1_seeds.log 2>&1 &
P1=$!
log "阶段3：GPU0 PID=$P0  GPU1 PID=$P1"
echo "$P0 $P1" > logs/seeds_pids.txt
wait $P0 $P1
log "阶段3完成：补种子结束"

# ── 阶段 4：10 种子最终表格 ───────────────────────────────────────────────
log "阶段4：聚合 10 种子最终表格"
python3 -m scripts.aggregate_cli >> "$LOG" 2>&1 && log "阶段4：聚合成功" || log "阶段4：聚合失败"

log "================ 周末主控全部完成 ================"
