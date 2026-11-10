#!/usr/bin/env bash
# 重跑完成后,一次性出全部检查与统计。用法: bash run_all_checks.sh <results目录> <输出目录> [manifests目录]
# 任何一步失败(尤其 02 退出码 1)= 数据有问题,先别看 01 的数字。
set -u
R="${1:?results 目录,如 ADMD提交/results/polypgen_loco}"; O="${2:?输出目录}"; M="${3:-}"
mkdir -p "$O"; cd "$(dirname "$0")"
echo "[1/4] ID 碰撞/错配检查"; python3 02_diag_id_collision.py --results "$R" --csv_out "$O/collided.csv" | tee "$O/diag_id_collision.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "!! 发现碰撞,先修数据(见 02_后续执行步骤.md 第 1 步)"; fi
echo "[2/4] 泄漏检查"; python3 03_leakage_check.py --results "$R" ${M:+--manifests "$M"} | tee "$O/leakage_check.txt"
echo "[3/4] 统计"; python3 01_stats_fixed.py --results "$R" --out "$O/stats.md" --ours admd --base sam_ft --drop_collided >/dev/null && echo "  -> $O/stats.md"
echo "[4/4] 门控汇总"; python3 05_gating_summary.py --results "$R" --out "$O/gating_summary.txt" >/dev/null && echo "  -> $O/gating_summary.txt"
