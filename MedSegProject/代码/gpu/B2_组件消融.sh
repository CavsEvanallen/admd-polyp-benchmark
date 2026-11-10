#!/usr/bin/env bash
# B2:ADMD 组件消融(MobileSAM,patience=15,与已有 admd 同设置):无查询调制 / 无对齐 / 无域监督。
# 估时:每个变体 15 个训练约 5 h,共约 16 h(可 SEEDS="0" 先跑一个种子约 5 h)。
set -e; cd "$(dirname "$0")"; source env.sh
for m in noquery noalign nodomsup; do for c in $CENTRES; do for s in $SEEDS; do
  train_one mobilesam $m $c $s 15 "" runs_ablation
  infer_one mobilesam $m $c $s "" runs_ablation results_ablation ""
done; done; done
echo "B2 done. 回传: results_ablation/ , runs_ablation/**/train_log.csv config.json"
