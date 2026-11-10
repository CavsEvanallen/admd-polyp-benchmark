#!/usr/bin/env bash
# A1:所有解码器统一 patience=30,MobileSAM 与 SAM ViT-B。已存在的结果会跳过。
# 估时(RTX 4090,按你们 train_log):MobileSAM sam_ft+wide 约 8 h;ViT-B 四个解码器全 3 种子约 32 h。
# 建议先跑:  ENCODERS=mobilesam bash A1_patience30_对称重跑.sh      (约 8 h)
#           再跑:  ENCODERS=sam_b SEEDS="0" ...  看一个种子,最后补全。
# 用法: source env.sh 后  bash A1_patience30_对称重跑.sh      (DRY=1 先检查路径)
set -e; cd "$(dirname "$0")"; source env.sh
ENCODERS=${ENCODERS:-"mobilesam sam_b"}
for enc in $ENCODERS; do
  enc_setup $enc
  if [ $enc = mobilesam ]; then METHODS="sam_ft wide_sam"       # admd / uniform_ens 的 MobileSAM p30 已有
  else METHODS="sam_ft wide_sam admd uniform_ens"; fi
  for m in $METHODS; do for c in $CENTRES; do for s in $SEEDS; do
    train_one $enc $m $c $s 30 _pat30 runs$SFX
    infer_one $enc $m $c $s _pat30 runs$SFX results${SFX}_pat30 ""
  done; done; done
done
echo "A1 done. 回传: results_pat30/ (只要 metrics.csv gating.csv timing.csv infer_config.json), results_sam_b_pat30/, runs*/**/train_log.csv config.json"
