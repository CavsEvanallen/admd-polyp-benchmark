#!/usr/bin/env bash
# A1i:patience=30 的模型已训练好(runs*/.../seedS_pat30/best.pt),但测试集推理没做(results_pat30 里只有 admd/uniform_ens)。
# 这个脚本只做推理 + 指标,不训练。已存在的结果自动跳过。
#   MobileSAM: sam_ft wide_sam            -> results_pat30/
#   SAM ViT-B: sam_ft wide_sam admd       -> results_sam_b_pat30/   (uniform_ens 的 pat30 没训练,不需要)
# 估时:每个 (方法×折×种子) 约 8 个 推理+指标,MobileSAM 共 30 组、ViT-B 共 45 组;先 DRY=1 看命令,
# 建议先 ENCODERS=mobilesam SEEDS="0" 跑一个种子(几分钟~十几分钟),把 results_pat30 发我看是否正常再全跑。
# 用法: source env.sh; cd "$WORK" 后  ENCODERS=mobilesam bash A1i_pat30仅推理.sh
set -e; cd "$(dirname "$0")"; source env.sh; cd "$WORK"
ENCODERS=${ENCODERS:-"mobilesam sam_b"}
for enc in $ENCODERS; do
  enc_setup $enc
  if [ $enc = mobilesam ]; then METHODS="sam_ft wide_sam admd uniform_ens"; else METHODS="sam_ft wide_sam admd"; fi
  for m in $METHODS; do for c in $CENTRES; do for s in $SEEDS; do
    d=$WORK/runs$SFX/loso/holdout_$c/$m/seed${s}_pat30
    [ -f $d/best.pt ] || { echo "MISSING $d/best.pt"; continue; }
    infer_one $enc $m $c $s _pat30 runs$SFX results${SFX}_pat30 ""
  done; done; done
done
echo "A1i done. 回传: results_pat30/ results_sam_b_pat30/ 里的 metrics.csv gating.csv(用 collect_for_claude.sh 即可)"
