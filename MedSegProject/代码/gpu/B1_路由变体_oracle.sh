#!/usr/bin/env bash
# B1:用已训练好的 ADMD(best.pt)做推理时路由变体:soft / hard / uniform / random(留出中心与公共集)
#     和 oracle(只能在源域图像上,用 val.csv 的真域标签)。只推理,不重训。
# 前提:runs/loso/holdout_Ck/admd/seedS/best.pt 还在。若不在,先问我,要重训(MobileSAM,5 折 × 1 种子即可)。
set -e; cd "$(dirname "$0")"; source env.sh
enc_setup mobilesam
SEEDS_B1=${SEEDS_B1:-"0 1 2"}
for c in $CENTRES; do for s in $SEEDS_B1; do
  run_dir=$WORK/runs/loso/holdout_$c/admd/seed$s
  [ -f $run_dir/best.pt ] || { echo "MISSING $run_dir/best.pt"; continue; }
  # 1) 源域验证图的提示(用于 oracle)
  vp=$PROMPTS/holdout_$c/seed$s/val.json
  [ -f $vp ] || run python $SCRIPTS/03_prompts.py --manifest $WORK/manifests/loso/holdout_$c/seed$s/val.csv --out $vp --seed $s
  for f in soft hard uniform random oracle; do for p in 1pt bbox; do
    o=$WORK/results_routing/loso/holdout_$c/admd_$f/Source_val/$p/seed$s
    [ -f $o/metrics.csv ] && continue
    run python $PLUGIN/scripts/infer.py --run $run_dir --manifest $WORK/manifests/loso/holdout_$c/seed$s/val.csv \
        --prompts $vp --prompt $p --cache_dir $CACHE --ckpt $CKPT --out $o --fusion $f
    run python $SCRIPTS/04_metrics.py --manifest $WORK/manifests/loso/holdout_$c/seed$s/val.csv --pred_dir $o/masks --out $o/metrics.csv
    [ "$DRY" = 1 ] || rm -rf $o/masks
  done; done
  # 2) 留出中心 + 公共集:soft/hard/uniform/random(oracle 无真域标签,不可用)
  for f in soft hard uniform random; do
    infer_one mobilesam admd $c $s "" runs results_routing_$f $f
  done
done; done
echo "B1 done. 回传: results_routing/ 与 results_routing_*/ (metrics.csv gating.csv infer_config.json)"
echo "注意:val 是早停用的验证集,oracle 与 soft 在 val 上的对比只用来回答「真域标签有没有用」,论文里会如实标注。"
