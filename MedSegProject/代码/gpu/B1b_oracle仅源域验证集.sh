#!/usr/bin/env bash
# B1b:正确的 oracle 路由。上一版把"留出中心"的 domain_id 设成 0..4(C1→0,…),但留出中心不是任何一个训练域,
# 这个标签只是指向一个任意分支(oracle 在 C2 上与 hard 的数字逐位相同,说明就是固定选了一个分支),所以结果不能用。
# oracle 只在源域图像上有意义:这里用 val.csv(早停验证集,每张图的 domain_id 是真实的训练域编号)。
# 对 soft / hard / uniform / random / oracle 五种路由在 val 上推理,同一批图、同一模型。只推理,不重训。
set -e; cd "$(dirname "$0")"; source env.sh; cd "$WORK"
enc_setup mobilesam
for c in $CENTRES; do for s in $SEEDS; do
  run_dir=$WORK/runs/loso/holdout_$c/admd/seed$s
  [ -f $run_dir/best.pt ] || { echo "MISSING $run_dir/best.pt"; continue; }
  vp=$PROMPTS/holdout_$c/seed$s/val.json
  [ -f $vp ] || run python $SCRIPTS/03_prompts.py --manifest $WORK/manifests/loso/holdout_$c/seed$s/val.csv --out $vp --seed $s
  for f in soft hard uniform random oracle; do for p in 1pt bbox; do
    o=$WORK/results_oracle_val/loso/holdout_$c/admd_$f/Source_val/$p/seed$s
    [ -f $o/metrics.csv ] && continue
    run python $PLUGIN/scripts/infer.py --run $run_dir --manifest $WORK/manifests/loso/holdout_$c/seed$s/val.csv \
        --prompts $vp --prompt $p --cache_dir $CACHE --ckpt $CKPT --out $o --fusion $f
    run python $SCRIPTS/04_metrics.py --manifest $WORK/manifests/loso/holdout_$c/seed$s/val.csv --pred_dir $o/masks --out $o/metrics.csv
    [ "$DRY" = 1 ] || rm -rf $o/masks
  done; done
done; done
echo "B1b done. 回传: results_oracle_val/(metrics.csv gating.csv)"
echo "检查:val.csv 的 domain_id 应是 0..4 且每个训练域都出现;若全是 -1,oracle 会报错或无意义,请先告诉我。"
