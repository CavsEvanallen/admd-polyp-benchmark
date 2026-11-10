#!/usr/bin/env bash
# B3:定性图。对已有 MobileSAM 的 sam_ft / admd / wide_sam(seed0, patience=15)在留出中心上重新推理并保留掩码,
# 再用 pick_examples.py 为每个中心挑 3 张图(中位数、ADMD 最差、适配器最好),把原图/真值/各方法预测/域概率拷到 examples/。
set -e; cd "$(dirname "$0")"; source env.sh; cd "$WORK"
PUBLIC=0
for m in sam_ft admd wide_sam; do for c in $CENTRES; do
  infer_one mobilesam $m $c 0 "" runs results_masks "" 1
done; done
run python pick_examples.py --work $WORK --results results_masks --out $WORK/examples --seed0 0
echo "B3 done. 回传: examples/ (约几十 MB)"
