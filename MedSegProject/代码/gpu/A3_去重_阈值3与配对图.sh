#!/usr/bin/env bash
# A3:训练池 vs 测试集去重,汉明距离 <=3,并导出前 6 对匹配图。
set -e; cd "$(dirname "$0")"; source env.sh
TRAIN="$(ls -d $WORK/data/PolypGen/data_C*/images_C* | tr '\n' ' ')"
TEST="$WORK/data/CVC-ColonDB/images $WORK/data/ETIS-LaribPolypDB/images $WORK/data/CVC-300/images"
run python $SCRIPTS/06_dedup_check.py --train_dirs $TRAIN --test_dirs $TEST --out $WORK/out_check/dedup_report_thr3.csv --thr 3
run python make_dup_montage.py --report $WORK/out_check/dedup_report_thr3.csv --work $WORK --out $WORK/out_check/dup_montage.png --n 6
echo "A3 done. 回传: out_check/dedup_report_thr3.csv, out_check/dup_montage.png"
echo "注意:--test_dirs 的路径要与你们实际的数据目录一致(报告里的路径形如 data/CVC-ColonDB/images/16.png)。"
