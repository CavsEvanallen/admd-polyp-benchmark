#!/usr/bin/env bash
# 把"我需要的原始结果"一次性打成一个小 zip(约几十 MB 以内),不含权重/掩码/cache。
# 用法: cd 到 $WORK(含 results*/ runs*/ manifests/ prompts/ 的目录),然后
#   bash collect_for_claude.sh            # 生成 ../ADMD_metrics_20261106.zip 和 inventory.txt
set -euo pipefail
OUT=${OUT:-../ADMD_metrics_20261106.zip}
INV=inventory.txt
{
  echo "# 生成时间 $(date -Is)"; echo "# 目录: $(pwd)"
  for r in results*/; do
    echo "== $r  metrics.csv 数: $(find "$r" -name metrics.csv | wc -l)  (空文件: $(find "$r" -name metrics.csv -size -2c | wc -l))"
    find "$r" -mindepth 3 -maxdepth 3 -type d | sed 's#^[^/]*/loso/holdout_[^/]*/##' | sort -u | head -20
  done
  echo; echo "== runs*: best.pt 数"; for r in runs*/; do echo "$r $(find "$r" -name best.pt | wc -l)"; done
} > "$INV" 2>&1
rm -f "$OUT"
# 1. 每张图的指标 + 门控(小文件)
find results* -name metrics.csv -o -name gating.csv | zip -q "$OUT" -@
# 2. 训练日志与配置(用于表 4 的轮数、成本)
find runs* \( -name train_log.csv -o -name config.json \) | zip -q "$OUT" -@
# 3. 已有的统计/计时文件
find . -maxdepth 3 \( -name 'timing*.csv' -o -name 'profile_cost*.csv' -o -name 'dedup_report*' \) -not -path './cache*' | zip -q "$OUT" -@ 2>/dev/null || true
# 4. 代码:训练/推理/计时/指标脚本(只要 .py 和 .sh,排除环境目录)
find . -maxdepth 4 \( -name '*.py' -o -name '*.sh' \) -not -path '*/site-packages/*' -not -path '*/venv/*' -not -path '*/.git/*' | zip -q "$OUT" -@
# 5. 提示文件各取 1 个样例(看格式即可)
P=$(find prompts -name 'test_*.json' | head -1 || true); [ -n "$P" ] && head -c 3000 "$P" > prompt_sample.txt && zip -q "$OUT" prompt_sample.txt || true
zip -q "$OUT" "$INV"
ls -la "$OUT"; echo "检查: unzip -l $OUT | tail -3"; echo "inventory.txt 里的 metrics.csv 数与你预期是否一致?"
