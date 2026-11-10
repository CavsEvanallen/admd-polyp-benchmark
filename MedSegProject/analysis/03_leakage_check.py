#!/usr/bin/env python3
"""
留一中心划分的泄漏检查。只读 results/(留出集的图像名),不需要原始数据。

做两件事:
  1. 按文件名风格推断每个留出集的"来源"(HMUH / UUID / 100H / ck…)和"分组"(同一次检查)。
     若不同折的留出集来源相同(如 C1、C2 都是 HMUH),说明这两折其实是同一个中心被拆成两半 ->
     留出 C1 时 C2 仍在训练集里,不是真正的"留出中心"。
  2. 若两折的留出集共享分组(同一次检查的相邻帧分在两折),直接证明帧级泄漏。
若有 manifests(--manifests 目录,含各折 train.csv/val.csv/test.csv),再检查训练/测试之间的分组重叠。
"""
import argparse, glob, os
import pandas as pd
from collections import Counter
from common import load_results, infer_group, infer_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--heldout_prefix', default='PolypGen_')
    ap.add_argument('--manifests', default=None, help='如 manifests/polypgen_loco;其下 holdout_C*/seed0/{train,val,test_*}.csv')
    a = ap.parse_args()
    D = load_results(a.results, a.heldout_prefix)
    H = D[(D.dsg == 'HELDOUT') & (D.method == D.method.iloc[0]) & (D.prompt == D.prompt.iloc[0])]
    info = {}
    for hold, g in H.groupby('hold'):
        imgs = g.image.drop_duplicates()
        src = Counter(imgs.map(infer_source))
        grp = set(imgs.map(infer_group))
        info[hold] = (src, grp, len(imgs))
        print(f'{hold}: n={len(imgs)}  来源={dict(src)}  分组数={len(grp)}')
    print('\n== 折与折之间 ==')
    holds = sorted(info)
    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            a_, b_ = info[holds[i]], info[holds[j]]
            dom_a, dom_b = a_[0].most_common(1)[0][0], b_[0].most_common(1)[0][0]
            shared = {x for x in a_[1] & b_[1] if not x.isdigit()}
            same = dom_a == dom_b
            if same or shared:
                print(f'  {holds[i]} vs {holds[j]}: 主要来源 {dom_a}/{dom_b}'
                      f'{"  <-- 同一来源" if same else ""}; 共享分组 {len(shared)}'
                      f'{"  <-- 同一次检查被拆到两折: 帧级泄漏" if shared else ""}')
    if a.manifests:
        print('\n== manifests: 训练/验证/测试 分组重叠 ==')
        for d in sorted(glob.glob(os.path.join(a.manifests, 'holdout_*', 'seed*'))):
            def groups(pat):
                s = set()
                for f in glob.glob(os.path.join(d, pat)):
                    s |= set(pd.read_csv(f)['image'].map(lambda p: infer_group(os.path.splitext(os.path.basename(p))[0])))
                return {x for x in s if not x.isdigit()}
            tr, va, te = groups('train*.csv'), groups('val*.csv'), groups('test_PolypGen*.csv')
            print(f'  {d}: train∩test={len(tr & te)}  val∩test={len(va & te)}  train∩val={len(tr & va)}')


if __name__ == '__main__':
    main()
