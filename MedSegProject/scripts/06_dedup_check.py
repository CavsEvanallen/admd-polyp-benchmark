#!/usr/bin/env python3
"""
训练池 vs 测试集 泄漏核查：精确重复（MD5）+ 近似重复（dHash 64 位，汉明距离 <= --thr）。

用法：
  python 06_dedup_check.py --train_dirs $WORK/data/sammed_subset $WORK/data/PraNet/TrainDataset/images \
                           --test_dirs  $WORK/data/CVC-ColonDB/images $WORK/data/PH2/images \
                           --out $WORK/results/dedup_report.csv --thr 6
输出 CSV：test_file, train_file, kind(md5|dhash), hamming；末尾打印每个测试集的命中数。
论文 Data 一节需要写："we found N exact and M near duplicates and removed them from the training pool" 或 "none"。
"""
import argparse, csv, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def files(d):
    return sorted(p for p in Path(d).rglob('*') if p.suffix.lower() in IMG_EXT)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dhash(p, size=8):
    im = Image.open(p).convert('L').resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count('1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dirs', nargs='+', required=True)
    ap.add_argument('--test_dirs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=int, default=6)
    a = ap.parse_args()

    train = []
    for d in a.train_dirs:
        for p in files(d):
            train.append((str(p), md5(p), dhash(p)))
    print(f'train pool: {len(train)} files')
    md5_index = {}
    for p, h, _ in train:
        md5_index.setdefault(h, []).append(p)
    train_hashes = np.array([t[2] for t in train], dtype=np.uint64)
    train_paths = [t[0] for t in train]

    hits = []
    summary = {}
    for d in a.test_dirs:
        n_exact = n_near = 0
        for p in files(d):
            h, dh = md5(p), dhash(p)
            if h in md5_index:
                n_exact += 1
                for tp in md5_index[h]:
                    hits.append([str(p), tp, 'md5', 0])
                continue
            x = np.bitwise_xor(train_hashes, np.uint64(dh))
            # popcount
            ham = np.array([bin(int(v)).count('1') for v in x])
            idx = np.nonzero(ham <= a.thr)[0]
            if len(idx):
                n_near += 1
                for i in idx:
                    hits.append([str(p), train_paths[i], 'dhash', int(ham[i])])
        summary[d] = (n_exact, n_near)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['test_file', 'train_file', 'kind', 'hamming'])
        w.writerows(hits)
    print(f'wrote {a.out} ({len(hits)} hit pairs)')
    for d, (e, n) in summary.items():
        print(f'  {d}: exact={e} near(<= {a.thr})={n}')


if __name__ == '__main__':
    main()
