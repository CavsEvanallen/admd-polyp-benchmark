#!/usr/bin/env python3
"""
生成实验清单（manifest）CSV：image, mask, dataset, domain_id, split

支持四种协议（--protocol）：
  thesis         复现大论文设置：ClinicDB、ISIC2018 各按 7:1:2 划分；ColonDB、PH2 全部作测试。
                 domain_id: 0 = 结肠镜, 1 = 皮肤镜（K=2，与已发表设置一致，用于 E0–E5）
  pranet         息肉通行协议：Kvasir 900 + ClinicDB 550 训练；五个测试集。
                 domain_id: 0 = Kvasir, 1 = ClinicDB（同模态双源域，K=2）
  polypgen_loco  PolypGen 六中心留一：每折 5 个中心训练（每中心 9:1 分 train/val），留出中心 + ColonDB/ETIS/CVC-300 测试。
                 domain_id: 训练中心在本折内的序号 0..4；测试集 domain_id = -1（未知域）
  skin           ISIC2018 训练集 8:2 分 train/val；PH2（及去重后的 ISIC2017 测试集）作测试。domain_id 全 0

目录规范见 01_datasets.md。用法示例：
  python 02_make_manifests.py --root $WORK/data --out $WORK/manifests --protocol thesis --seed 0
  python 02_make_manifests.py --root $WORK/data --out $WORK/manifests --protocol polypgen_loco
"""
import argparse, csv, random, sys
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_SUFFIXES = ('_mask', '_lesion', '_segmentation', '_Segmentation', '_gt', '_GT')


def list_images(d):
    return sorted(p for p in Path(d).rglob('*') if p.suffix.lower() in IMG_EXT and not p.name.startswith('.'))


def norm_stem(stem):
    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def pair(img_dir, mask_dir, dataset):
    masks = {}
    for p in list_images(mask_dir):
        masks.setdefault(norm_stem(p.stem), p)
    rows, missing = [], 0
    for img in list_images(img_dir):
        m = masks.get(img.stem)
        if m is None:
            missing += 1
            continue
        rows.append([str(img.resolve()), str(m.resolve()), dataset])
    if missing:
        print(f'  [warn] {dataset}: {missing} images without mask under {img_dir}', file=sys.stderr)
    if not rows:
        raise RuntimeError(f'{dataset}: no image/mask pairs found in {img_dir} / {mask_dir}')
    return rows


def load_dataset(root, name, dataset_label=None):
    """在 root/name 下找 images+masks（或 Original+Ground Truth）。"""
    d = Path(root) / name
    for im, mk in (('images', 'masks'), ('Original', 'Ground Truth'), ('image', 'mask'), ('imgs', 'masks'), ('Images', 'Masks')):
        if (d / im).exists() and (d / mk).exists():
            return pair(d / im, d / mk, dataset_label or name)
    raise FileNotFoundError(f'{d}: 需要 images/ 与 masks/ 子目录（见 01_datasets.md）')


def polypgen_centers(root):
    """返回 {center_id: rows}。兼容 PolypGen2021_MultiCenterData_v3/data_C1/images_C1 结构。"""
    hits = sorted(Path(root).glob('PolypGen*/data_C*')) or sorted(Path(root).glob('PolypGen*/*/data_C*'))
    if not hits:
        raise FileNotFoundError('未找到 PolypGen*/data_C*/ 目录')
    centers = {}
    for d in hits:
        cid = int(d.name.split('_C')[-1])
        im = next(d.glob('images_C*'), None)
        mk = next(d.glob('masks_C*'), None)
        if im is None or mk is None:
            print(f'  [warn] {d}: 缺 images_Cx 或 masks_Cx，跳过', file=sys.stderr)
            continue
        centers[cid] = pair(im, mk, f'PolypGen_C{cid}')
    return centers


def shuffled_split(rows, fracs, seed):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    a = int(round(n * fracs[0]))
    b = a + int(round(n * fracs[1])) if len(fracs) > 1 else n
    return rows[:a], rows[a:b], rows[b:]


def write_csv(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'mask', 'dataset', 'domain_id', 'split'])
        w.writerows(items)
    print(f'  wrote {path}  ({len(items)} rows)')


def tag(rows, domain_id, split):
    return [r + [domain_id, split] for r in rows]


def protocol_thesis(root, out, seed):
    clinic = load_dataset(root, 'CVC-ClinicDB')
    isic = load_dataset(root, 'ISIC2018')
    colon = load_dataset(root, 'CVC-ColonDB')
    ph2 = load_dataset(root, 'PH2')
    tr0, va0, te0 = shuffled_split(clinic, (0.7, 0.1), seed)
    tr1, va1, te1 = shuffled_split(isic, (0.7, 0.1), seed)
    o = out / 'thesis' / f'seed{seed}'
    write_csv(o / 'train.csv', tag(tr0, 0, 'train') + tag(tr1, 1, 'train'))
    write_csv(o / 'val.csv', tag(va0, 0, 'val') + tag(va1, 1, 'val'))
    write_csv(o / 'test_CVC-ClinicDB.csv', tag(te0, 0, 'test'))
    write_csv(o / 'test_ISIC2018.csv', tag(te1, 1, 'test'))
    write_csv(o / 'test_CVC-ColonDB.csv', tag(colon, 0, 'test'))
    write_csv(o / 'test_PH2.csv', tag(ph2, 1, 'test'))


def protocol_pranet(root, out, seed):
    pr = Path(root) / 'PraNet'
    o = out / 'pranet'
    if (pr / 'TrainDataset').exists():
        train = pair(pr / 'TrainDataset' / 'images', pr / 'TrainDataset' / 'masks', 'PraNet-Train')
        # PraNet 训练包里 Kvasir 与 ClinicDB 混在一起；按文件名区分：ClinicDB 为纯数字，Kvasir 为 cju*
        for r in train:
            name = Path(r[0]).stem
            r[2] = 'CVC-ClinicDB' if name.isdigit() else 'Kvasir-SEG'
        kv = [r for r in train if r[2] == 'Kvasir-SEG']
        cl = [r for r in train if r[2] == 'CVC-ClinicDB']
        print(f'  PraNet train: Kvasir {len(kv)}, ClinicDB {len(cl)}')
        tests = {}
        for name in ('Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300'):
            d = pr / 'TestDataset' / name
            if d.exists():
                tests[name] = pair(d / 'images', d / 'masks', name)
        # 训练集内再切 10% 做验证（PraNet 原协议没有验证集，这里为早停需要）
        tr_kv, va_kv, _ = shuffled_split(kv, (0.9, 0.1), seed)
        tr_cl, va_cl, _ = shuffled_split(cl, (0.9, 0.1), seed)
        write_csv(o / f'seed{seed}' / 'train.csv', tag(tr_kv, 0, 'train') + tag(tr_cl, 1, 'train'))
        write_csv(o / f'seed{seed}' / 'val.csv', tag(va_kv, 0, 'val') + tag(va_cl, 1, 'val'))
        for name, rows in tests.items():
            did = {'Kvasir': 0, 'CVC-ClinicDB': 1}.get(name, -1)
            write_csv(o / f'seed{seed}' / f'test_{name}.csv', tag(rows, did, 'test'))
    else:
        print('  [warn] 未找到 PraNet 划分包，退回随机划分（与文献不完全一致，论文里要说明）', file=sys.stderr)
        kv = load_dataset(root, 'Kvasir-SEG')
        cl = load_dataset(root, 'CVC-ClinicDB')
        kv_tr, kv_te, _ = shuffled_split(kv, (0.9, 0.1), seed)
        cl_tr, cl_te, _ = shuffled_split(cl, (550 / 612, 62 / 612), seed)
        tr_kv, va_kv, _ = shuffled_split(kv_tr, (0.9, 0.1), seed)
        tr_cl, va_cl, _ = shuffled_split(cl_tr, (0.9, 0.1), seed)
        write_csv(o / f'seed{seed}' / 'train.csv', tag(tr_kv, 0, 'train') + tag(tr_cl, 1, 'train'))
        write_csv(o / f'seed{seed}' / 'val.csv', tag(va_kv, 0, 'val') + tag(va_cl, 1, 'val'))
        write_csv(o / f'seed{seed}' / 'test_Kvasir.csv', tag(kv_te, 0, 'test'))
        write_csv(o / f'seed{seed}' / 'test_CVC-ClinicDB.csv', tag(cl_te, 1, 'test'))
        for name in ('CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300'):
            try:
                write_csv(o / f'seed{seed}' / f'test_{name}.csv', tag(load_dataset(root, name), -1, 'test'))
            except FileNotFoundError as e:
                print(f'  [warn] {e}', file=sys.stderr)


def protocol_polypgen_loco(root, out, seed):
    centers = polypgen_centers(root)
    extra = {}
    for name in ('CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300'):
        for cand in (name, f'PraNet/TestDataset/{name}'):
            try:
                extra[name] = load_dataset(root, cand, name)
                break
            except FileNotFoundError:
                continue
    for held in sorted(centers):
        o = out / 'polypgen_loco' / f'holdout_C{held}' / f'seed{seed}'
        train, val = [], []
        src = [c for c in sorted(centers) if c != held]
        for k, c in enumerate(src):           # k = 本折内的域序号 0..4
            tr, va, _ = shuffled_split(centers[c], (0.9, 0.1), seed)
            train += tag(tr, k, 'train')
            val += tag(va, k, 'val')
        write_csv(o / 'train.csv', train)
        write_csv(o / 'val.csv', val)
        write_csv(o / f'test_PolypGen_C{held}.csv', tag(centers[held], -1, 'test'))
        for name, rows in extra.items():
            write_csv(o / f'test_{name}.csv', tag(rows, -1, 'test'))
        with open(o / 'domains.txt', 'w') as f:
            f.write('\n'.join(f'{k}\tPolypGen_C{c}' for k, c in enumerate(src)) + '\n')


def protocol_skin(root, out, seed):
    isic = load_dataset(root, 'ISIC2018')
    tr, va, _ = shuffled_split(isic, (0.8, 0.2), seed)
    o = out / 'skin' / f'seed{seed}'
    write_csv(o / 'train.csv', tag(tr, 0, 'train'))
    write_csv(o / 'val.csv', tag(va, 0, 'val'))
    write_csv(o / 'test_PH2.csv', tag(load_dataset(root, 'PH2'), -1, 'test'))
    for name in ('ISIC2018_Test', 'ISIC2017_Test'):
        try:
            write_csv(o / f'test_{name}.csv', tag(load_dataset(root, name), -1, 'test'))
        except FileNotFoundError:
            print(f'  [info] {name} 不存在，跳过', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', required=True, choices=['thesis', 'pranet', 'polypgen_loco', 'skin', 'all'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    todo = ['thesis', 'pranet', 'polypgen_loco', 'skin'] if a.protocol == 'all' else [a.protocol]
    for p in todo:
        print(f'== protocol {p} (seed {a.seed}) ==')
        try:
            globals()[f'protocol_{p}'](a.root, out, a.seed)
        except (FileNotFoundError, RuntimeError) as e:
            print(f'  [skip] {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
