"""公共函数:读取 results/<实验>/holdout_*/<方法>/<数据集>/<提示>/seed*/metrics.csv,
并从图像名推断"分组"(同一患者/同一次检查/同一视频序列),用于聚类统计和泄漏检查。"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

HELD_PREFIX = 'PolypGen_'      # 留出中心数据集的目录名前缀;若改名(如 Source_A),用 --heldout_prefix


def infer_group(image: str) -> str:
    """图像名 -> 分组。规则(按顺序):
    1. 去掉 'dataset__' 前缀(补丁后的新命名)
    2. HMUH_06 200731_200713_BN001_002 / 序列帧: 去掉末尾 _NNN 帧号 -> 同一次检查
    3. UUID(8-4-4-4-12): 无法判断,视为各自独立(需要数据方提供患者表,见建议文档)
    4. 100H0003 这类 Kvasir 风格: 取前 6 位作"源标记",不代表患者
    5. 纯数字(ColonDB/ETIS/CVC-300): CVC 系列是视频序列帧,按 每 N 帧并为一组 不可靠,返回自身
    """
    s = str(image).split('__')[-1]
    m = re.match(r'^(.*)_(\d{1,4})$', s)
    if m and ('BN' in s or 'HMUH' in s):
        return m.group(1)
    return s


def infer_source(image: str) -> str:
    """粗粒度来源(只用于发现划分泄漏的线索,不是真值)。"""
    s = str(image).split('__')[-1]
    if s.startswith('HMUH'):
        return 'HMUH'
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-', s):
        return 'UUID'
    if s.startswith('100H'):
        return '100H'
    if re.match(r'^ck[a-z0-9]{4}', s):
        return 'ck'
    if s.isdigit():
        return 'numeric'
    return 'other'


def load_results(root, heldout_prefix=HELD_PREFIX, check_mask_size=False):
    """返回长表。列: hold, method, ds, dsg, prompt, seed, image, 各指标, flag, [size_ok]"""
    root = Path(root)
    rows = []
    for f in sorted(root.glob('holdout_*/*/*/*/seed*/metrics.csv')):
        hold, method, ds, prompt, seed = f.relative_to(root).parts[:5]
        d = pd.read_csv(f)
        d['hold'], d['method'], d['ds'], d['prompt'], d['seed'] = hold, method, ds, prompt, seed
        if check_mask_size:
            mdir = f.parent / 'masks'
            ok = []
            from PIL import Image
            for img, h, w in zip(d['image'], d['h'], d['w']):
                p = mdir / f'{img}.png'
                try:
                    with Image.open(p) as im:
                        ok.append(im.size == (int(w), int(h)))
                except Exception:
                    ok.append(False)
            d['size_ok'] = ok
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f'{root} 下没有 holdout_*/<方法>/<数据集>/<提示>/seed*/metrics.csv')
    D = pd.concat(rows, ignore_index=True)
    if check_mask_size:
        # 第二类碰撞:尺寸相同的两个数据集(如 ColonDB 与 CVC-300 都是 574x500)同名图共用缓存,
        # 掩码尺寸检测不出来,但两个数据集里会出现"逐项完全相同"的指标行。保留较小的数据集,标记较大者。
        key = ['hold', 'method', 'prompt', 'seed', 'dice', 'hd95', 'assd', 'gt_area']
        n_ds = D.groupby(['hold', 'method', 'prompt', 'seed', 'ds'])['image'].transform('size')
        D['_n'] = n_ds
        dup = D[(D['gt_area'] > 0)].groupby(key)['ds'].transform('nunique') > 1
        dup = dup.reindex(D.index, fill_value=False)
        minn = D[dup].groupby(key)['_n'].transform('min')
        D['dup_larger'] = False
        D.loc[dup, 'dup_larger'] = D.loc[dup, '_n'] > minn
        D['size_ok'] = D['size_ok'] & ~D['dup_larger']
        D = D.drop(columns=['_n'])
    D['image'] = D['image'].astype(str)
    D['dsg'] = np.where(D['ds'].str.startswith(heldout_prefix), 'HELDOUT', D['ds'])
    D['group'] = D['image'].map(infer_group)
    return D
