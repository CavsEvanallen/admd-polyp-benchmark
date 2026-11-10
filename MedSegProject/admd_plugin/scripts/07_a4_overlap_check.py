import glob, pandas as pd, numpy as np
from pathlib import Path
R=Path('/tmp/h4/ADMD提交4'); rng=np.random.default_rng(0)
def load(root):
    rows=[]
    for f in glob.glob(str(R/root/'loso/holdout_*/*/Source_CVC-*/*/seed*/metrics.csv')):
        p=Path(f).relative_to(R/root/'loso').parts
        d=pd.read_csv(f); d['hold'],d['method'],d['ds'],d['prompt'],d['seed']=p[0],p[1],p[2],p[3],p[4]; rows.append(d)
    d=pd.concat(rows); return d[d.gt_area>0]
d=load('results')
# 1. identity check: ColonDB images whose metrics equal CVC-300 images (same fold/method/prompt/seed)
key=['hold','method','prompt','seed','dice','hd95','gt_area','pred_area']
c=d[d.ds=='Source_CVC-ColonDB']; v=d[d.ds=='Source_CVC-300']
m=c.merge(v,on=key,suffixes=('_c','_v'))
pairs=m.groupby(['image_c','image_v']).size().reset_index(name='n')
full=pairs[pairs.n>=0.9*pairs.n.max()]
print('matched ColonDB images:',full.image_c.nunique(),' CVC-300 images:',full.image_v.nunique(),' (cells per pair max',pairs.n.max(),')')
nums=sorted(int(x.split('__')[1]) for x in full.image_c); print('ColonDB numbers range',nums[0],nums[-1],'count',len(nums), 'contiguous' if nums==list(range(nums[0],nums[-1]+1)) else 'NOT contiguous')
dupset=set(full.image_c)
def boot(x):
    idx=rng.integers(0,len(x),(3000,len(x))); return np.percentile(x[idx].mean(1),[2.5,97.5])
for root,enc in [('results','MobileSAM'),('results_sam_b','SAM ViT-B')]:
    d=load(root); print('\n==',enc,'ColonDB all vs without overlap')
    for pr in ['1pt','bbox']:
        for lab,ex in [('all',False),('excl',True)]:
            s=d[(d.ds=='Source_CVC-ColonDB')&(d.prompt==pr)]
            pv=s.pivot_table(index='image',columns='method',values='dice')
            if ex: pv=pv[~pv.index.isin(dupset)]
            r=f'{pr:5s}{lab:5s}n={len(pv)} std={pv.sam_ft.mean():.4f} admd={pv.admd.mean():.4f} uni={pv.uniform_ens.mean():.4f} wide={pv.wide_sam.mean():.4f}'
            for a in ['admd','wide_sam']:
                x=(pv[a]-pv.sam_ft).values; lo,hi=boot(x); r+=f' | {a}-std {x.mean():+.4f}[{lo:+.4f},{hi:+.4f}]'
            print(r)
