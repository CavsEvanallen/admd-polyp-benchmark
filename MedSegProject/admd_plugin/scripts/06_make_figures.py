import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
C={'admd':'#0072B2','uniform_ens':'#009E73','wide_sam':'#D55E00'}
N={'admd':'ADMD','uniform_ens':'Uniform ensemble','wide_sam':'Pixel adapter'}
# ---- Fig 2: paired Dice differences vs standard decoder
d=pd.read_csv('fig2.csv')
order=['C1','C2','C3','C4','C5','CVC-ColonDB','ETIS-LaribPolypDB','CVC-300']
lab=['C1','C2','C3','C4','C5','ColonDB','ETIS','CVC-300']
fig,ax=plt.subplots(2,2,figsize=(10,6),sharex=True)
for i,enc in enumerate(['MobileSAM','SAM ViT-B']):
    for j,pr in enumerate(['1pt','bbox']):
        a=ax[j,i]; s=d[(d.enc==enc)&(d.prompt==pr)]
        a.axhspan(-0.01,0.01,color='#dddddd',alpha=.6,lw=0); a.axhline(0,color='k',lw=.8)
        for k,m in enumerate(['admd','uniform_ens','wide_sam']):
            t=s[s.method==m].set_index('ds').loc[order]; x=np.arange(8)+(k-1)*0.25
            a.errorbar(x,t['diff'],yerr=[t['diff']-t.lo,t.hi-t['diff']],fmt='o',ms=4,color=C[m],capsize=2,label=N[m])
        a.axvline(4.5,color='#999',lw=.6,ls=':'); a.set_xticks(range(8)); a.set_xticklabels(lab,rotation=30)
        a.set_title(f'{enc}, {"point" if pr=="1pt" else "box"} prompt',loc='left',fontsize=10)
        if i==0: a.set_ylabel('ΔDice vs standard decoder')
ax[0,0].legend(frameon=False,ncol=3,loc='upper left',fontsize=8)
ax[0,0].text(0.01,0.02,'grey band = ±0.01 equivalence margin',transform=ax[0,0].transAxes,fontsize=7,color='#555')
fig.tight_layout(); fig.savefig('figs/fig2_paired_dice.png',dpi=300); fig.savefig('figs/fig2_paired_dice.pdf'); plt.close()
# ---- Fig 3: recognizer behaviour (MobileSAM, ADMD, box)
g=pd.read_csv('fig3.csv'); ds=['C1','C2','C3','C4','C5','CVC-ColonDB','ETIS-LaribPolypDB','CVC-300']
fig,(a,b)=plt.subplots(1,2,figsize=(10,3.6))
bins=np.linspace(0.25,0.5,26)
a.hist(g.maxp,bins=bins,color='#0072B2',alpha=.85); a.set_xlim(0.18,0.5); a.axvline(0.2,color='k',ls='--',lw=1); a.text(0.205,a.get_ylim()[1]*0.92,'chance (0.20)',fontsize=8)
a.set_xlabel('max domain probability (per image, mean over folds and seeds)'); a.set_ylabel('images'); a.set_title('(a) Recognizer confidence, all test images',loc='left',fontsize=10)
from scipy.stats import spearmanr
cols=plt.cm.tab10(np.arange(8))
for k,n in enumerate(ds):
    t=g[g.ds==n]; r=spearmanr(t.maxp,t.dice)[0]
    b.scatter(t.maxp,t.dice,s=5,alpha=.35,color=cols[k],label=f'{n.replace("-LaribPolypDB","")} (ρ={r:+.2f})')
b.set_xlabel('max domain probability'); b.set_ylabel('Dice (box prompt)'); b.set_ylim(0.3,1.0); b.set_title('(b) Confidence vs. accuracy',loc='left',fontsize=10)
b.legend(frameon=False,fontsize=7,markerscale=2,ncol=2,loc='lower right')
fig.tight_layout(); fig.savefig('figs/fig3_recognizer.png',dpi=300); fig.savefig('figs/fig3_recognizer.pdf'); plt.close()

fig,ax=plt.subplots(figsize=(12,5.2)); ax.set_xlim(0,120); ax.set_ylim(0,52); ax.axis('off')
def box(x,y,w,h,t,fc='#EAF2F8',fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec='#2C3E50',lw=1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=1,color='#2C3E50'))
box(1,34,14,9,'Image\n1024×1024',fc='#F4F4F4'); box(20,34,17,9,'Frozen encoder\nMobileSAM /\nSAM ViT-B',fc='#D5D8DC')
box(1,7,14,9,'Prompt\npoint / box',fc='#F4F4F4'); box(20,7,17,9,'SAM prompt\nencoder (frozen)',fc='#D5D8DC')
arr(15,38.5,20,38.5); arr(15,11.5,20,11.5)
ax.text(54,50.2,'ADMD decoder (trainable, 20.56 M parameters)',ha='center',fontsize=10,fontweight='bold',color='#0072B2')
ax.add_patch(plt.Rectangle((41,3),77,45.5,fill=False,ec='#0072B2',ls='--',lw=1))
box(44,38,24,8,'Recognizer\nd = softmax(W · mean(F)),  K = 5',fc='#FCF3CF',fs=7.5)
box(44,26,24,8,'Query gate\nQ̃ = Q ⊙ (1 + tanh(W_q d))',fc='#FCF3CF',fs=7.5)
box(44,13,24,9,'Gated alignment\nh = F + γ·max(d)·Attn(F ; Q̃)',fc='#FCF3CF',fs=7.5)
arr(37,38.5,44,42); arr(37,11.5,44,30); arr(56,38,56,34); arr(56,26,56,22); arr(37,36,44,17)
ax.text(40.2,41.5,'F',fontsize=7,ha='right',color='#555'); ax.text(40.2,25,'Q',fontsize=7,ha='right',color='#555')
for i in range(5):
    y=5+i*8.6; box(77,y,17,6.4,f'Branch {i+1}\n(SAM decoder)',fc='#D6EAF8',fs=7); arr(68,17.5,77,y+3.2)
box(97,18,20,12,'Fusion router\nw = softmax(W_a · mean(h))\nM = Σ_k w_k M_k',fc='#FCF3CF',fs=7)
for i in range(5): arr(94,5+i*8.6+3.2,97,24)
ax.text(1,0.8,'Controls: standard decoder = K = 1, no gate, no alignment;  uniform ensemble = same five branches, w = 1/K;  pixel adapter = random MLP on F in front of a standard decoder.',fontsize=7,color='#444')
fig.savefig('figs/fig1_architecture.png',dpi=300,bbox_inches='tight'); fig.savefig('figs/fig1_architecture.pdf',bbox_inches='tight')
