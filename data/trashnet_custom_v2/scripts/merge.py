import os, json, csv, shutil, collections, numpy as np, cv2
SEED=42; rng=np.random.default_rng(SEED)
CL=['cardboard','glass','metal','paper','plastic','trash']
rows=list(csv.DictReader(open('gd6/index.csv'))); G=np.load('Gbits.npy'); Gs=np.load('Gsmall.npy'); fl=json.load(open('gd_flags.json'))
# conservative second pass: any GD image within pHash<=8 and corr>=0.90 of a TrashNet image is also excluded
m=json.load(open('meta.json')); T=[];Ts=[]
def ph(g):
    g=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32); d=cv2.dct(g)[:8,:8].flatten(); return d>np.median(d[1:])
def small(g):
    g=cv2.resize(g,(64,48),interpolation=cv2.INTER_AREA).astype(np.float32); g-=g.mean(); return g/(np.linalg.norm(g)+1e-6)
for p in m['paths']:
    g=cv2.resize(cv2.imread(p,cv2.IMREAD_GRAYSCALE),(256,192),interpolation=cv2.INTER_AREA); T.append(ph(g)); Ts.append(small(g))
T=np.array(T,np.float32); Ts=np.array(Ts).reshape(len(T),-1)
DT=G.shape[1]-(G@T.T+(1-G)@(1-T).T)
loose=set(int(a) for a,b in np.argwhere(DT<=8) if float(Gs[a]@Ts[b])>=0.90)
excl=set(fl['overlap'])|set(fl['dropg'])|set(fl['cross'])|loose
stats=dict(gd_total=len(rows),gd_exact_trashnet_overlap=len(fl['overlap']),gd_within_dups=len(fl['dropg']),gd_loose_trashnet_matches_extra=len(loose-set(fl['overlap'])),gd_excluded_total=len(excl))
avail={c:[i for i,r in enumerate(rows) if r['class']==c and i not in excl] for c in CL}
stats['gd_novel_available']={c:len(v) for c,v in avail.items()}
SRC='trashnet_custom'; OUT='trashnet_custom_v2'; shutil.rmtree(OUT,ignore_errors=True); shutil.copytree(SRC,OUT)
man=list(csv.DictReader(open(f'{OUT}/manifest.csv')))
for r in man: r['dataset']='trashnet'
NT,NV,NTR=80,40,400; add=collections.Counter()
for c in CL:
    ids=list(rng.permutation(avail[c])); te=ids[:NT]; va=ids[NT:NT+NV]; tr=ids[NT+NV:NT+NV+NTR]
    for part,lst in (('test_external',te),('val',va),('train',tr)):
        for i in lst:
            f=rows[i]['file']; dst=f'{part}/{c}/gd_{f}'; os.makedirs(f'{OUT}/{part}/{c}',exist_ok=True)
            shutil.copy(f'gd6/{c}/{f}',f'{OUT}/{dst}')
            man.append(dict(file=dst,split=part,cls=c,source='gd_'+f,type='original',background='',dataset='garbage_dataset_kaggle')); add[(part,c)]+=1
with open(f'{OUT}/manifest.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=['file','split','cls','source','type','background','dataset']); w.writeheader(); w.writerows(man)
tab=collections.Counter((r['split'],r['cls']) for r in man)
stats['table']={s:{c:tab[(s,c)] for c in CL} for s in ['train','val','test','test_external','val_bgshift','test_bgshift']}
stats['gd_added']={p:{c:add[(p,c)] for c in CL} for p in ('train','val','test_external')}
stats['split_totals']={s:sum(v.values()) for s,v in stats['table'].items()}
old=json.load(open(f'{OUT}/build_stats.json')); old['merge']=stats; json.dump(old,open(f'{OUT}/build_stats.json','w'),indent=1)
print(json.dumps(stats,indent=1))
# integrity: external test never overlaps train/val sources by hash of image content
