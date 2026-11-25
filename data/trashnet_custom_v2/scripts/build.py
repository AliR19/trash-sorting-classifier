import os, json, hashlib, csv, shutil, sys, collections
import numpy as np, cv2
from sklearn.model_selection import train_test_split
import seg, bgs, comp
SEED=42; rng=np.random.default_rng(SEED)
m=json.load(open('meta.json')); P=m['paths']; L=np.array(m['labels']); C=m['classes']; MD5=m['md5']
D=np.load('D.npy'); n=len(P); stats={'original_images':n,'per_class_original':{C[i]:int((L==i).sum()) for i in range(6)}}
# ---- Change 1: exact duplicates ------------------------------------------------
by=collections.defaultdict(list)
for i,h in enumerate(MD5): by[h].append(i)
drop=set(); cross=0; same=0
for g in by.values():
    if len(g)>1:
        if len({L[i] for i in g})>1: drop|=set(g); cross+=len(g)          # conflicting labels -> remove every copy
        else: drop|=set(g[1:]); same+=len(g)-1                             # same label -> keep one copy
stats['removed_cross_class_exact']=cross; stats['removed_same_class_exact_copies']=same
# ---- Change 2: near-duplicates (same class, pHash<=6 and pixel correlation>=0.95) -
def small(p):
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); g=cv2.resize(g,(64,48),interpolation=cv2.INTER_AREA).astype(np.float32); g-=g.mean(); return g/(np.linalg.norm(g)+1e-6)
near=0
for a in range(n):
    if a in drop: continue
    for b in range(a+1,n):
        if b in drop or L[a]!=L[b] or D[a,b]>6: continue
        if float((small(P[a])*small(P[b])).sum())>=0.95: drop.add(b); near+=1
stats['removed_near_duplicates']=near
keep=[i for i in range(n) if i not in drop]; y=L[keep]
stats['clean_images']=len(keep); stats['per_class_clean']={C[i]:int((y==i).sum()) for i in range(6)}
# ---- Change 3: stratified 70/15/15 split ---------------------------------------
idx=np.array(keep)
tr,tmp=train_test_split(idx,test_size=0.30,stratify=L[idx],random_state=SEED)
va,te=train_test_split(tmp,test_size=0.50,stratify=L[tmp],random_state=SEED)
split={**{int(i):'train' for i in tr},**{int(i):'val' for i in va},**{int(i):'test' for i in te}}
OUT='trashnet_custom'; shutil.rmtree(OUT,ignore_errors=True)
def save(path,img): os.makedirs(os.path.dirname(path),exist_ok=True); cv2.imwrite(path,img,[cv2.IMWRITE_JPEG_QUALITY,92])
rows=[]; accepted=[]; kinds=list(bgs.KINDS)
for i in keep:
    img=seg.load(P[i]); fn=os.path.basename(P[i]); sp=split[i]; c=C[L[i]]
    save(f'{OUT}/{sp}/{c}/{fn}',img); rows.append(dict(file=f'{sp}/{c}/{fn}',split=sp,cls=c,source=fn,type='original',background=''))
    mk=seg.cutout(img)
    if mk is not None: accepted.append((i,img,mk))
stats['cutout_accepted']=len(accepted); stats['cutout_rejected']=len(keep)-len(accepted)
acc_by_split=collections.Counter(split[i] for i,_,_ in accepted); stats['cutout_accepted_by_split']=dict(acc_by_split)
# ---- Change 4: synthetic background-replaced images (TRAIN only) -----------------
VAR={'trash':4}; k=0
for i,img,mk in accepted:
    sp=split[i]; c=C[L[i]]; fn=os.path.splitext(os.path.basename(P[i]))[0]
    if sp=='train':
        for v in range(VAR.get(c,1)):
            kind=kinds[(k:=k+1)%len(kinds)]; out=comp.composite(img,mk,bgs.make_bg(kind,rng),rng)
            f=f'train/{c}/{fn}_bg{v}_{kind}.jpg'; save(f'{OUT}/{f}',out); rows.append(dict(file=f,split='train',cls=c,source=fn,type='synthetic_background',background=kind))
    else:
        # ---- Change 5: background-shift robustness sets (val / test), one variant each ------
        kind=kinds[(k:=k+1)%len(kinds)]; out=comp.composite(img,mk,bgs.make_bg(kind,rng),rng)
        f=f'{sp}_bgshift/{c}/{fn}_{kind}.jpg'; save(f'{OUT}/{f}',out); rows.append(dict(file=f,split=f'{sp}_bgshift',cls=c,source=fn,type='synthetic_background',background=kind))
with open(f'{OUT}/manifest.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tab=collections.Counter((r['split'],r['cls']) for r in rows)
stats['table']={s:{c:tab[(s,c)] for c in C} for s in ['train','val','test','val_bgshift','test_bgshift']}
stats['train_total_with_synthetic']=sum(stats['table']['train'].values())
stats['train_synthetic']=sum(1 for r in rows if r['split']=='train' and r['type']!='original')
json.dump(stats,open(f'{OUT}/build_stats.json','w'),indent=1); print(json.dumps(stats,indent=1))
