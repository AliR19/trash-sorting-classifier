import os, json, numpy as np, cv2, csv, collections
def ph(g):
    g=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32); d=cv2.dct(g)[:8,:8].flatten(); return d>np.median(d[1:])
def small(g):
    g=cv2.resize(g,(64,48),interpolation=cv2.INTER_AREA).astype(np.float32); g-=g.mean(); return g/(np.linalg.norm(g)+1e-6)
# TrashNet (all 2527 originals, resized the same way as the dataset: 256x192)
m=json.load(open('meta.json')); TP=m['paths']
T=[]; Ts=[]
for p in TP:
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); g=cv2.resize(g,(256,192),interpolation=cv2.INTER_AREA); T.append(ph(g)); Ts.append(small(g))
rows=list(csv.DictReader(open('gd6/index.csv')))
G=[];Gs=[]
for r in rows:
    g=cv2.imread(f"gd6/{r['class']}/{r['file']}",cv2.IMREAD_GRAYSCALE); G.append(ph(g)); Gs.append(small(g))
T=np.array(T,np.float32); G=np.array(G,np.float32); Ts=np.array(Ts).reshape(len(T),-1); Gs=np.array(Gs).reshape(len(G),-1)
np.save('Gbits.npy',G); np.save('Gsmall.npy',Gs)
def ham(A,B): return A.shape[1]-(A@B.T+(1-A)@(1-B).T)
DT=ham(G,T)                                   # GD x TrashNet
cand=np.argwhere(DT<=6); ov=set()
for gi,ti in cand:
    if float(Gs[gi]@Ts[ti])>=0.95: ov.add(int(gi))
print('GD images that match a TrashNet image (pHash<=6 & corr>=0.95):',len(ov),'of',len(G))
print('by class:',collections.Counter(rows[i]['class'] for i in ov))
# stricter exact-ish matches
ex=set(int(gi) for gi,ti in np.argwhere(DT<=2) if float(Gs[gi]@Ts[ti])>=0.99); print('near-identical (d<=2, corr>=0.99):',len(ex))
# within-GD near duplicates
DG=ham(G,G); np.fill_diagonal(DG,99); dropg=set(); cross=set()
for a,b in np.argwhere(np.triu(DG<=6,1)):
    if rows[a]['class']==rows[b]['class']:
        if float(Gs[a]@Gs[b])>=0.95: dropg.add(int(b))
    elif DG[a,b]<=2 and float(Gs[a]@Gs[b])>=0.99: cross.add(int(a)); cross.add(int(b))
print('GD within-class near-duplicates to drop:',len(dropg),'| GD cross-class conflicting copies:',len(cross))
json.dump(dict(overlap=sorted(ov),dropg=sorted(dropg),cross=sorted(cross)),open('gd_flags.json','w'))
