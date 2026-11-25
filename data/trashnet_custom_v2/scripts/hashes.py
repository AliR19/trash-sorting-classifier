import os, hashlib, numpy as np, cv2, json
ROOT='/tmp/tn/dataset-resized'
CLASSES=sorted(d for d in os.listdir(ROOT) if os.path.isdir(f'{ROOT}/{d}'))
paths,labels=[],[]
for ci,c in enumerate(CLASSES):
    for f in sorted(os.listdir(f'{ROOT}/{c}')):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            paths.append(f'{ROOT}/{c}/{f}'); labels.append(ci)
labels=np.array(labels)
def phash(p):
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); g=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    d=cv2.dct(g)[:8,:8]; m=np.median(d.flatten()[1:]); bits=(d.flatten()>m)
    return bits
H=np.array([phash(p) for p in paths])
md5=[hashlib.md5(open(p,'rb').read()).hexdigest() for p in paths]
np.save('H.npy',H); json.dump(dict(paths=paths,labels=labels.tolist(),md5=md5,classes=CLASSES),open('meta.json','w'))
n=len(paths); Hi=H.astype(np.int8)
D=(Hi[:,None,:]!=Hi[None,:,:]).sum(2); np.save('D.npy',D.astype(np.uint8))
iu=np.triu_indices(n,1); d=D[iu]
for t in (0,2,4,6,8,10): print('pairs with hamming <=',t,':',int((d<=t).sum()))
