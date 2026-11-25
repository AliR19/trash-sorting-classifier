import numpy as np, cv2
W,H=256,192
def load(p):
    return cv2.resize(cv2.imread(p),(W,H),interpolation=cv2.INTER_AREA)
def cutout(img):
    """img BGR 256x192 -> soft mask (float 0..1) or None if unreliable"""
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float32)
    # background model: smooth plane fitted to border pixels (handles gradients / shadows at the edge)
    b=10; ys,xs=np.mgrid[0:H,0:W]
    border=np.zeros((H,W),bool); border[:b]=1; border[-b:]=1; border[:,:b]=1; border[:,-b:]=1
    A=np.stack([np.ones(border.sum()),xs[border]/W,ys[border]/H,(xs[border]/W)**2,(ys[border]/H)**2,(xs[border]*ys[border])/(W*H)],1)
    X=np.stack([np.ones(H*W),xs.ravel()/W,ys.ravel()/H,(xs.ravel()/W)**2,(ys.ravel()/H)**2,(xs.ravel()*ys.ravel())/(W*H)],1)
    bg=np.zeros_like(lab)
    for c in range(3):
        coef,*_=np.linalg.lstsq(A,lab[...,c][border],rcond=None); bg[...,c]=(X@coef).reshape(H,W)
    d=np.sqrt(((lab-bg)**2*np.array([0.6,1.5,1.5],np.float32)).sum(2))
    d=cv2.GaussianBlur(d,(0,0),1.2)
    t=max(8.0,float(np.percentile(d[border],99))*1.3)
    mask=np.full((H,W),cv2.GC_PR_BGD,np.uint8)
    mask[d>t]=cv2.GC_PR_FGD; mask[d>t*2.2]=cv2.GC_FGD; mask[border]=cv2.GC_BGD
    if (mask==cv2.GC_PR_FGD).sum()+(mask==cv2.GC_FGD).sum()<150: return None
    bgm=np.zeros((1,65),np.float64); fgm=np.zeros((1,65),np.float64)
    try: cv2.grabCut(img,mask,None,bgm,fgm,3,cv2.GC_INIT_WITH_MASK)
    except cv2.error: return None
    m=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,cc,st,_=cv2.connectedComponentsWithStats(m)
    if n<=1: return None
    keep=np.zeros_like(m)
    big=st[1:,cv2.CC_STAT_AREA].max()
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=0.15*big: keep[cc==i]=1
    # fill holes
    inv=(1-keep).astype(np.uint8); ff=inv.copy(); h2=np.zeros((H+2,W+2),np.uint8); cv2.floodFill(ff,h2,(0,0),2); keep[ff==1]=1
    frac=keep.mean()
    if frac<0.03 or frac>0.5: return None                      # tiny speck or object fills the frame (close-up)
    if keep[border].mean()>0.01: return None                   # object touches the image border -> cut-off
    cnts,_=cv2.findContours(keep.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    allp=np.vstack(cnts); hull=cv2.contourArea(cv2.convexHull(allp)); area=float(keep.sum())
    if area/max(hull,1)<0.6: return None                       # fragmented mask
    soft=cv2.GaussianBlur(keep.astype(np.float32),(0,0),1.0)
    return np.clip(soft,0,1)
