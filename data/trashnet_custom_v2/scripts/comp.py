import numpy as np, cv2
W,H=256,192
def composite(img,mask,bg,rng):
    """paste cut-out `img` (BGR uint8) with soft `mask` onto `bg` (float BGR). Random flip/rotation/scale/shift, soft shadow, lighting match."""
    s=rng.uniform(0.8,1.15); ang=rng.uniform(-25,25); flip=rng.random()<0.5
    M=cv2.getRotationMatrix2D((W/2,H/2),ang,s); M[0,2]+=rng.uniform(-0.12,0.12)*W; M[1,2]+=rng.uniform(-0.12,0.12)*H
    im=img.copy(); mk=mask.copy()
    if flip: im=im[:,::-1]; mk=mk[:,::-1]
    im=cv2.warpAffine(np.ascontiguousarray(im),M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    mk=cv2.warpAffine(np.ascontiguousarray(mk),M,(W,H),flags=cv2.INTER_LINEAR,borderValue=0)
    # soft contact shadow
    sh=cv2.GaussianBlur(mk,(0,0),4.0); dx,dy=int(rng.integers(3,9)),int(rng.integers(3,9))
    sh=np.roll(np.roll(sh,dy,0),dx,1)*rng.uniform(0.25,0.5)
    out=bg*(1-sh[...,None])
    # lighting match: scale object brightness toward background brightness a little
    gain=np.clip((bg.mean()+60)/(im.mean()+60),0.8,1.25)
    obj=np.clip(im.astype(np.float32)*gain,0,255)
    out=out*(1-mk[...,None])+obj*mk[...,None]
    # global photometric jitter + sensor noise + blur
    out=out*rng.uniform(0.85,1.12)+rng.uniform(-10,10)
    out=cv2.GaussianBlur(out,(0,0),rng.uniform(0.3,0.9))+rng.standard_normal(out.shape).astype(np.float32)*rng.uniform(1,3)
    return np.clip(out,0,255).astype(np.uint8)
