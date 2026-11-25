import numpy as np, cv2
W,H=256,192
def _noise(rng,scale,shape=(H,W)):
    s=max(2,int(scale)); small=rng.standard_normal((H//s+2,W//s+2)).astype(np.float32)
    return cv2.resize(small,(W,H),interpolation=cv2.INTER_CUBIC)
def _fbm(rng,scales=(48,24,12,6,3),w=(1,.6,.4,.3,.2)):
    return sum(wi*_noise(rng,s) for s,wi in zip(scales,w))
def concrete(rng):
    g=140+22*_fbm(rng)+6*rng.standard_normal((H,W)); g=np.clip(g,0,255); t=rng.uniform(-8,8)
    return np.stack([g+t,g,g-t],2)
def asphalt(rng):
    g=70+16*_fbm(rng)+22*(rng.random((H,W))>0.965)+8*rng.standard_normal((H,W)); g=np.clip(g,0,255)
    return np.stack([g,g,g+3],2)
def wood(rng):
    x=np.arange(W,dtype=np.float32)[None,:]; y=np.arange(H,dtype=np.float32)[:,None]
    ang=rng.uniform(-0.25,0.25); xr=x*np.cos(ang)+y*np.sin(ang)
    grain=np.sin(xr/rng.uniform(2.5,5)+3*_noise(rng,24))+0.5*np.sin(xr/1.3+_noise(rng,6))
    base=np.array([rng.uniform(70,110),rng.uniform(110,150),rng.uniform(150,200)],np.float32)
    k=0.78+0.12*grain+0.05*_noise(rng,10)
    return np.clip(base[None,None,:]*k[...,None],0,255)
def grass(rng):
    n=_fbm(rng,(40,16,6,3,2),(1,.8,.6,.5,.4)); y=np.arange(H,dtype=np.float32)[:,None]
    streak=cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32),(0,0),sigmaX=0.7,sigmaY=3.0)
    g=np.clip(105+25*n+30*streak,20,255)
    return np.clip(np.stack([g*0.45+10,g,g*0.35],2),0,255)
def tiles(rng):
    s=rng.integers(36,64); base=rng.uniform(150,215); t=base+10*_noise(rng,32)+4*rng.standard_normal((H,W))
    x=np.arange(W)[None,:]; y=np.arange(H)[:,None]; off=rng.integers(0,s)
    line=((x+off)%s<2)|((y+off)%s<2); t=np.where(line,base-55,t); tint=np.array([rng.uniform(-10,5),rng.uniform(-5,5),rng.uniform(-5,10)])
    return np.clip(np.stack([t,t,t],2)+tint,0,255)
def carpet(rng):
    m=rng.uniform(70,150); base=np.array([m+rng.uniform(-15,15),m+rng.uniform(-15,15),m+rng.uniform(-15,15)])
    n=18*rng.standard_normal((H,W,1))+8*_fbm(rng)[...,None]
    return np.clip(base[None,None,:]+n,0,255)
def bin_interior(rng):
    yy=np.linspace(0,1,H)[:,None]; xx=np.linspace(-1,1,W)[None,:]
    v=1-0.55*(xx**2+(yy*2-1)**2)/2; g=(38+22*_fbm(rng))*v+4*rng.standard_normal((H,W))
    img=np.stack([g,g,g+4],2)
    for _ in range(rng.integers(5,10)):  # crumpled bag highlights
        cx,cy=rng.integers(0,W),rng.integers(0,H); r=rng.integers(12,40)
        col=rng.choice([40,70,120]); cv2.circle(img,(int(cx),int(cy)),int(r),(float(col),)*3,-1)
    img=cv2.GaussianBlur(img.astype(np.float32),(0,0),3)
    return np.clip(img+6*rng.standard_normal((H,W,1)),0,255)
def clutter(rng):
    img=concrete(rng)*0.7
    for _ in range(rng.integers(14,26)):
        pts=(rng.random((rng.integers(3,6),2))*[W,H]).astype(np.int32); col=[float(rng.integers(20,230)) for _ in range(3)]
        cv2.fillPoly(img,[pts],col)
    img=cv2.GaussianBlur(img.astype(np.float32),(0,0),2.2)
    return np.clip(img+5*rng.standard_normal((H,W,1)),0,255)
KINDS={'concrete':concrete,'asphalt':asphalt,'wood':wood,'grass':grass,'tiles':tiles,'carpet':carpet,'bin':bin_interior,'clutter':clutter}
def make_bg(kind,rng): return np.clip(KINDS[kind](rng),0,255).astype(np.float32)
