# Run on the computer where Kaggle's archive.zip was downloaded: takes the 6 matching classes from original/,
# centre-crops to 4:3, resizes to 256x192, writes gd6_256x192.zip with index.csv (md5 of original bytes).
import zipfile, io, hashlib, os, csv
from PIL import Image
SRC=os.path.expanduser('~/Downloads/archive.zip'); OUT=os.path.expanduser('~/Downloads/gd6_256x192.zip')
CLASSES=['cardboard','glass','metal','paper','plastic','trash']
zin=zipfile.ZipFile(SRC); names=[n for n in zin.namelist() if n.startswith('original/') and n.split('/')[1] in CLASSES and n.lower().endswith(('.jpg','.jpeg','.png'))]
rows=[]
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_STORED) as zout:
    for n in sorted(names):
        raw=zin.read(n); c=n.split('/')[1]; im=Image.open(io.BytesIO(raw)).convert('RGB'); w,h=im.size
        tw=min(w,int(h*4/3)); th=min(h,int(w*3/4)); l=(w-tw)//2; t=(h-th)//2
        im=im.crop((l,t,l+tw,t+th)).resize((256,192),Image.BILINEAR); b=io.BytesIO(); im.save(b,'JPEG',quality=92)
        zout.writestr(f'gd6/{c}/{os.path.basename(n)}',b.getvalue()); rows.append((c,os.path.basename(n),hashlib.md5(raw).hexdigest(),w,h))
    s=io.StringIO(); wr=csv.writer(s); wr.writerow(['class','file','md5_original_bytes','orig_w','orig_h']); wr.writerows(rows); zout.writestr('gd6/index.csv',s.getvalue())
