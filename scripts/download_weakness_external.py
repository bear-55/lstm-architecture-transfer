"""Fetch public external replication archives; validate source MD5 and size."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/weakness_resolution_20261102/external'
ITEMS=[('Kratzert_et_al_2019_HESSD_LSTM_4_REGIONAL_MODELING.zip','83ea5312635e44dc824eeb99eda12f06','Kratzert_et_al_2019_HESSD_LSTM_4_REGIONAL_MODELING.zip',939191262,'224985a9c4d0932e607f4c9fe8d61ee8')]

def valid(p,size,md5):
    if not p.exists() or p.stat().st_size!=size:return False
    h=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()==md5

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for name,resource,remote,size,md5 in ITEMS:
        p=DEST/name
        if valid(p,size,md5):print('VERIFIED_EXISTING',name);continue
        url=f'https://www.hydroshare.org/resource/{resource}/data/contents/{remote}'
        tmp=p.with_suffix(p.suffix+'.part')
        with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
            while block:=r.read(2**20):f.write(block)
        if not valid(tmp,size,md5):raise ValueError('Source checksum mismatch: '+name)
        tmp.replace(p);print('DOWNLOADED_AND_VERIFIED',name)

if __name__=='__main__':main()
