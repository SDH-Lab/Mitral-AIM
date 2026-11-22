"""64^3, every original frame, clip-consistent intensity and spatial mapping."""
import argparse,json,re,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F
import pandas as pd

VERSION='64_all_frames_clip_consistent_v1'
def spatial_resize(array):
    return F.interpolate(torch.from_numpy(array.astype(np.float32))[None,None],size=(64,64,64),mode='trilinear',align_corners=False)[0,0].numpy()
def normalize_clip(x):
    positive=x[x>0]
    scale=float(np.quantile(positive,.99)) if positive.size else 1.
    scale=max(scale,1e-6)
    return np.clip(x/scale,0,1),scale

def prepare(task):
    root,name,out,fallback=task;torch.set_num_threads(1);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    paths=sorted((Path(root)/name).glob('frame_*.mha'),key=lambda p:int(re.fullmatch(r'frame_(\d+)\.mha',p.name).group(1)))
    assert len(paths)>=2
    indices=[int(re.fullmatch(r'frame_(\d+)\.mha',p.name).group(1)) for p in paths]
    assert len(indices)==len(set(indices))
    volumes=[];geometry=None
    for path in paths:
        source=Path(fallback.get(str(path.relative_to(root)),str(path)))
        image=sitk.ReadImage(str(source))
        geo=(image.GetSize(),image.GetSpacing(),image.GetOrigin(),image.GetDirection())
        if geometry is None:geometry=geo
        else:
            assert all(np.allclose(a,b,rtol=0,atol=1e-6) for a,b in zip(geometry,geo)), 'Frame geometry differs: require explicit common physical resampling'
        array=sitk.GetArrayFromImage(image);assert np.isfinite(array).all()
        volumes.append(spatial_resize(array))
    x=np.stack(volumes);raw_mean=x.mean((1,2,3));raw_delta=np.abs(np.diff(x,axis=0)).mean((1,2,3))
    x,scale=normalize_clip(x)
    result=dict(file_name=name,frames=len(paths),original_frame_indices=indices,geometry=geometry,clip_positive_q99=scale,
                frame_mean_before=raw_mean.tolist(),frame_mean_after=x.mean((1,2,3)).tolist(),
                adjacent_abs_change_before=raw_delta.tolist(),adjacent_abs_change_after=np.abs(np.diff(x,axis=0)).mean((1,2,3)).tolist(),
                temporal_smoothing=False,clipped_fraction=float(np.mean(x>=1)))
    np.save(Path(out)/(name+'.npy'),x[:,None].astype(np.float16))
    return result

def load_fallback(path):
    fallback = {}
    if not path:
        return fallback
    for record in json.loads(Path(path).read_text()):
        if not record.get("verified"):
            raise ValueError("fallback entry is not marked verified")
        backup = Path(record["backup"])
        digest = hashlib.sha256(backup.read_bytes()).hexdigest()
        if digest != record["backup_sha256"]:
            raise ValueError(f"fallback hash mismatch for {record['relative']}")
        fallback[record["relative"]] = str(backup)
    return fallback

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='CSV with patient_id, file_name, fold, binary')
    p.add_argument('--root', required=True, help='Directory of per-clip folders containing frame_*.mha')
    p.add_argument('--fallback', default='', help='Optional JSON of verified replacement frames. Leave empty when every frame is readable from --root.')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False);fallback=load_fallback(a.fallback)
    df=pd.read_csv(a.manifest);records=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for i,r in enumerate(pool.map(prepare,[(a.root,n,a.out,fallback) for n in df.file_name])):
            records.append(r)
            if i%25==0:print('completed',i+1,'frames',sum(q['frames'] for q in records),flush=True)
    fallback_sha256 = hashlib.sha256(Path(a.fallback).read_bytes()).hexdigest() if a.fallback else ""
    manifest=dict(version=VERSION,config=vars(a),clips=records,manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),fallback_sha256=fallback_sha256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),spatial_mapping='Shared index-center mapping trilinear align_corners=False; all frame geometry asserted identical',intensity='One positive q99 over all resized voxels of entire clip; shared clipping [0,1]',frame_policy='ALL original frames, numeric order, no temporal interpolation or smoothing')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',len(records),sum(r['frames'] for r in records),flush=True)
if __name__=='__main__':main()
