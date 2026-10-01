"""Assign frozen feature vectors to independently fitted cells."""
import numpy as np

def nearest_cells(x,centres):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=centres.shape[1] or not np.all(np.isfinite(x)):
        raise ValueError('Feature shape/nonfinite error')
    # Chunk to limit n x cells memory. No fitting to log/evaluation inputs.
    outputs=[]
    for start in range(0,len(x),2048):
        block=x[start:start+2048]
        dist=np.sum(block*block,1)[:,None]+np.sum(centres*centres,1)[None,:]-2*block@centres.T
        outputs.append(np.argmin(dist,axis=1))
    return np.concatenate(outputs) if outputs else np.array([],dtype=int)


