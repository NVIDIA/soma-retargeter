"""Pose-conditioned correction retrieval. Retrieved values are proposals only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def yaw(q):
    matrix=R.from_quat(np.asarray(q)[:,[4,5,6,3]]).as_matrix()
    return np.arctan2(matrix[:,1,0],matrix[:,0,0])

def local_xy(v,heading,inverse=False):
    v=np.asarray(v);c=np.cos(heading);s=np.sin(heading)
    if inverse:s=-s
    result=v.copy();result[...,0]=v[...,0]*c-v[...,1]*s;result[...,1]=v[...,0]*s+v[...,1]*c
    return result

def foot_features(q,legs,actual_position,actual_rotation,goal_position,goal_rotation):
    heading=yaw(q);error=goal_position-actual_position
    error=local_xy(error,heading[:,None],inverse=True)
    rotation=(R.from_matrix(goal_rotation.reshape(-1,3,3)).inv()*R.from_matrix(actual_rotation.reshape(-1,3,3))).as_rotvec().reshape(len(q),6)
    angles=R.from_quat(q[:,[4,5,6,3]]).as_euler('xyz')[:,:2]
    return np.c_[q[:,7+legs]/.2,np.sin(angles),np.cos(angles),q[:,2]/.1,error.reshape(len(q),6)/.03,rotation/.3]

def arm_features(q,joints,torso_rotation,desired_directions,actual_wrist,desired_wrist):
    local=np.einsum('nij,nkj->nki',torso_rotation.transpose(0,2,1),desired_directions)
    error=(R.from_matrix(desired_wrist).inv()*R.from_matrix(actual_wrist)).as_rotvec()
    return np.c_[q[:,7+joints]/.35,local.reshape(len(q),6),error/.5]

class Bank:
    def __init__(self,path,verify_hash=True):
        self.path=Path(path);self.meta=json.loads(self.path.with_suffix('.json').read_text())
        if verify_hash:assert sha(self.path)==self.meta['bank_sha256'],'Solution bank hash mismatch'
        else:assert (self.path.stat().st_size,self.path.stat().st_mtime_ns)==(self.meta['bank_bytes'],self.meta['bank_mtime_ns'])
        self.arrays={k:v for k,v in np.load(self.path,allow_pickle=False).items()}
        self.bases=np.array([k.removesuffix('_M') for k in self.meta['donor_keys']])
        self.trees={};self.indices={};self.projections={};self.neighbor_cache={}
        for state in range(4):
            ids=np.flatnonzero(self.arrays['contact']==state)
            if len(ids):self.add_tree(('lower',state),ids,self.arrays['lower_features'][ids])
        for side in ('left','right'):
            ids=np.flatnonzero(np.max(abs(self.arrays[side+'_delta']),axis=1)>np.deg2rad(.005))
            if len(ids):self.add_tree((side,None),ids,self.arrays[side+'_features'][ids])

    def add_tree(self,token,ids,features):
        sample=features[np.linspace(0,len(features)-1,min(len(features),8192)).astype(int)].astype(float)
        center=sample.mean(0);_,_,vectors=np.linalg.svd(sample-center,full_matrices=False)
        projection=vectors[:min(8,features.shape[1])].T
        self.projections[token]=(center,projection)
        self.trees[token]=cKDTree((features-center)@projection);self.indices[token]=ids

    def query(self,kind,features,key,contact=None,k=8,absolute=False):
        features=np.asarray(features);result=np.zeros((len(features),15 if kind=='lower' else 7))
        distances=np.full(len(features),np.inf);owners=np.full(len(features),-1,dtype=int)
        categories=range(4) if kind=='lower' else [None]
        for category in categories:
            wanted=np.flatnonzero(contact==category) if kind=='lower' else np.arange(len(features))
            tree=self.trees.get((kind,category))
            if tree is None or not len(wanted):continue
            center,projection=self.projections[(kind,category)]
            signature=(kind,category,key,hashlib.sha256(features[wanted].tobytes()).hexdigest())
            cached=self.neighbor_cache.get(signature)
            if cached is None:
                count=min(256,tree.n)
                _,j=tree.query((features[wanted]-center)@projection,k=count,workers=1)
                if count==1:j=j[:,None]
                ids=self.indices[(kind,category)][j]
                field='lower_features' if kind=='lower' else kind+'_features'
                d=np.linalg.norm(self.arrays[field][ids]-features[wanted,None,:],axis=2)
                self.neighbor_cache[signature]=(d,ids)
            else:d,ids=cached
            count=ids.shape[1]
            owner=self.arrays['owner'][ids]
            # Both the same motion and its mirror are excluded, including in benchmarks.
            d=np.where(self.bases[owner]==key.removesuffix('_M'),np.inf,d)
            chosen=np.argsort(d,axis=1)[:,:min(k,count)]
            d=np.take_along_axis(d,chosen,axis=1);ids=np.take_along_axis(ids,chosen,axis=1)
            good=np.isfinite(d);weights=np.where(good,1/(d*d+.05),0.)
            weights/=np.maximum(weights.sum(1,keepdims=True),1e-30)
            field='lower_delta' if kind=='lower' else kind+'_delta'
            values=self.arrays[field][ids]
            if absolute:
                values=values.copy()
                if kind=='lower':
                    values[...,3:]+=self.arrays['lower_features'][ids][...,:12]*.2
                    values[...,2]+=self.arrays['lower_features'][ids][...,16]*.1
                    values[...,:2]=0.
                else:values+=self.arrays[kind+'_features'][ids][...,:7]*.35
            result[wanted]=np.sum(values*weights[...,None],axis=1)
            distances[wanted]=np.min(d,axis=1)
            best=np.argmin(d,axis=1);best_ids=ids[np.arange(len(wanted)),best]
            owners[wanted]=np.where(np.isfinite(distances[wanted]),self.arrays['owner'][best_ids],-1)
        return result,distances,owners
