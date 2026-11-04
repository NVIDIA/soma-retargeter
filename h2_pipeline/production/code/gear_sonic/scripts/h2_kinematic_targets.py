"""Robot-length arm targets from source bone directions, before IK/CSV export."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def segment_targets(targets, target_names, source_world, source_names, anatomy):
    result=np.asarray(targets).copy();source=np.asarray(source_world)
    if len(source)!=len(result) or not np.isfinite(source).all() or not np.isfinite(result).all():
        raise ValueError('Finite, frame-aligned source and target transforms required')
    hip=result[:,target_names.index('Hips')];chest=result[:,target_names.index('Chest')]
    torso=hip[:,:3]+R.from_quat(hip[:,3:]).apply(np.broadcast_to(anatomy['torso_from_pelvis'],(len(hip),3)).copy())
    chest[:,:3]=torso
    for side in ('Left','Right'):
        hp=source[:,[source_names.index(side+n) for n in ('Arm','ForeArm','Hand')],:3]
        vectors=np.diff(hp,axis=1);lengths=np.linalg.norm(vectors,axis=-1)
        target_lengths=np.asarray(anatomy[side]['lengths'])
        if np.any(lengths<1e-8) or target_lengths.shape!=(2,) or np.any(target_lengths<=0):
            raise ValueError('Nondegenerate human and robot arm segments required')
        vectors/=lengths[...,None]
        shoulder=torso+R.from_quat(chest[:,3:]).apply(np.broadcast_to(anatomy[side]['shoulder_from_torso'],(len(hip),3)).copy())
        elbow=shoulder+vectors[:,0]*target_lengths[0];wrist=elbow+vectors[:,1]*target_lengths[1]
        for name,pos in zip(('Arm','ForeArm','Hand'),(shoulder,elbow,wrist)):
            result[:,target_names.index(side+name),:3]=pos
    return result
