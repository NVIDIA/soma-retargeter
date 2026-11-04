# SPDX-License-Identifier: Apache-2.0
"""Batched target updates and chunked result downloads; no IK math changes."""
import numpy as np
import warp as wp


@wp.kernel
def set_frame_targets(targets: wp.array2d(dtype=wp.transform),
                      offsets: wp.array(dtype=wp.int32), lengths: wp.array(dtype=wp.int32),
                      frame: int, effector: int,
                      positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.vec4)):
    env=wp.tid()
    tx=targets[offsets[env]+wp.min(frame,lengths[env]-1),effector]
    positions[env]=wp.transform_get_translation(tx)
    q=wp.transform_get_rotation(tx)
    rotations[env]=wp.vec4(q[0],q[1],q[2],q[3])


@wp.kernel
def set_foot_targets(targets: wp.array2d(dtype=wp.transform),
                     offsets: wp.array(dtype=wp.int32), lengths: wp.array(dtype=wp.int32),
                     frame: int, indices: wp.array(dtype=wp.int32),
                     feet: wp.array2d(dtype=wp.transform)):
    env,foot=wp.tid()
    feet[env,foot]=targets[offsets[env]+wp.min(frame,lengths[env]-1),indices[foot]]


@wp.kernel
def split_effector_targets(targets: wp.array2d(dtype=wp.transform), effector: int,
                           positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.vec4)):
    env=wp.tid()
    tx=targets[env,effector]
    positions[env]=wp.transform_get_translation(tx)
    q=wp.transform_get_rotation(tx)
    rotations[env]=wp.vec4(q[0],q[1],q[2],q[3])


@wp.kernel
def record_frame(q: wp.array2d(dtype=wp.float32), buffer: wp.array3d(dtype=wp.float32), slot: int):
    env,coord=wp.tid()
    buffer[slot,env,coord]=q[env,coord]


class TargetStream:
    def __init__(self, motions, feet_indices):
        self.lengths=np.array([len(m) for m in motions],dtype=np.int32)
        offsets=np.r_[0,np.cumsum(self.lengths[:-1],dtype=np.int64)]
        if int(self.lengths.sum(dtype=np.int64))>=2**31:raise ValueError('Target index overflow')
        self.targets=wp.array(np.concatenate(motions).astype(np.float32),dtype=wp.transform)
        self.offsets=wp.array(offsets.astype(np.int32),dtype=wp.int32)
        self.device_lengths=wp.array(self.lengths,dtype=wp.int32)
        self.feet_indices=wp.array(feet_indices,dtype=wp.int32)
        self.feet=wp.empty((len(motions),len(feet_indices)),dtype=wp.transform)

    def update(self,frame,position_objectives,rotation_objectives,with_feet):
        for i,(position,rotation) in enumerate(zip(position_objectives,rotation_objectives)):
            wp.launch(set_frame_targets,dim=len(self.lengths),inputs=[self.targets,self.offsets,self.device_lengths,frame,i,position.target_positions,rotation.target_rotations])
        if with_feet:
            wp.launch(set_foot_targets,dim=self.feet.shape,inputs=[self.targets,self.offsets,self.device_lengths,frame,self.feet_indices,self.feet])
