# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import numpy as np
import warp as wp

from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.animation.animation_buffer import AnimationBuffer, create_animation_buffer_for_skeleton


@wp.func
def wp_axis_angle_to_quat_xyzw(
    axis : wp.vec3,
    angle : wp.float32,
    eps : wp.float32 = 1e-8):
    """Convert an axis–angle rotation to a normalized quaternion [x, y, z, w]."""

    angle = wp.radians(angle)

    norm = wp.length(axis)
    unit_axis = axis / wp.max(norm, eps)

    half = angle * 0.5
    s = wp.sin(half)
    c = wp.cos(half)

    xyz = unit_axis * s
    q = wp.quat(xyz[0], xyz[1], xyz[2], c)
    return wp.normalize(wp.quat(q[0], q[1], q[2], q[3]))


@wp.func
def wp_get_quaternion_from_axis(
    axis : wp.int32,
    angle : wp.float32):
    """Build a quaternion for a rotation around the X, Y, or Z axis."""

    if axis == 0:
        return wp_axis_angle_to_quat_xyzw(wp.vec3(1.0, 0.0, 0.0), angle)
    elif axis == 1:
        return wp_axis_angle_to_quat_xyzw(wp.vec3(0.0, 1.0, 0.0), angle)
    else:
        return wp_axis_angle_to_quat_xyzw(wp.vec3(0.0, 0.0, 1.0), angle)


@wp.func
def wp_euler_to_quaternion(
    euler_angles : wp.array(dtype=wp.float32),
    rotation_order : wp.array(dtype=wp.int32),
):
    """Convert Euler angles and rotation order to a normalized quaternion."""

    quaternion = wp.quat(0.0, 0.0, 0.0, 1.0)
    for i in range(rotation_order.shape[0]):
        quaternion *= wp_get_quaternion_from_axis(rotation_order[i], euler_angles[i])

    return wp.normalize(quaternion)


@wp.kernel
def wp_convert_frame_animation(
    reference_local_transforms : wp.array(dtype=wp.transform),
    positions_exists : wp.array2d(dtype=wp.bool),
    rotations_exists : wp.array2d(dtype=wp.bool),
    animation_data_positions : wp.array3d(dtype=wp.float32),
    animation_data_rotations : wp.array3d(dtype=wp.float32),
    joint_indices : wp.array(dtype=wp.int32),
    rotate_order : wp.array2d(dtype=wp.int32),
    frame_data : wp.array2d(dtype=wp.transform),
):
    """Convert raw BVH per-frame joint data to local transforms."""

    frame, joint_index = wp.tid()

    if positions_exists[frame, joint_index]:
        positions = wp.vec3(animation_data_positions[frame, joint_index, 0], animation_data_positions[frame, joint_index, 1], animation_data_positions[frame, joint_index, 2]) * 0.01
    else:
        positions = reference_local_transforms[joint_indices[joint_index]].p

    if rotations_exists[frame, joint_index]:
        rotation = wp_euler_to_quaternion(animation_data_rotations[frame, joint_index], rotate_order[joint_index])
    else:
        rotation = reference_local_transforms[joint_indices[joint_index]].q

    frame_data[frame, joint_index] = wp.transform(positions, rotation)


def axis_angle_to_quat_xyzw(axis, angle, degrees=False, eps=1e-8):
    """Convert an axis–angle rotation to a normalized Warp quaternion [x, y, z, w]."""

    axis = np.asarray(axis, dtype=np.float32)
    angle = np.asarray(angle, dtype=np.float32)
    if degrees:
        angle = np.deg2rad(angle)

    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    unit_axis = axis / np.clip(norm, eps, None)

    half = angle * 0.5
    s = np.sin(half)[..., None]
    c = np.cos(half)[..., None]

    xyz = unit_axis * s
    q = np.concatenate([xyz, c], axis=-1)
    # normalize for numerical safety
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), eps, None)
    return wp.quat(*q.astype(np.float32).flatten())


def euler_to_quaternion(euler_angles, rotation_order):
    """Convert Euler angles and rotation order (e.g. 'XYZ') to a normalized quaternion."""

    def get_quaternion_from_axis(axis, angle):
        if axis == 'x':
            return axis_angle_to_quat_xyzw(np.array([1, 0, 0]), angle, degrees=True)
        elif axis == 'y':
            return axis_angle_to_quat_xyzw(np.array([0, 1, 0]), angle, degrees=True)
        elif axis == 'z':
            return axis_angle_to_quat_xyzw(np.array([0, 0, 1]), angle, degrees=True)
        else:
            raise ValueError(f"Unsupported axis: {axis}")

    quaternion = wp.quat(0.0, 0.0, 0.0, 1.0)
    for i, r in enumerate(rotation_order):
        quaternion *= get_quaternion_from_axis(r, euler_angles[i])

    return wp.normalize(quaternion)


def get_global_positions(skeleton, global_transforms):
    """Compute per-joint start/end positions for visualization from global transforms."""

    start_positions = np.zeros((skeleton.num_joints, 3))
    end_positions = np.zeros((skeleton.num_joints, 3))
    for joint_index in range(skeleton.num_joints):
        if skeleton.parent_indices[joint_index] != -1:
            start_positions[joint_index] = np.array(global_transforms[joint_index].p)
            end_positions[joint_index] = np.array(global_transforms[skeleton.parent_indices[joint_index]].p)

    return start_positions, end_positions


class Animation:
    """Simple animation wrapper over a Skeleton and per-frame joint transforms."""

    def __init__(self, skeleton, frame_data, frame_range):
        self.skeleton = skeleton
        # frame data is a list of transforms of shape of (num_frames, num_joints)
        self.frame_data = frame_data

        self.num_frames = frame_range[0]
        self.fps = frame_range[1]

    def get_global_transforms(self, frame_index):
        return self.skeleton.compute_global_transforms(self.frame_data[frame_index])

    def get_local_transforms(self, frame_index):
        return self.frame_data[frame_index]

    def set_local_transforms(self, frame_index, local_transforms):
        self.frame_data[frame_index] = local_transforms

    def set_local_transform(self, frame_index, joint_index, local_transform):
        self.frame_data[frame_index][joint_index] = local_transform

    def get_global_positions(self, frame_index):
        global_transforms = self.get_global_transforms(frame_index)

        return get_global_positions(self.skeleton, global_transforms)


class BVHJoint(object):
    """
    Node in a BVH joint hierarchy, holding offsets, channels, and animation data.

    References:
        http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf
        https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
    """

    def __init__(self, name):
        self.name = name
        self._parent = None
        self._path = None
        self._children = []
        self._offset = []
        self._translate = []
        self._rotate = []
        self._channels = []
        self._animation = []

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        if parent:
            parent._children.append(self)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def children(self):
        return self._children

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels):
        self._channels = channels

    @property
    def channel_number(self):
        return len(self._channels)

    @property
    def animation(self):
        return self._animation

    @property
    def frames(self):
        return len(self._animation)

    @property
    def frame_time(self):
        return self._frame_time

    @frame_time.setter
    def frame_time(self, frame_time):
        self._frame_time = frame_time

    @property
    def rotate_order(self):
        rotateChannels = [channel[0].lower() for channel in self.channels if 'rotation' in channel]
        return ''.join(rotateChannels)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    def add_child(self, child):
        self._children.append(child)
        child.parent = self

    def add_frame_animation(self, data):
        self._animation.append(data)


class BVHImporter(object):
    """Helper for parsing BVH files into Skeletons and AnimationBuffers."""

    @classmethod
    def bvh_parser(cls, file_path, remove_namespace=True):
        """Parse a BVH file and return the root BVHJoint."""

        if not os.path.exists(file_path) or os.path.splitext(file_path)[-1].lower() != '.bvh':
            raise ValueError('Invalid BVH file path: {}'.format(file_path))

        # read bvh data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            f.close()

        # parse joint hierarchy data
        joints = []                     # store all the joints
        jointWalker = []                # joint pointer, the last joint will be the parent of current joint
        ignoreNextBrackets = False      # bypass the next curly brackets

        motionLine = -1                 # start line number of motion data
        for i, line in enumerate(data):
            token = line.split()
            if not token:
                continue
            # Root
            if token[0] == 'ROOT':
                joint_name = token[1]
                if remove_namespace and ':' in joint_name:
                    joint_name = joint_name.split(':')[-1]
                joint = BVHJoint(joint_name)
                joint.path = '/{}'.format(joint_name)
                jointWalker.append(joint)
                joints.append(joint)
            # Joint
            elif token[0] == 'JOINT':
                joint_name = token[1]
                if remove_namespace and ':' in joint_name:
                    joint_name = joint_name.split(':')[-1]
                joint = BVHJoint(joint_name)
                joint.parent = jointWalker[-1]
                joint.path = '{}/{}'.format(joint.parent.path, joint_name)
                jointWalker.append(joint)
                joints.append(joint)
            # End Site
            elif line.strip() == 'End Site':
                ignoreNextBrackets = True
            # Offset
            elif token[0] == 'OFFSET':
                if not ignoreNextBrackets:
                    joint.offset = [float(strVal) for strVal in token[1:]]
            # channels
            elif token[0] == 'CHANNELS':
                joint.channels = [strVal for strVal in token[2:]]
            # }
            elif token[0] == '}':
                if ignoreNextBrackets:
                    ignoreNextBrackets = False
                    continue
                else:
                    jointWalker.pop(-1)
            elif token[0] == 'MOTION':
                motionLine = i
                break

        # parse motion data
        frametime = 0
        for i, line in enumerate(data[motionLine:]):
            token = line.split()
            if not token:
                continue

            # Motion
            if token[0] in ['MOTION', 'Frames:']:
                continue
            elif ' '.join(token[:2]) == 'Frame Time:':
                frametime = float(token[-1])
            else:
                data = [float(strVal) for strVal in token]
                start = 0
                end = 0
                for jnt in joints:
                    end += jnt.channel_number
                    jnt.add_frame_animation(data[start:end])
                    jnt.frame_time = frametime
                    start += jnt.channel_number

        return joints[0]

    @classmethod
    def get_rotation_order(cls, channels):
        """Return rotation order string (e.g. 'xyz') extracted from BVH channels."""

        rotationOrder = ''
        # find rotation in the channel
        for channel in channels:
            if 'rotation' in channel:
                rotationOrder += channel[0].lower()

        return rotationOrder

    def create_skeleton(self, bvh_path):
        """Create a Skeleton from a BVH file and return it with the root BVHJoint."""

        #check if bvh file exists
        if not os.path.exists(bvh_path):
            raise ValueError(f"BVH file not found: {bvh_path}")

        rootBVHJoint_obj = BVHImporter.bvh_parser(bvh_path)
        rig_data = BVHImporter.construct_skeleton(rootBVHJoint_obj)

        num_joints = len(rig_data)
        local_transforms = [wp.transform() for _ in range(num_joints)]
        joint_names = [rig_data[i]['name'] for i in range(num_joints)]
        rotate_order = [rig_data[i]['rotate_order'] for i in range(num_joints)]

        name_to_index = {joint['name']: idx for idx, joint in enumerate(rig_data)}

        # Now, for each joint, get the index of its parent (or -1 if no parent)
        parent_indices = [
            name_to_index[rig_data[i]['parent']] if rig_data[i]['parent'] in name_to_index else -1
            for i in range(num_joints)
        ]

        for i in range(num_joints):
            rotation = euler_to_quaternion(rig_data[i]['rotation'], rotate_order[i])
            position = wp.vec3(rig_data[i]['translation']) * 0.01
            local_transforms[i] = wp.transform(position, rotation)

        return Skeleton(num_joints, joint_names, parent_indices, local_transforms), rootBVHJoint_obj

    @classmethod
    def construct_skeleton(cls, BVHJoint_obj, parent=None):
        """Convert a BVH joint hierarchy into a flat rig description list."""

        rig_data = []
        # convert this to quaternion
        rig_data.append({
            'name': BVHJoint_obj.name,
            'parent': parent.name if parent else None,
            'translation': BVHJoint_obj.offset[:3],
            'rotation': [0.0, 0.0, 0.0],
            'rotate_order': cls.get_rotation_order(BVHJoint_obj.channels)
        })

        for child in BVHJoint_obj.children:
            rig_data.extend(cls.construct_skeleton(BVHJoint_obj=child, parent=child.parent))
        return rig_data

    @staticmethod
    def get_frame_range(bvh_joint_obj):
        """Return (num_frames, fps) from a fully parsed BVHJoint tree."""

        return (bvh_joint_obj.frames, round(1/bvh_joint_obj.frame_time, 2))

    @classmethod
    def create_animation(cls, BVHJoint, skeleton):
        """Create an AnimationBuffer for the given skeleton from BVH motion data."""

        frame_data = cls.load_animation(BVHJoint, skeleton)
        frame_range = cls.get_frame_range(BVHJoint)
        animation_buffer = AnimationBuffer(skeleton, frame_range[0], frame_range[1], frame_data)
        return animation_buffer

    @classmethod
    def load_animation(cls, BVHJoint, skeleton):
        """Load all BVH frames and convert them to Warp transforms using a GPU kernel."""

        if not BVHJoint:
            raise ValueError("[ERROR]: BVHJoint data is None, check the BVH file path!")

        cls.animation_load_time = 0.0
        cls.animation_convert_time = 0.0

        # Load animation
        actual_frame_range = cls.get_frame_range(BVHJoint)

        frame_data = []

        positions_array = []
        rotations_array = []
        joint_indices = []
        rotate_orders = []

        start_time = time.time()
        for frame in range(actual_frame_range[0]):
            positions_array.append([])
            rotations_array.append([])
            cls.load_frame_animation_data(skeleton, BVHJoint, frame, positions_array[-1], rotations_array[-1], joint_indices, rotate_orders)
        end_time = time.time()
        cls.animation_load_time += end_time - start_time

        start_time = time.time()
        positions_exists = [len(positions_array[frame][joint_index]) > 0 for frame in range(len(positions_array)) for joint_index in range(len(joint_indices))]
        positions_exists = np.array(positions_exists).reshape(len(positions_array), len(joint_indices))
        rotations_exists = [len(rotations_array[frame][joint_index]) > 0 for frame in range(len(rotations_array)) for joint_index in range(len(joint_indices))]
        rotations_exists = np.array(rotations_exists).reshape(len(rotations_array), len(joint_indices))

        frame_data_wp = wp.empty(shape=(actual_frame_range[0], len(joint_indices)), dtype=wp.transform)
        rotate_order_np = np.zeros((len(joint_indices), 3), dtype=np.int32)
        for i, rotate_order in enumerate(rotate_orders):
            for a in range(3):
                rotate_order_np[i][a] = 0 if rotate_order[a] == 'x' else 1 if rotate_order[a] == 'y' else 2

        reference_local_transforms_wp = wp.array(skeleton.reference_local_transforms, dtype=wp.transform)
        positions_array_np = np.zeros((len(positions_array), len(joint_indices), 3), dtype=np.float32)
        rotations_array_np = np.zeros((len(rotations_array), len(joint_indices), 3), dtype=np.float32)

        for frame in range(len(positions_array)):
            for joint_index in range(len(joint_indices)):
                if len(positions_array[frame][joint_index]) > 0:
                    positions_array_np[frame][joint_index] = positions_array[frame][joint_index]

                if len(rotations_array[frame][joint_index]) > 0:
                    rotations_array_np[frame][joint_index] = rotations_array[frame][joint_index]

        wp.launch(
            wp_convert_frame_animation,
            dim=[len(frame_data_wp), len(joint_indices)],
            inputs=[
                reference_local_transforms_wp,
                wp.array2d(positions_exists, dtype=wp.bool),
                wp.array2d(rotations_exists, dtype=wp.bool),
                wp.array3d(positions_array_np, dtype=wp.float32),
                wp.array3d(rotations_array_np, dtype=wp.float32),
                wp.array(joint_indices, dtype=wp.int32),
                wp.array2d(rotate_order_np, dtype=wp.int32)],
            outputs=[frame_data_wp])
        frame_data = frame_data_wp.numpy()

        end_time = time.time()
        cls.animation_convert_time += end_time - start_time

        return frame_data

    @classmethod
    def load_frame_animation_data(cls, skeleton, BVHJoint, frame, positions_array, rotations_array, joint_indices, rotate_orders):
        """Collect raw position/rotation channel values for one BVH joint at one frame."""

        positions_array.append([])
        rotations_array.append([])

        positions = positions_array[-1]
        rotations = rotations_array[-1]
        for i, channel in enumerate(BVHJoint.channels):
            if 'position' in channel:
                positions.append(BVHJoint.animation[frame][i])
            elif 'rotation' in channel:
                rotations.append(BVHJoint.animation[frame][i])

        #find index of joint with BVHJoint.name in skeleton.joint_names
        if frame == 0:
            joint_indices.append(skeleton.joint_names.index(BVHJoint.name))
            rotate_orders.append(BVHJoint.rotate_order)

        for child in BVHJoint.children:
            cls.load_frame_animation_data(skeleton, child, frame, positions_array, rotations_array, joint_indices, rotate_orders)


def load_bvh(bvh_file: str, input_skeleton=None):
    """
    Load a BVH animation file and create ``Skeleton`` and ``AnimationBuffer`` objects.

    Args:
        bvh_file (str): Path to the BVH file to load.
        input_skeleton (optional): An existing skeleton to conform the loaded animation to.
            If provided, the loaded animation will be adapted to match this skeleton's structure.
            Defaults to None.

    Returns:
        tuple (Skeleton, AnimationBuffer):
            - The ``Skeleton`` object (either created from the BVH file or the input_skeleton).
            - The ``AnimationBuffer`` object containing frame data and sample rate information.
    """
    importer = BVHImporter()
    skeleton, rootJnt_obj = importer.create_skeleton(bvh_file)
    animation = importer.create_animation(rootJnt_obj, skeleton)
    print(
        f"[INFO]: Loaded BVH file [{bvh_file}] with "
        f"{animation.num_frames} frames @ {animation.sample_rate} FPS in "
        f"{(importer.animation_load_time + importer.animation_convert_time):.2f}s")

    if input_skeleton is not None:
        # conform to input skeleton
        new_animation = create_animation_buffer_for_skeleton(animation, input_skeleton)
        return input_skeleton, new_animation
    else:
        return skeleton, animation
