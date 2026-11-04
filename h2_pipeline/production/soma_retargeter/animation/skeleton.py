# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import soma_retargeter.utils.pose_utils as pose_utils


class Skeleton:
    """
    Hierarchical skeleton description.
    Stores joint names, parent indices, and a reference pose (local transforms).
    """
    def __init__(self, num_joints, joint_names, parent_indices, local_transforms : np.ndarray | list):
        """
        Initialize a Skeleton object with joint hierarchy and transformation data.
        Args:
            num_joints (int): The total number of joints in the skeleton.
            joint_names (list): A list of names for each joint. Must have length equal to num_joints.
            parent_indices (array-like): An array of parent joint indices for each joint, stored as int32.
                Must have length equal to num_joints.
            local_transforms (np.ndarray | list): Local transformation matrices for the skeleton joints.
                Can be a numpy array or list. Must have length equal to num_joints along the first axis.
                Will be converted to float32 numpy array.
        Raises:
            ValueError: If the length of joint_names does not match num_joints.
            ValueError: If the length of parent_indices does not match num_joints.
            ValueError: If the length of local_transforms does not match num_joints.
        """
        self._num_joints = int(num_joints)
        self.joint_names = list(joint_names)
        self.parent_indices = np.asarray(parent_indices, dtype=np.int32)

        self.up_axis = wp.vec3(0, 0, 1)
        self.forward_axis = wp.vec3(0, -1, 0)

        if len(self.joint_names) != self._num_joints:
            raise ValueError(
                f"[ERROR]: joint_names count [{len(self.joint_names)}] "
                f"is not equal to num_joints [{self._num_joints}]"
            )

        if len(self.parent_indices) != self._num_joints:
            raise ValueError(
                f"[ERROR]: parent_indices count [{len(self.parent_indices)}] "
                f"is not equal to num_joints [{self._num_joints}]"
            )

        if isinstance(local_transforms, np.ndarray):
            if local_transforms.shape[0] != self._num_joints:
                raise ValueError(
                    f"[ERROR]: local_transforms count [{local_transforms.shape[0]}] "
                    f"is not equal to num_joints [{self._num_joints}]"
                )
            self._reference_local_transforms = np.asarray(local_transforms, dtype=np.float32, copy=True)
        else:
            if len(local_transforms) != self._num_joints:
                raise ValueError(
                    f"[ERROR]: local_transforms count [{len(local_transforms)}] "
                    f"is not equal to num_joints [{self._num_joints}]"
                )
            self._reference_local_transforms = np.asarray(local_transforms, dtype=np.float32)

    def joint_index(self, joint_name):
        """
        Get the index of a joint by its name.
        Args:
            joint_name (str): The name of the joint to find.
        Returns:
            int: The index of the joint in the skeleton's joint list. Returns -1 if the joint name is not found.
        Raises:
            RuntimeError: If the skeleton has not been initialized (num_joints is 0).
        """
        if self._num_joints == 0:
            raise RuntimeError("[ERROR]: Skeleton has not been initialized.")

        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            return -1

    def joint_name(self, index):
        """
        Retrieve the name of a joint by its index.
        Args:
            index (int): The index of the joint to retrieve. Must be in the range [0, num_joints).
        Returns:
            str: The name of the joint at the specified index.
        Raises:
            ValueError: If the index is out of valid range [0, num_joints).
        """
        if index < 0 or index >= self._num_joints:
            raise ValueError(f"[ERROR]: Joint index [{index}] should be in [0, {self._num_joints}).")

        return self.joint_names[index]

    def joint_parent(self, index):
        """
        Retrieve the parent joint index for a given joint.
        Args:
            index (int): The index of the joint whose parent is to be retrieved.
                        Must be in the range [0, num_joints).
        Returns:
            int: The index of the parent joint.
        Raises:
            ValueError: If the index is out of valid range [0, num_joints).
        """
        if index < 0 or index >= self._num_joints:
            raise ValueError(f"[ERROR]: Joint index [{index}] should be in [0, {self._num_joints}).")

        return self.parent_indices[index]

    @property
    def num_joints(self):
        """
        Get the number of joints in the skeleton.

        Returns:
            int: The total number of joints.
        """
        return self._num_joints

    @property
    def reference_local_transforms(self):
        """
        Get a copy of the reference local transforms.

        Returns a deep copy of the reference local transforms array to prevent
        external modification of the internal state.

        Returns:
            np.ndarray: A copy of the reference local transforms.
        """
        return np.copy(self._reference_local_transforms)

    def compute_global_transforms(self, local_transforms, root_tx=wp.transform_identity()):
        """
        Compute global (world-space) transforms for all joints in the skeleton hierarchy.
        Args:
            local_transforms: Per-joint local transforms in the same layout as reference_local_transforms.
            root_tx (wp.transform, optional): Root transformation matrix. Defaults to identity transform.
        Returns:
            Global transformation matrices for all joints in world space.
        """
        return pose_utils.compute_global_pose(self, local_transforms, root_tx)


class SkeletonInstance:
    """
    A poseable instance of a ``Skeleton``.

    Stores its own local transforms and world transform (xform), and
    can compute global transforms from the underlying Skeleton hierarchy.
    """
    def __init__(self, skeleton : Skeleton, color : wp.vec3, xform : wp.transform):
        """
        Initialize a ``SkeletonInstance`` object.
        Args:
            skeleton (Skeleton): Underlying ``Skeleton`` describing joints and hierarchy.
            color (wp.vec3): Color associated with this instance, e.g. for rendering.
            xform (wp.transform): World-space transform of the skeleton root for this
                                instance (position and orientation in the scene)
        """
        self.skeleton = skeleton
        self.xform = xform
        self.color = color
        self.local_transforms = skeleton.reference_local_transforms

    def reset_local_transforms(self):
        """
        Reset the local transforms to the skeleton's reference local transforms.

        This method restores all local transformation matrices to their initial
        reference state as defined in the skeleton's reference_local_transforms.
        """
        self.local_transforms = self.skeleton.reference_local_transforms

    def set_local_transforms(self, local_transforms : np.ndarray):
        """
        Set the local transforms for all joints in the skeleton.
        Args:
            local_transforms (np.ndarray): A 2D array of local transformation matrices with shape
                (num_joints, wp.transform).
        Raises:
            ValueError: If the number of rows in local_transforms does not match the number of joints
                in the skeleton (skeleton.num_joints).
        """
        if local_transforms.shape[0] != self.skeleton.num_joints:
            raise ValueError(
                f"[ERROR]: local_transforms count [{local_transforms.shape[0]}] "
                f"is not equal to skeleton.num_joints [{self.skeleton.num_joints}]"
            )

        self.local_transforms = local_transforms

    def set_local_transform(self, joint_index, local_transform):
        """
        Set the local transform for a joint in the skeleton.
        Args:
            joint_index (int): The index of the joint to set the transform for.
            local_transform: The local transformation to apply to the joint.
        Raises:
            ValueError: If joint_index is negative or greater than or equal to
                        the total number of joints in the skeleton.
        """
        if joint_index < 0 or joint_index >= self.skeleton.num_joints:
            raise ValueError(
                f"[ERROR]: joint_index [{joint_index}] is equal or greater than "
                f"skeleton.num_joints [{self.skeleton.num_joints}]"
            )

        self.local_transforms[joint_index] = local_transform

    def get_local_transforms(self):
        """
        Retrieve the local transformation matrices for all joints in the skeleton.
        Returns:
            np.ndarray: The local transformation matrices of the skeleton.
        """
        return self.local_transforms

    def get_local_transform(self, joint_index):
        """
        Retrieve the local transformation matrix for a specified joint.
        Args:
            joint_index (int): The index of the joint whose local transform is to be retrieved.
                Must be a valid index within the range [0, skeleton.num_joints).
        Returns:
            The local transformation matrix of the specified joint.
        Raises:
            ValueError: If joint_index is negative or greater than or equal to
                skeleton.num_joints.
        """
        if joint_index < 0 or joint_index >= self.skeleton.num_joints:
            raise ValueError(
                f"[ERROR]: joint_index [{joint_index}] is equal or greater than "
                f"skeleton.num_joints [{self.skeleton.num_joints}]"
            )

        return self.local_transforms[joint_index]

    def compute_global_transforms(self):
        """
        Compute the global transformation matrices for all joints in the skeleton.

        Returns:
            np.ndarray: Global transformation matrices for all joints in the skeleton.
        """
        return self.skeleton.compute_global_transforms(self.local_transforms, self.xform)

    @property
    def num_joints(self) -> int:
        """
        Get the total number of joints in the skeleton.

        Returns:
            int: The number of joints defined in this skeleton.
        """
        return self.skeleton.num_joints

    @property
    def parent_indices(self):
        """
        Get the parent indices of the skeleton joints.

        Returns:
            list: A list of integers representing the parent joint index for each joint in the skeleton.
                  The parent index indicates which joint is the parent of the current joint in the hierarchy.
                  A value of -1 indicates a root joint with no parent.
        """
        return self.skeleton.parent_indices

    @property
    def reference_local_transforms(self):
        """
        Get the reference local transforms of the skeleton.

        Returns:
            The reference local transforms of the skeleton.
        """
        return self.skeleton.reference_local_transforms
