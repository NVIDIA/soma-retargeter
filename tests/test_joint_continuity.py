# SPDX-FileCopyrightText: Copyright (c) 2026 gohiroaki
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from soma_retargeter.pipelines.joint_continuity import IKJointContinuity
from soma_retargeter.pipelines.ik_pass import BasePass


class JointContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device('cpu')
        # A free root makes q and qd indices different; a fixed body exercises
        # the distinction between body and joint indexing.
        xml = '''<mujoco><worldbody><body name="root"><freejoint name="root_joint"/>
        <geom type="sphere" size=".1" mass="1"/><body name="fixed" pos="0 0 .3">
        <geom type="sphere" size=".1" mass="1"/><body name="arm">
        <joint name="arm_joint" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".05" fromto="0 0 0 0 0 .3" mass="1"/>
        </body></body></body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.xml'
            path.write_text(xml)
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(path))
            cls.model = builder.finalize()

    def objective(self, weight=2):
        obj = IKJointContinuity(self.model, 2, {'arm_joint': weight})
        obj.bind_device('cpu')
        obj.set_batch_layout(3, 1, 4)
        obj.init_buffers(self.model, ik.IKJacobianType.ANALYTIC)
        return obj

    def residual(self, obj, values, mapping):
        out = wp.zeros((len(values), 3), dtype=wp.float32)
        obj.compute_residuals(None, wp.array(values, dtype=wp.float32), self.model,
                              out, 1, wp.array(mapping, dtype=wp.int32))
        return out.numpy()

    def test_expanded_batch_and_exact_jacobian(self):
        obj = self.objective()
        coord, dof = int(obj._coords[0]), int(obj._dofs[0])
        self.assertNotEqual(coord, dof)
        previous = np.zeros((2, self.model.joint_coord_count), np.float32)
        previous[:, coord] = [.1, -.2]
        obj.update_reference(wp.array(previous, dtype=wp.float32))
        values = np.zeros((4, self.model.joint_coord_count), np.float32)
        values[:, coord] = [.3, .4, .5, .6]
        mapping = [1, 0, 1, 0]
        residual = self.residual(obj, values, mapping)
        np.testing.assert_allclose(residual[:, 1], 2*(values[:, coord]-previous[mapping, coord]))
        jac = wp.full((4, 3, self.model.joint_dof_count), 99., dtype=wp.float32)
        obj.compute_jacobian_analytic(None, wp.array(values, dtype=wp.float32),
                                     self.model, jac, None, 1)
        expected = np.zeros((4, self.model.joint_dof_count)); expected[:, dof] = 2
        np.testing.assert_allclose(jac.numpy()[:, 1], expected)
        plus = values.copy(); minus = values.copy()
        plus[:, coord] += 1e-3; minus[:, coord] -= 1e-3
        fd = (self.residual(obj, plus, mapping)-self.residual(obj, minus, mapping))/(2e-3)
        np.testing.assert_allclose(fd[:, 1], expected[:, dof], atol=1e-4)

    def test_frozen_reference_and_disable(self):
        obj = self.objective()
        q = wp.ones((2, self.model.joint_coord_count), dtype=wp.float32)
        obj.update_reference(q)
        q.fill_(3.)
        np.testing.assert_allclose(obj.previous_q.numpy(), 1.)
        obj.update_reference(q, enabled=False)
        self.assertTrue(np.all(self.residual(obj, np.zeros(q.shape, np.float32), [0, 1]) == 0))

    def test_invalid_selection(self):
        for weights in ({}, {'missing': 2}, {'root_joint': 2}, {'arm_joint': -1},
                        {'arm_joint': float('nan')}, {'arm_joint': float('inf')}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                IKJointContinuity(self.model, 1, weights)

    def test_solver_balances_tracking_and_previous_frame(self):
        target = IKJointContinuity(self.model, 1, {'arm_joint': 1})
        prior = IKJointContinuity(self.model, 1, {'arm_joint': 2})
        solver = ik.IKSolver(self.model, n_problems=1, objectives=[target, prior],
                             jacobian_mode=ik.IKJacobianType.ANALYTIC)
        q = wp.array(self.model.joint_q.numpy()[None], dtype=wp.float32)
        reference = q.numpy().copy(); reference[0, target._coords[0]] = .8
        target.update_reference(wp.array(reference, dtype=wp.float32))
        prior.update_reference(q)
        solver.step(q, q, iterations=24)
        # min (q-.8)^2 + (2*q)^2 -> q=.16, not .8 or the old pose 0.
        self.assertAlmostEqual(float(q.numpy()[0, target._coords[0]]), .16, places=3)

    def test_base_pass_reference_updates_once_per_frame_after_warmup(self):
        obj = self.objective()
        q = wp.zeros((2, self.model.joint_coord_count), dtype=wp.float32)
        calls = []
        def solve(state, iterations):
            calls.append((obj.previous_q.numpy().copy(), obj.scale.numpy()[0]))
            q.fill_(len(calls))
        solver = SimpleNamespace(joint_q=q, solve=solve)
        pass_ = BasePass(solver, wp.clone(q), 2, [np.empty((4, 0, 7), np.float32)]*2,
                        [], [], SimpleNamespace(set_weight=lambda w: None), 0, 2, 24, None,
                        continuity_objective=obj)
        for frame in range(4):
            pass_.execute(frame)
        self.assertEqual([scale for _, scale in calls], [0, 0, 0, 1])
        np.testing.assert_allclose(calls[-1][0], 3)
        # solve changed q to 4; the frozen reference is still 3.
        np.testing.assert_allclose(obj.previous_q.numpy(), 3)


if __name__ == '__main__':
    unittest.main()
