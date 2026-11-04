"""CUDA float64 FK/Jacobians for the existing constrained reconstruction.

The SQP constraints/OSQP solves and independent MuJoCo admission are unchanged.
Only optimizer-side geometric evaluations are replaced with batched CUDA math.
"""
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class Geometry:
    def __init__(self, batch, device):
        from gear_sonic.scripts.h2_gpu_reference_ik import GPUReferenceIK
        from gear_sonic.scripts.h2_waist_gpu_ik import skew, log_rotation
        self.batch = batch
        self.skew = skew
        self.log = log_rotation
        self.s = GPUReferenceIK(batch.BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml', device)
        self.device = device
        self.t = self.s.t
        self.calls = 0
        self.seconds = 0.

    def poses(self, q, bodies, joints=None):
        q = np.asarray(q)
        rr = self.t(R.from_quat(q[:, [4, 5, 6, 3]]).as_matrix())
        p, r, axes, pivots = self.s.fk(self.t(q[:, :3]), rr, self.t(q[:, 7:]))
        if joints is None:
            return p[:, bodies], r[:, bodies]
        jp, jr = self.s.body_jac(p, axes, pivots, bodies, joints)
        return p[:, bodies], r[:, bodies], jp, jr

    def orientation(self, actual, target, jr):
        e = self.log(target.transpose(-1, -2) @ actual)
        if jr is None:
            return e, None
        k = self.skew(e)
        theta = e.norm(dim=-1)
        # Same inverse left Jacobian as native_projection.log_jac.
        safe = theta.clamp(min=1e-12)
        c = (1 - .5*safe/torch.tan(.5*safe)) / safe.square()
        c = torch.where(theta < 1e-4, torch.full_like(c, 1/12), c)
        inverse = self.s.eye - .5*k + c[..., None, None]*(k@k)
        return e, inverse @ target.transpose(-1, -2) @ jr

    def upper(self, p, x, jac=False):
        begin = time.perf_counter()
        U = self.batch.U
        q = p.assemble(x)
        metrics = {}
        bodies = [b for side in ('Left', 'Right') for b in p.arm_bodies[side]] + [p.torso]
        deg = 180/np.pi
        for rate, ref in p.refs.items():
            qs = U.sample(q, ref['tt'])
            if jac:
                pos, rot, jp, jr = self.poses(qs, bodies, p.js.tolist())
            else:
                pos, rot = self.poses(qs, bodies)
                jp = jr = None
            values, grads = {}, {}

            def put(section, side, metric, v, g=None):
                key = section, side, metric
                values[key] = v
                if jac:
                    grads[key] = g

            for si, side in enumerate(('Left', 'Right')):
                ids = slice(si*3, si*3+3)
                vector = pos[:, ids][:, 1:] - pos[:, ids][:, :-1]
                length = vector.norm(dim=-1)
                unit = vector / length[..., None]
                human = self.t(ref['arms'][side]['dirs'])
                if jac:
                    dv = jp[:, ids][:, 1:] - jp[:, ids][:, :-1]
                    dl = (unit[..., None, :] @ dv).squeeze(-2)
                    du = ((self.s.eye-unit[..., :, None]*unit[..., None, :])/length[..., None, None]) @ dv
                for i, name in enumerate(('upper_direction_error_deg', 'forearm_direction_error_deg')):
                    dot = (unit[:, i]*human[:, i]).sum(-1).clamp(-1, 1)
                    angle = torch.acos(dot)
                    grad = -(human[:, i, None, :]@du[:, i]).squeeze(-2)/torch.sqrt((1-dot.square()).clamp(min=0)).clamp(min=1e-8)[:, None] if jac else None
                    put('arms', side, name, angle*deg, grad*deg if jac else None)
                    for axis in range(3):
                        put('aux', side, f'direction_{i}_{axis}', (unit[:, i, axis]-human[:, i, axis])*deg,
                            du[:, i, axis]*deg if jac else None)
                dot = (unit[:, 0]*unit[:, 1]).sum(-1).clamp(-1, 1)
                flex = torch.acos(dot)
                df = flex-self.t(ref['arms'][side]['flex'])
                if jac:
                    gf = -((unit[:, 1, None, :]@du[:, 0]).squeeze(-2)+(unit[:, 0, None, :]@du[:, 1]).squeeze(-2))/torch.sqrt((1-dot.square()).clamp(min=0)).clamp(min=1e-8)[:, None]
                for name in ('elbow_flex_error_deg', 'elbow_flex_error_when_human_straight_deg'):
                    put('arms', side, name, df.abs()*deg, df.sign()[:, None]*gf*deg if jac else None)
                put('aux', side, 'flex_signed', df*deg, gf*deg if jac else None)
                err = vector.sum(1)-(human*length[..., None]).sum(1)
                norm = err.norm(dim=-1)
                total = length.sum(1)
                if jac:
                    je = dv.sum(1)-(human[..., :, None]*dl[..., None, :]).sum(1)
                    gh = ((err/norm.clamp(min=1e-12)[:, None])[:, None, :]@je).squeeze(-2)/total[:, None]-norm[:, None]/total[:, None].square()*dl.sum(1)
                put('arms', side, 'normalized_hand_error', norm/total, gh if jac else None)
                index = si*3+2
                target = self.t(ref['arms'][side]['wrist'])
                e, ej = self.orientation(rot[:, index], target, jr[:, index] if jac else None)
                theta = e.norm(dim=-1)
                if jac:
                    # Norm gradient uses the rotation axis; inverse-Jacobian
                    # factors cancel along that axis, matching the CPU formula.
                    axis = e/theta.clamp(min=1e-12)[:, None]
                    gw = (axis[:, None, :]@target.transpose(-1, -2)@jr[:, index]).squeeze(-2)*deg
                put('arms', side, 'wrist_orientation_error_deg', theta*deg, gw if jac else None)
                for axis in range(3):
                    put('aux', side, f'wrist_log_{axis}', e[:, axis]*deg, ej[:, axis]*deg if jac else None)
            matrix = rot[:, 6]
            up = matrix[:, :, 2]
            left = self.t(ref['left'])
            target = self.t(ref['torso'])
            a = (up*left).sum(-1)
            b = up[:, 2]
            expected = torch.atan2((target[:, :, 2]*left).sum(-1), target[:, 2, 2])
            lateral = torch.atan2(a, b)-expected
            if jac:
                ju = -self.skew(up)@jr[:, 6]
                gl = lateral.sign()[:, None]*(b[:, None]*(left[:, None, :]@ju).squeeze(-2)-a[:, None]*ju[:, 2])/(a.square()+b.square()).clamp(min=1e-12)[:, None]*deg
            put('torso', '', 'torso_lateral_error_deg', lateral.abs()*deg, gl if jac else None)
            e, ej = self.orientation(matrix, target, jr[:, 6] if jac else None)
            theta = e.norm(dim=-1)
            if jac:
                gt = ((e/theta.clamp(min=1e-12)[:, None])[:, None, :]@target.transpose(-1, -2)@jr[:, 6]).squeeze(-2)*deg
            put('torso', '', 'torso_orientation_error_deg', theta*deg, gt if jac else None)
            for axis in range(3):
                put('aux', '', f'torso_log_{axis}', e[:, axis]*deg, ej[:, axis]*deg if jac else None)
            if hasattr(p, 'fixed_goal'):
                dp = (pos-self.t(ref['fixed_position']))/.0005
                fixed = self.t(ref['fixed_rotation'])
                for i in range(7):
                    for axis in range(3):
                        put('fixed', '', f'position_{i}_{axis}', dp[:, i, axis], jp[:, i, axis]/.0005 if jac else None)
                    if i in (2, 5, 6):
                        e, ej = self.orientation(rot[:, i], fixed[:, i], jr[:, i] if jac else None)
                        for axis in range(3):
                            put('fixed', '', f'rotation_{i}_{axis}', e[:, axis]/np.deg2rad(.1), ej[:, axis]/np.deg2rad(.1) if jac else None)
            keys = list(values)
            # One host transfer for all metric values/Jacobians at this rate.
            packed = torch.stack([torch.cat([values[k][:, None], grads[k]], -1) if jac else values[k][:, None] for k in keys], 1).cpu().numpy()
            for i, key in enumerate(keys):
                mask = ref['arms'][key[1]]['straight'] if key[2] == 'elbow_flex_error_when_human_straight_deg' else np.ones(len(qs), bool)
                metrics[(rate, *key)] = (packed[mask, i, 0], packed[mask, i, 1:] if jac else None,
                                        ref['l'][mask], ref['r'][mask], ref['a'][mask])
        observed = []
        for c in p.checks:
            v = metrics[(c['fps'], c['section'], c.get('side', ''), c['metric'])][0]
            if not len(v):
                continue
            value = float(v.mean() if c['stat'] == 'mean' else np.percentile(v, 95) if c['stat'] == 'p95' else v.max())
            scale = max(c['g1'], .0005 if c['metric'] == 'normalized_hand_error' else .05)
            observed.append(dict(check=c, value=value, scale=scale, over=max(0., (value-c['g1']-1e-6)/scale)))
        over = np.array([o['over'] for o in observed])
        speed = float((abs(np.diff(q[:, 7:], axis=0))*120/p.speeds).max())
        bounds = max(float((p.lo-x).max()), float((x-p.hi).max()), 0.)
        score = dict(cost=float(over@over+10*over.max()**2), failed=int((over>0).sum()),
                     max_relative_excess=float(over.max()), speed_ratio=speed, bounds=bounds,
                     physical=speed <= 1+1e-6 and bounds <= 1e-8)
        self.calls += 1
        self.seconds += time.perf_counter()-begin
        return score, metrics, observed

    def foot_poses(self, model, q):
        p, r = self.poses(q, model.bodies)
        if not hasattr(model, '_cuda_local'):
            qi = np.asarray(q)[0]
            model.d.qpos[:] = qi
            model.mj.mj_kinematics(model.m, model.d)
            local = []
            for body, geom in zip(model.bodies, model.geom):
                corners = (model.corner_signs*model.m.geom_size[geom])@model.d.geom_xmat[geom].reshape(3, 3).T+model.d.geom_xpos[geom]
                local.append((corners-model.d.xpos[body])@model.d.xmat[body].reshape(3, 3))
            model._cuda_local = self.t(np.array(local))
        corners = torch.einsum('nsij,skj->nski', r, model._cuda_local)+p[:, :, None, :]
        return p.cpu().numpy(), r.cpu().numpy(), corners[..., 2].amin(-1).cpu().numpy()

    def foot_linear(self, p, x, jac=True):
        q = p.assemble(x)
        times = np.array([o['time'] for o in p.obs])
        qs = self.batch.F.sample(q, times)
        if jac:
            pos, rot, jp, jr = self.poses(qs, p.fm.bodies, p.js.tolist())
            jp = torch.cat([self.s.eye.expand(len(qs), 2, 3, 3), jp], -1)
            jr = torch.cat([torch.zeros((len(qs), 2, 3, 3), device=self.device, dtype=torch.float64), jr], -1)
        else:
            pos, rot = self.poses(qs, p.fm.bodies)
            jp = jr = None
        goal_p = self.t(np.array([o['p'] for o in p.obs]))
        goal_r = self.t(np.array([o['rot'] for o in p.obs]))
        e, ej = self.orientation(rot, goal_r, jr)
        # CPU residual layout is [left position, left rotation, right position,
        # right rotation], followed by contact terms for grounded target sides.
        residual = torch.cat([(pos-goal_p)/.005, e/np.deg2rad(1.)], -1).flatten(1)
        jacobian = torch.cat([jp/.005, ej/np.deg2rad(1.)], -2).flatten(1, 2) if jac else None
        offsets = torch.einsum('nsij,skj->nski', rot, self.t(p.local))
        heights = pos[:, :, None, 2]+offsets[..., 2]
        h_jac = jp[:, :, None, 2, :]-(self.skew(offsets)@jr[:, :, None])[:, :, :, 2, :] if jac else None
        corner = heights.argmin(-1)
        contact_residual = (heights.amin(-1)-(p.floor+.0001))/.0005
        contact_jac = h_jac.gather(2, corner[:, :, None, None].expand(-1, -1, 1, p.d)).squeeze(2)/.0005 if jac else None
        if jac:
            data = torch.cat([torch.cat([residual[:, :, None], jacobian], -1).flatten(1), heights.flatten(1),
                              h_jac.flatten(1), contact_residual, contact_jac.flatten(1)], -1).cpu().numpy()
            offset = 12*(p.d+1)
            rr = data[:, :offset].reshape(len(qs), 12, p.d+1)
            hh = data[:, offset:offset+16].reshape(-1, 2, 8)
            hj = data[:, offset+16:offset+16+16*p.d].reshape(-1, 2, 8, p.d)
            cr = data[:, offset+16+16*p.d:offset+18+16*p.d]
            cj = data[:, offset+18+16*p.d:].reshape(-1, 2, p.d)
        else:
            data = torch.cat([residual, heights.flatten(1), contact_residual], -1).cpu().numpy()
            rr, hh, cr = data[:, :12], data[:, 12:28].reshape(-1, 2, 8), data[:, 28:30]
        out = []
        for i, o in enumerate(p.obs):
            grounded = np.flatnonzero(o['goal_lift'] > 1e-8)
            er = np.r_[rr[i, :, 0] if jac else rr[i], cr[i, grounded]]*o['weight']
            jj = np.vstack([rr[i, :, 1:], cj[i, grounded]])*o['weight'] if jac else None
            out.append((o, er, jj, hh[i], hj[i] if jac else None))
        return out

    def foot_measure(self, p, x, dense=True):
        cost = sum(float(e@e) for _, e, _, _, _ in self.foot_linear(p, x, False))
        added = ((x-p.x0)/p.scales).ravel()
        cost += .1*float(np.sum((p.root_smooth@added)**2))+.00001*float(added@added)
        times = np.arange((p.n-1)*5+1)/600 if dense else p.old
        q = p.assemble(x)
        _, _, height = self.foot_poses(p.fm, self.batch.F.sample(q, times))
        _, _, lift = p.goals(times)
        ix = np.unravel_index(height.argmin(), height.shape)
        ratio = float((abs(np.diff(q[:, 7:], axis=0))*120/p.speeds).max())
        excess = max(float((p.low-x).max()), float((x-p.high).max()), 0.)
        return dict(cost=cost, grounded_target_height_max_mm=float(np.where(lift > 1e-8, height, -np.inf).max()*1000),
                    min_clearance_mm=float(height.min()*1000), speed_ratio=ratio, box_violation=excess,
                    feasible=bool(height.min() >= p.floor-1e-7 and ratio <= 1+1e-6 and excess <= 1e-8),
                    ground_peak=dict(sample=int(ix[0]), rate=600 if dense else 120, side=int(ix[1])),
                    dense_violations=[float(times[k]) for k in np.flatnonzero(height.min(1) < p.floor+1e-5)[:64]])


def install(batch, device):
    geometry = Geometry(batch, device)
    U, F = batch.U, batch.F
    # Keep originals available for CPU/GPU numerical equivalence tests.
    geometry.cpu_upper = U.Problem.evaluate
    geometry.cpu_poses = F.FootModel.poses
    geometry.cpu_linear = F.SourceFootProblem.linearize
    geometry.cpu_measure = F.SourceFootProblem.measure
    U.Problem.evaluate = torch.inference_mode()(lambda p, x, jac=False: geometry.upper(p, x, jac))
    F.FootModel.poses = torch.inference_mode()(lambda p, q: geometry.foot_poses(p, q))
    F.SourceFootProblem.linearize = torch.inference_mode()(lambda p, x: geometry.foot_linear(p, x, True))
    F.SourceFootProblem.measure = torch.inference_mode()(lambda p, x, dense=True: geometry.foot_measure(p, x, dense))
    return geometry
