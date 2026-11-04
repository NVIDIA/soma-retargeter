"""Share the unchanged CUDA geometry implementation across CPU motion workers."""
from pathlib import Path
from types import SimpleNamespace
from multiprocessing.connection import Client, Listener
import json
import os
import queue
import threading
import time
import traceback
import numpy as np


def prepare_compiled_model(batch, directory):
    """Serialize the identical compiled model once, avoiding XML compiler heaps."""
    import ctypes
    import gc
    import mujoco
    xml = batch.BUNDLE/'code/gear_sonic/data/assets/robot_description/mjcf/h2.xml'
    binary = Path(directory)/'h2.mjb'
    original = mujoco.MjModel.from_xml_path(str(xml))
    mujoco.mj_saveModel(original, str(binary))
    loaded = mujoco.MjModel.from_binary_path(str(binary))
    arrays = scalars = 0
    for name in dir(original):
        # mjSpec compilation bookkeeping is not part of serialized mjModel.
        if name.startswith('_') or name == 'signature':
            continue
        left, right = getattr(original, name), getattr(loaded, name)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right, equal_nan=True), 'Serialized model differs: '+name
            arrays += 1
        elif isinstance(left, (str, int, float)):
            assert left == right, 'Serialized model differs: '+name
            scalars += 1
    for section in ('opt', 'stat'):
        a, b = getattr(original, section), getattr(loaded, section)
        for name in dir(a):
            if name.startswith('_'):
                continue
            left, right = getattr(a, name), getattr(b, name)
            if isinstance(left, np.ndarray):
                assert np.array_equal(left, right, equal_nan=True), section+'.'+name
                arrays += 1
            elif isinstance(left, (str, int, float)):
                assert left == right, section+'.'+name
                scalars += 1
    record = dict(xml=str(xml), xml_sha256=batch.sha(xml), binary=str(binary),
                  binary_sha256=batch.sha(binary), mujoco_version=mujoco.__version__,
                  verified_arrays=arrays, verified_scalars=scalars,
                  binary_size=binary.stat().st_size, binary_mtime_ns=binary.stat().st_mtime_ns)
    binary.chmod(0o400)
    path = Path(directory)/'compiled_model.json'
    path.write_text(json.dumps(record)); path.chmod(0o600)
    os.environ['H2_COMPILED_MODEL_CONFIG'] = str(path)
    del original, loaded
    gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
    print(json.dumps(dict(stage='compiled_model_verified', **{k: record[k] for k in
        ('xml_sha256', 'binary_sha256', 'verified_arrays', 'verified_scalars')})), flush=True)
    return record


def install_compiled_model():
    import mujoco
    record = json.loads(Path(os.environ['H2_COMPILED_MODEL_CONFIG']).read_text())
    assert mujoco.__version__ == record['mujoco_version']
    binary = Path(record['binary'])
    assert (binary.stat().st_size, binary.stat().st_mtime_ns) == (
        record['binary_size'], record['binary_mtime_ns'])
    original = mujoco.MjModel.from_xml_path
    target = Path(record['xml']).resolve()
    def load(path, *args, **kwargs):
        if not args and not kwargs and Path(path).resolve() == target:
            return mujoco.MjModel.from_binary_path(str(binary))
        return original(path, *args, **kwargs)
    mujoco.MjModel.from_xml_path = staticmethod(load)


class GeometryClient:
    def __init__(self, batch, endpoint, authkey, device):
        self.batch, self.device = batch, device
        self.connection = Client(endpoint, authkey=bytes.fromhex(authkey))
        self.calls = 0
        self.seconds = 0.

    def request(self, operation, **data):
        self.connection.send(dict(operation=operation, **data))
        answer = self.connection.recv()
        if 'error' in answer:
            raise RuntimeError('Shared CUDA geometry: '+answer['error'])
        return answer['result']

    def upper(self, p, x, jac=False):
        begin = time.perf_counter()
        state = {name: getattr(p, name) for name in
                 ('refs', 'js', 'arm_bodies', 'torso', 'checks', 'speeds', 'lo', 'hi')}
        if hasattr(p, 'fixed_goal'):
            state['fixed_goal'] = True
        answer = self.request('upper', state=state, q=p.assemble(x), x=x, jac=jac)
        self.calls += 1
        self.seconds += time.perf_counter()-begin
        return answer

    def foot_poses(self, model, q):
        if not hasattr(model, '_shared_local'):
            model.d.qpos[:] = np.asarray(q)[0]
            model.mj.mj_kinematics(model.m, model.d)
            local = []
            for body, geom in zip(model.bodies, model.geom):
                corners = ((model.corner_signs*model.m.geom_size[geom])
                           @model.d.geom_xmat[geom].reshape(3, 3).T
                           +model.d.geom_xpos[geom])
                local.append((corners-model.d.xpos[body])
                             @model.d.xmat[body].reshape(3, 3))
            model._shared_local = np.array(local)
        return self.request('foot_poses', q=q, bodies=model.bodies, local=model._shared_local)

    def foot_linear(self, p, x, jac=True):
        state = {name: getattr(p, name) for name in ('obs', 'js', 'local', 'floor', 'd')}
        return self.request('foot_linear', state=state, q=p.assemble(x),
                            bodies=p.fm.bodies, x=x, jac=jac)

    def foot_measure(self, p, x, dense=True):
        # Reuse the exact existing composition of CUDA geometry and CPU gates.
        from gpu_acceleration import Geometry
        return Geometry.foot_measure(self, p, x, dense)


def install(batch, endpoint, authkey, device):
    geometry = GeometryClient(batch, endpoint, authkey, device)
    U, F = batch.U, batch.F
    geometry.cpu_upper = U.Problem.evaluate
    geometry.cpu_poses = F.FootModel.poses
    geometry.cpu_linear = F.SourceFootProblem.linearize
    geometry.cpu_measure = F.SourceFootProblem.measure
    U.Problem.evaluate = lambda p, x, jac=False: geometry.upper(p, x, jac)
    F.FootModel.poses = lambda p, q: geometry.foot_poses(p, q)
    F.SourceFootProblem.linearize = lambda p, x: geometry.foot_linear(p, x, True)
    F.SourceFootProblem.measure = lambda p, x, dense=True: geometry.foot_measure(p, x, dense)
    return geometry


def serve(config, device):
    import torch
    import batch_reopt as B
    from gpu_acceleration import Geometry
    if 'H2_COMPILED_MODEL_CONFIG' in os.environ:
        install_compiled_model()
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    B.worker_init()
    geometry = Geometry(B, 'cuda:'+str(device))
    entry = next(r for r in config['actors'] if r['device'] == device)
    listener = Listener(entry['endpoint'], authkey=bytes.fromhex(config['authkey']))
    os.chmod(entry['endpoint'], 0o600)
    incoming = queue.Queue()

    def read(connection):
        try:
            while True:
                incoming.put((connection, connection.recv()))
        except (EOFError, OSError):
            connection.close()

    def accept():
        while True:
            try:
                connection = listener.accept()
            except OSError:
                return
            threading.Thread(target=read, args=(connection,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    Path(entry['ready']).write_text(json.dumps(dict(pid=os.getpid(), device=device)))
    print(json.dumps(dict(stage='shared_geometry_ready', device=device, pid=os.getpid())), flush=True)
    while True:
        connection, request = incoming.get()
        op = request['operation']
        if op == 'stop':
            connection.send(dict(result='stopped'))
            break
        try:
            with torch.inference_mode():
                if op == 'foot_poses':
                    model = SimpleNamespace(bodies=request['bodies'],
                                            _cuda_local=geometry.t(request['local']))
                    result = geometry.foot_poses(model, request['q'])
                else:
                    p = SimpleNamespace(**request['state'])
                    p.assemble = lambda x: request['q']
                    if op == 'upper':
                        result = geometry.upper(p, request['x'], request['jac'])
                    elif op == 'foot_linear':
                        p.fm = SimpleNamespace(bodies=request['bodies'])
                        result = geometry.foot_linear(p, request['x'], request['jac'])
                    else:
                        raise ValueError('Unknown geometry operation: '+op)
            answer = dict(result=result)
        except Exception as error:
            traceback.print_exc()
            answer = dict(error=f'{type(error).__name__}: {error}')
        try:
            connection.send(answer)
        except (OSError, EOFError):
            pass
    listener.close()


def stop(endpoint, authkey):
    with Client(endpoint, authkey=bytes.fromhex(authkey)) as connection:
        connection.send(dict(operation='stop'))
        return connection.recv()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--device', type=int, required=True)
    args = ap.parse_args()
    serve(json.loads(args.config.read_text()), args.device)
