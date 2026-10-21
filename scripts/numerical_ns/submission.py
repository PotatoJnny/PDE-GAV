"""SDK candidate: finish a classical numerical solve before seeing test points."""
import json
import os
from pathlib import Path
import socket
import subprocess
from collections.abc import Mapping
import numpy as np
from pde_agent_verifier_solver.sdk import AnswerView, QueryKind, open_public_case_session
from wire import send, receive


def plain(x):
    if isinstance(x, Mapping):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


class Prepared:
    def __init__(self, sock, process, log):
        self.sock, self.process, self.log = sock, process, log

    def close(self):
        try:
            self.sock.close()
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.log.close()

    def evaluate(self, query):
        points = np.array([[p['t'], p['x'], p['y']] for p in query.points])
        try:
            send(self.sock, {'op': 'evaluate'}, points)
            h, values = receive(self.sock)
            if h['op'] != 'answer':
                raise RuntimeError(h.get('error', 'numerical evaluation failed'))
            if values.shape != (len(points), 2) or not np.isfinite(values).all():
                raise ValueError('invalid numerical answer')
            return AnswerView(QueryKind.POINTS,
                {'velocity_x': tuple(values[:, 0]), 'velocity_y': tuple(values[:, 1])},
                points=query.points, shape=(len(points),))
        finally:
            self.close()


def prepare_solution(case, solve_budget=None):
    root = Path(__file__).resolve().parent
    profile = json.loads((root / 'profile.json').read_text())
    left, right = socket.socketpair()
    env = os.environ.copy()
    # Each isolated runtime supplies its own native dependencies.
    for name in ('PYTHONPATH', 'VIRTUAL_ENV', 'LD_PRELOAD'):
        env.pop(name, None)
    if profile['method'] == 'fem':
        env.pop('LD_LIBRARY_PATH', None)
        env['PATH'] = str(Path(profile['python']).parent) + ':' + env.get('PATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    log = open('numerical_backend.log', 'wb')
    # Inherit the Controller's process group: cancellation cleans all descendants.
    process = subprocess.Popen([profile['python'], str(root/'backend.py'), str(right.fileno())],
        pass_fds=(right.fileno(),), env=env, stdout=log, stderr=subprocess.STDOUT)
    right.close()
    prepared = Prepared(left, process, log)
    public = plain(case.public_data)
    # Deliberate allowlist: no case identity, private bank or hidden query in child.
    metadata = {k: public[k] for k in ('domain', 'parameters', 'boundary_condition')}
    session = open_public_case_session(case)
    methods = {'I': session.query_initial_velocity, 'F': session.query_forcing}
    if hasattr(session, 'query_boundary_velocity'):
        methods['B'] = session.query_boundary_velocity
    try:
        send(left, {'op': 'prepare', 'metadata': metadata, 'profile': profile})
        while True:
            h, points = receive(left)
            if h['op'] == 'ready':
                Path('numerical_diagnostics.json').write_text(json.dumps(h.get('diagnostics', {}), indent=2))
                return prepared
            if h['op'] == 'error':
                raise RuntimeError(h['error'])
            if h['op'] not in methods:
                raise ValueError('forbidden numerical input request')
            # Service Oracle requests on the metered candidate thread. Controller
            # queue credit and the single upstream solve budget remain authoritative.
            results = []
            for start in range(0, len(points), 4096):
                results.append(np.asarray(methods[h['op']](points[start:start+4096]), dtype=float))
            send(left, {'op': 'input'}, np.concatenate(results, axis=0))
    except BaseException:
        prepared.close()
        raise
