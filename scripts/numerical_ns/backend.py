"""Numerical process has only an explicit public-input callback, never a Case."""
import json
import socket
import sys
import threading
import traceback
import numpy as np
from wire import send, receive


class Inputs:
    def __init__(self, sock, metadata):
        self.sock, self.metadata = sock, metadata
        self.lock = threading.Lock()
        self.counts = {'I': 0, 'F': 0, 'B': 0}

    def query(self, op, points):
        points = np.asarray(points, dtype=float)
        if op == 'B' and self.metadata['boundary_condition']['kind'] == 'no_slip_velocity':
            return np.zeros((len(points), 2))
        with self.lock:
            self.counts[op] += len(points)
            send(self.sock, {'op': op}, points)
            h, values = receive(self.sock)
            if h['op'] != 'input' or values.shape != (len(points), 2) or not np.isfinite(values).all():
                raise ValueError('invalid public input')
            return values


def main():
    sock = socket.socket(fileno=int(sys.argv[1]))
    try:
        h, _ = receive(sock)
        assert h['op'] == 'prepare'
        inputs = Inputs(sock, h['metadata'])
        method = h['profile']['method']
        if method == 'mac':
            from mac import solve
        elif method == 'fem':
            from fem import solve
        elif method == 'nektar':
            from nektar import solve
        else:
            raise ValueError(method)
        solution = solve(inputs, h['profile'])
        send(sock, {'op': 'ready', 'diagnostics': {**solution.diagnostics, 'input_points': inputs.counts}})
        h, points = receive(sock)
        assert h['op'] == 'evaluate'
        values = solution.evaluate(points)
        solution.close()
        send(sock, {'op': 'answer'}, values)
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        try:
            send(sock, {'op': 'error', 'error': error})
        except OSError:
            pass
        raise
    finally:
        sock.close()


if __name__ == '__main__':
    main()
