"""Array RPC shared by the isolated numerical backend and its SDK adapter."""
import json
import struct
import numpy as np


def read_exact(sock, n):
    chunks = bytearray()
    while len(chunks) < n:
        part = sock.recv(n - len(chunks))
        if not part:
            raise EOFError("numerical backend disconnected")
        chunks.extend(part)
    return bytes(chunks)


def send(sock, metadata, array=None):
    h = dict(metadata)
    if array is not None:
        array = np.ascontiguousarray(array, dtype='<f8')
        h['shape'] = list(array.shape)
    data = json.dumps(h, allow_nan=False).encode()
    sock.sendall(struct.pack('<I', len(data)) + data)
    if array is not None:
        sock.sendall(array.tobytes())


def receive(sock):
    n = struct.unpack('<I', read_exact(sock, 4))[0]
    if n > 4_000_000:
        raise ValueError('invalid RPC header size')
    h = json.loads(read_exact(sock, n))
    a = None
    if 'shape' in h:
        count = int(np.prod(h['shape']))
        if not 0 <= count <= 50_000_000:
            raise ValueError('invalid array size')
        a = np.frombuffer(read_exact(sock, count * 8), dtype='<f8').reshape(h['shape']).copy()
    return h, a
