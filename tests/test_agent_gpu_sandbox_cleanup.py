"""CPU-only bounded subprocess tests; no model, CUDA, credentials, or host changes."""
import ctypes
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


SPEC = importlib.util.spec_from_file_location(
    'gpu_sandbox_cleanup',
    Path(__file__).resolve().parents[1] / 'scripts' / 'qualify_agent_gpu_sandbox.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-tree qualification')


TREE = r'''
import ctypes, os, signal, subprocess, sys, time
from pathlib import Path
role, topology, behavior = sys.argv[1:]
Path(role + '.pid').write_text(str(os.getpid()))
if role == 'leaf':
    if topology == 'separate-session':
        parent = os.getppid()
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
        if os.getppid() != parent:
            raise SystemExit(1)
    else:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path('leaf.ready').write_text('ready')
else:
    if role == 'cli':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    child_role = 'helper' if role == 'cli' else 'leaf'
    subprocess.Popen([sys.executable, '-u', __file__, child_role, topology, behavior])
    deadline = time.monotonic() + 5
    while not Path('leaf.ready').exists():
        if time.monotonic() >= deadline:
            raise SystemExit(2)
        time.sleep(0.01)
    if role == 'cli':
        print('captured timeout stdout', flush=True)
        print('captured timeout stderr', file=sys.stderr, flush=True)
        if behavior == 'cli-exits':
            raise SystemExit(0)
while True:
    time.sleep(30)
'''


@pytest.fixture
def reap_own_test_children(tmp_path):
    # Adoption is local to this pytest process, not a host setting. It lets the
    # test reap its own deliberately orphaned fixture descendants, including a
    # separate-session leaf coupled to helper death like the pinned bwrap path.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    try:
        yield
    finally:
        for path in tmp_path.glob('*.pid'):
            pid = int(path.read_text())
            try:
                child, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if child == 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def run(tmp_path, command, *, timeout=2):
    return MODULE._run_sandbox_command(
        command, workspace=tmp_path, environment=dict(os.environ),
        output=tmp_path, timeout=timeout)


def test_normal_command_captures_and_reaps_private_group(tmp_path, monkeypatch):
    spawned = []
    original = subprocess.Popen
    def track(*args, **kwargs):
        assert kwargs['start_new_session'] is True
        process = original(*args, **kwargs)
        spawned.append(process)
        return process
    monkeypatch.setattr(MODULE.subprocess, 'Popen', track)
    result = run(tmp_path, [sys.executable, '-c',
        "import sys; print('stdout marker'); print('stderr marker', file=sys.stderr)"])
    assert result.returncode == 0
    assert result.stdout == (tmp_path / 'stdout.log').read_text() == 'stdout marker\n'
    assert result.stderr == (tmp_path / 'stderr.log').read_text() == 'stderr marker\n'
    assert spawned[0].returncode == 0
    with pytest.raises(ChildProcessError):
        os.waitpid(spawned[0].pid, os.WNOHANG)


@pytest.mark.parametrize('topology', ['same-group', 'separate-session'])
@pytest.mark.parametrize('behavior', ['cli-running', 'cli-exits'])
def test_timeout_cleans_helper_and_inherited_pipes(
        tmp_path, reap_own_test_children, topology, behavior):
    tree = tmp_path / 'tree.py'
    tree.write_text(TREE)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run(tmp_path, [sys.executable, '-u', str(tree), 'cli', topology, behavior], timeout=0.8)
    assert time.monotonic() - started < 7
    assert caught.value.output == (tmp_path / 'stdout.log').read_text() == 'captured timeout stdout\n'
    assert caught.value.stderr == (tmp_path / 'stderr.log').read_text() == 'captured timeout stderr\n'
    assert (tmp_path / 'leaf.ready').exists()
    cli = int((tmp_path / 'cli.pid').read_text())
    with pytest.raises(ChildProcessError):
        os.waitpid(cli, os.WNOHANG)
    # Reap precisely the fixture's adopted descendants; successful reaping
    # proves they are terminated, not merely absent from the original group.
    for role in ('helper', 'leaf'):
        pid = int((tmp_path / (role + '.pid')).read_text())
        deadline = time.monotonic() + 2
        while True:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
            if reaped == pid:
                break
            assert time.monotonic() < deadline, f'{role} survived cleanup'
            time.sleep(0.01)


def test_unclosed_pipe_failure_is_bounded_and_retains_partial_output(tmp_path, monkeypatch):
    class StuckCapture:
        stdout = io.StringIO()
        stderr = io.StringIO()
        calls = []
        def communicate(self, *, timeout):
            self.calls.append(timeout)
            raise subprocess.TimeoutExpired(['fixture'], timeout,
                                            output=b'partial stdout', stderr=b'partial stderr')
    process = StuckCapture()
    stopped = []
    monkeypatch.setattr(MODULE.subprocess, 'Popen', lambda *a, **k: process)
    monkeypatch.setattr(MODULE, '_stop_process_group', lambda child: stopped.append(child))
    with pytest.raises(RuntimeError, match='capture pipes remained open after cleanup'):
        run(tmp_path, ['fixture'], timeout=0.1)
    assert process.calls == [0.1, 2]
    assert stopped == [process, process]
    assert process.stdout.closed and process.stderr.closed
    assert (tmp_path / 'stdout.log').read_text() == 'partial stdout'
    assert (tmp_path / 'stderr.log').read_text() == 'partial stderr'
