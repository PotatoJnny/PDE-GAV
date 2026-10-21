"""Optional buffered event recording for a measured run; no arrays or credentials."""
import json
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread
import time


class ExecutionTrace:
    def __init__(self, path):
        self.path = Path(path)
        self._queue = Queue()
        self._failure = None
        self._closed = False
        self._thread = Thread(target=self._write, name="execution-trace", daemon=True)
        self._thread.start()

    def record(self, event, **fields):
        if self._closed:
            raise RuntimeError("execution trace is closed")
        if self._failure is not None:
            raise RuntimeError("execution trace writer failed") from self._failure
        self._queue.put({"event": event, "time_unix": time.time(),
                         "time_monotonic": time.monotonic(), **fields})

    def _write(self):
        try:
            with self.path.open("a", buffering=262144) as stream:
                last_flush = time.monotonic()
                while True:
                    try:
                        item = self._queue.get(timeout=.5)
                    except Empty:
                        stream.flush()
                        continue
                    if item is None:
                        break
                    if isinstance(item, Event):
                        stream.flush()
                        item.set()
                    else:
                        stream.write(json.dumps(item, separators=(",", ":"))+"\n")
                    if time.monotonic()-last_flush >= .5:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.flush()
        except BaseException as error:
            self._failure = error

    def flush(self):
        marker = Event()
        self._queue.put(marker)
        while not marker.wait(.1):
            if self._failure is not None:
                raise RuntimeError("execution trace writer failed") from self._failure
            if not self._thread.is_alive():
                raise RuntimeError("execution trace writer stopped")

    def close(self):
        if not self._closed:
            self.flush()
            self._closed = True
            self._queue.put(None)
            self._thread.join()
