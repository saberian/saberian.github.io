"""Run the shared evaluator driver in a restricted local Docker container."""

import json
import os
import selectors
import subprocess
import time
import uuid

from evaluator import DRIVER, SUPERVISOR

DEFAULT_IMAGE = "python:3.12.10-slim-bookworm"


def image_identity(image=DEFAULT_IMAGE):
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True, timeout=10))[0]
    return {"id": info["Id"], "digests": info.get("RepoDigests", []), "architecture": info["Architecture"], "os": info["Os"]}


def bounded_process(command, input_bytes, timeout=30, output_limit=131072):
    """Bound host-side output buffering as well as execution time."""
    if len(input_bytes) > 262144:
        raise ValueError("Execution request exceeds 256 KiB")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    pending = memoryview(input_bytes)
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream, event, name in ((process.stdin, selectors.EVENT_WRITE, "stdin"),
                    (process.stdout, selectors.EVENT_READ, "stdout"), (process.stderr, selectors.EVENT_READ, "stderr")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, event, name)
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Docker execution transport timed out; sample unscored")
                for key, _ in selector.select(timeout=0.1):
                    stream, name = key.fileobj, key.data
                    try:
                        if name == "stdin":
                            count = os.write(stream.fileno(), pending[:8192])
                            pending = pending[count:]
                            if not pending:
                                selector.unregister(stream)
                                stream.close()
                        else:
                            chunk = os.read(stream.fileno(), 8192)
                            if not chunk:
                                selector.unregister(stream)
                                stream.close()
                                continue
                            streams[name].extend(chunk)
                            if sum(map(len, streams.values())) > output_limit:
                                raise RuntimeError("Execution transport output limit exceeded; sample unscored")
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        selector.unregister(stream)
                        stream.close()
            code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        return code, *(streams[name].decode(errors="replace") for name in ("stdout", "stderr"))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def run_docker(request, image):
    name = "coding-eval-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user=65534:65534", "--memory=512m", "--memory-swap=512m", "--cpus=1",
        "--pids-limit=32", "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
        "--log-driver=none", "-i", image, "python", "-I", "-c", SUPERVISOR, DRIVER]
    started = time.monotonic()
    try:
        code, stdout, stderr = bounded_process(command, json.dumps(request).encode())
        if code:
            raise RuntimeError(f"Docker/supervisor failed ({code}); sample unscored: {stderr[:500]}")
        try:
            payload = json.loads(stdout)
            if not isinstance(payload, dict):
                raise ValueError("Expected a supervisor object")
        except ValueError as exc:
            raise RuntimeError("Invalid supervisor output; sample unscored") from exc
        return payload, {"runtime": "docker", "container_name": name, "evaluation_seconds": time.monotonic() - started}
    finally:
        cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=10)
        if cleanup.returncode and "No such container" not in cleanup.stderr:
            raise RuntimeError(f"Could not confirm container cleanup: {cleanup.stderr[:500]}")
