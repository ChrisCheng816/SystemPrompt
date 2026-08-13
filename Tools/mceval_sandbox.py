"""Host-side client for the persistent McEval Docker sandbox."""

from __future__ import annotations

import json
import subprocess


class DockerSandbox:
    def __init__(self, docker_prefix: list[str], image: str, timeout: int):
        command = [
            *docker_prefix, "run", "--rm", "-i", "--network", "none", "--read-only",
            "--tmpfs", "/work:rw,nosuid,nodev,noexec,uid=10001,gid=10001,size=512m",
            "--pids-limit", "64", "--memory", "2g", "--cpus", "1", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", image,
        ]
        self.timeout = timeout
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def execute(self, source: str, language: str) -> dict:
        if self.process.stdin is None or self.process.stdout is None:
            return self._error("sandbox stdin/stdout is unavailable")
        try:
            self.process.stdin.write(json.dumps({"source": source, "language": language, "timeout": self.timeout}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError:
            line = ""
        if not line:
            detail = self.process.stderr.read().strip() if self.process.stderr is not None else ""
            return self._error(f"sandbox worker exited without a response: {detail}")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return self._error(f"invalid sandbox response: {line.strip()}")

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    @staticmethod
    def _error(message: str) -> dict:
        return {"is_pass": False, "stage": "infrastructure", "stdout": "", "stderr": message,
                "return_code": None, "timeout": False}
