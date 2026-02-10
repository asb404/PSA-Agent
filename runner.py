import subprocess, os, tempfile, time, json, sys
import shutil
from typing import Dict, Any

def _preexec_set_limits(cpu_time: int = None, memory_bytes: int = None):
    try:
        import resource
        if cpu_time is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            desired = min(cpu_time, hard if hard != resource.RLIM_INFINITY else cpu_time)
            resource.setrlimit(resource.RLIMIT_CPU, (desired, desired))
        if memory_bytes is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if hard != resource.RLIM_INFINITY:
                desired_mem = min(memory_bytes, hard)
            else:
                desired_mem = memory_bytes
            resource.setrlimit(resource.RLIMIT_AS, (desired_mem, desired_mem))
    except Exception as e:
        print(f'warning: could not set rlimits: {e}', file=sys.stderr)


def run_pytest_in_sandbox(code_dir: str, timeout: int = 20, cpu_time: int = 5, memory_bytes: int = 500_000_000) -> Dict[str, Any]:
    """
    Run pytest in `code_dir` with timeout and OS-level resource limits.
    Returns a dict with keys: returncode, stdout, stderr, duration.
    """
    cmd = [sys.executable, '-m', 'pytest', '-q', '--disable-warnings', '--maxfail=1']
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True, timeout=timeout,
                              preexec_fn=lambda: _preexec_set_limits(cpu_time=cpu_time, memory_bytes=memory_bytes))
        duration = time.time() - start
        return {
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'duration': duration
        }
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        return {
            'returncode': -1,
            'stdout': e.stdout or '',
            'stderr': f'timeout after {timeout}s',
            'duration': duration
        }
    except Exception as e:
        duration = time.time() - start
        return {
            'returncode': -2,
            'stdout': '',
            'stderr': f'error: {e}',
            'duration': duration
        }
