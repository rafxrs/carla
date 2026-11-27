"""Windows launchers must not leave benchmark JVM/agent descendants alive."""

import ctypes
import os
import subprocess
import sys

import pytest
from scripts.run_jsettlers_benchmark import terminate_process_tree


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher process-tree regression")
def test_referee_timeout_terminates_grandchildren():
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = None
    try:
        child_pid = int(process.stdout.readline())
        handle = kernel.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE
        assert handle, ctypes.get_last_error()
        terminate_process_tree(process)
        assert kernel.WaitForSingleObject(handle, 5000) == 0  # WAIT_OBJECT_0
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            terminate_process_tree(process)
        process.stdout.close()
        if handle:
            kernel.CloseHandle(handle)
