"""The execution worker must not re-import its caller."""

import json
import subprocess
import sys


def test_worker_does_not_import_caller(tmp_path):
    caller = tmp_path / "caller.py"
    caller.write_text(
        "import multiprocessing\n"
        "from rl_lab.tools.exec import make_python_tool\n"
        "if __name__ != '__main__':\n"
        "    raise RuntimeError('Worker re-imported caller')\n"
        "multiprocessing.set_start_method('spawn')\n"
        "print(make_python_tool(timeout=0.5)(\"print('hello')\"))\n"
    )
    process = subprocess.run([sys.executable, str(caller)], capture_output=True, text=True, timeout=5)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["success"] and result["stdout"] == "hello\n"
