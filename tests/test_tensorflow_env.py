import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge._tensorflow_env import (
    configure_tensorflow_cuda_runtime,
    find_tensorflow_cuda_library_dirs,
)


def test_tensorflow_cuda_library_dirs_discovered():
    lib_dirs = find_tensorflow_cuda_library_dirs()

    assert lib_dirs
    assert any("site-packages/nvidia/" in path for path in lib_dirs)


def test_tensorflow_cuda_runtime_updates_ld_library_path(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    lib_dirs = configure_tensorflow_cuda_runtime()
    ld_library_path = __import__("os").environ.get("LD_LIBRARY_PATH", "")

    assert lib_dirs
    for path in lib_dirs:
        assert path in ld_library_path
