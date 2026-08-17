import ctypes
import numpy as np
from pathlib import Path

LIB_PATH = Path(__file__).resolve().parent.parent/"build"/"libvindex.so"
lib = ctypes.CDLL(str(LIB_PATH))

lib.scale_array.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_float]

lib.scale_array.restype = None

def scale(arr: np.ndarray, factor:float) -> None:
    lib.scale_array(arr, arr.size, factor)

if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    print(a)
    print(a.ctypes.data)
    scale(a, 3.0)
    print(a)
    print(a.ctypes.data)

    assert np.allclose(a, [3.0, 6.0, 9.0, 12.0]), "error"