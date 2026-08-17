import numpy as np
import ctypes
from pathlib import Path

lib = ctypes.CDLL(str(Path(__file__).resolve().parent/"libscale.so"))

lib.square_all.argtypes = [np.ctypeslib.ndpointer(dtype = np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int]

lib.square_all.restype = None      # void — returns nothing

arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
print("before:", arr)

# Pass the array itself. ndpointer handles taking the address.
# arr.size is the element count — C needs it because a pointer carries no length.
lib.square_all(arr, arr.size)

print("after: ", arr)  