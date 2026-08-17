import ctypes
import math
from pathlib import Path


LIB_PATH = Path(__file__).resolve().parent.parent/"build"/"libvindex.so"

print(LIB_PATH)

lib = ctypes.CDLL(str(LIB_PATH))

lib.add.argtypes = [ctypes.c_float, ctypes.c_float]

lib.add.restype = ctypes.c_float

result = lib.add(2.5, 3.0)
print(f"result is {result}")

assert math.isclose(5.5, result), f"C boundary broken: got {result}, expected 5.5"