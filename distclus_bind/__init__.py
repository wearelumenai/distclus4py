import numpy as np

from bind import ffi, lib

TYPE_MAP = {}

for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2 ** log_bytes))
        dtype = '%s%d' % (prefix[0], 2 ** log_bytes)
        TYPE_MAP[ctype] = np.dtype(dtype)

TYPE_MAP['float'] = np.dtype('f4')
TYPE_MAP['double'] = np.dtype('f8')


def as_array(ptr, l):
    _type = ffi.getctype(ffi.typeof(ptr).item)
    if _type not in TYPE_MAP:
        raise RuntimeError("Cannot create an array for element type: %s" % _type)
    return np.frombuffer(ffi.buffer(ptr, l * ffi.sizeof(_type)), TYPE_MAP[_type])


def as_array2(ptr, l1, l2):
    arr = as_array(ptr, l1 * l2)
    arr.shape = (l1, l2)
    return arr


def kmeans(data, k, iter):
    arr = ffi.cast("double*", data.ctypes.data)
    l1 = ffi.cast("size_t", data.shape[0])
    l2 = ffi.cast("size_t", data.shape[1])
    res = lib.Kmeans(arr, l1, l2, k, iter)
    centers = as_array2(res.data, res.l1, res.l2)
    return centers
