from collections import namedtuple

import numpy as np

from .ffi import ffi, lib

TYPE_MAP = {
    'float': np.dtype('f4'),
    'double': np.dtype('f8'),
    'int': np.dtype('i4'),
    'long': np.dtype('i8')
}

Array = namedtuple("Array", "addr l1 l2 l3")
Array.__new__.__defaults__ = (None, 0, 0, 0)


def to_c_array(data):
    """
    Convert a numpy 2D array to C pointer
    """
    if data is None:
        return Array(addr=ffi.NULL, l1=0, l2=0, l3=0)
    arr = ffi.cast("double*", data.ctypes.data)
    zero = ffi.cast("size_t", 0)
    l1, l2, l3 = ffi.cast("size_t", data.shape[0]), zero, zero
    if len(data.shape) > 1:
        l2 = ffi.cast("size_t", data.shape[1])
    if len(data.shape) > 2:
        l3 = ffi.cast("size_t", data.shape[2])
    return Array(addr=arr, l1=l1, l2=l2, l3=l3)


def to_managed_array(ptr):
    """
    Convert a C pointer to a numpy 2D array and ensure it will be freed when
    array is garbage collected
    """
    length = ptr.l1
    shape = (ptr.l1,)
    if hasattr(ptr, "l2") and ptr.l2 > 0:
        length *= ptr.l2
        shape = (ptr.l1, ptr.l2)
    if hasattr(ptr, "l3") and ptr.l3 > 0:
        length *= ptr.l3
        shape = (ptr.l1, ptr.l2, ptr.l3)
    _type = get_type(ptr.addr)
    gc_data = finalize(ptr, _type)
    arr = np.frombuffer(
        ffi.buffer(gc_data, length * ffi.sizeof(_type)), TYPE_MAP[_type]
    )
    arr.shape = shape
    return arr


def finalize(ptr, _type):
    """
    Wrap a given pointer in a managed object in order to free it
    """
    if _type == 'int' or _type == 'long':
        gc_data = ffi.gc(ptr.addr, lib.FreeIntArray)
    else:
        gc_data = ffi.gc(ptr.addr, lib.FreeRealArray)
    return gc_data


def get_type(cdata):
    """
    Get the C type of the given cdata
    """
    _type = ffi.getctype(ffi.typeof(cdata).item)
    if _type not in TYPE_MAP:
        raise RuntimeError("unknown type: {}".format(_type))
    return _type


def initializer(name):
    """
    Convert a string to a CFFI initializer enum type
    """
    if hasattr(name, 'descr'):
        return lib.I_OC, name.descr
    elif name in ['random', 'rand']:
        return lib.I_RANDOM, 0
    elif name == 'given':
        return lib.I_GIVEN, 0
    elif name in ['kmeans_pp', 'kmeans++']:
        return lib.I_KMEANS_PP, 0


def space(name):
    """
    Convert a string to a CFFI space enum type
    """
    return getattr(lib, 'S_{0}'.format(name.upper()))


def oc(name):
    """
    Convert a string to a CFFI oc enum type
    """
    return getattr(lib, 'O_{0}'.format(name.upper()))


def figure(name):
    """
    Convert a string to a CFFI oc enum type
    """
    return getattr(lib, 'F_{0}'.format(name.upper()))


def par(p):
    """
    Convert a boolean parallel indicator to an int for CFFI binding
    """
    return 1 if p else 0


def none2zero(s):
    """
    Convert a None value to 0 for CFFI binding
    """
    return 0 if s is None else s


def handle_error(err):
    """
    Raise an error from a CFFI message string
    """
    if err:
        raise RuntimeError(ffi.string(err))
